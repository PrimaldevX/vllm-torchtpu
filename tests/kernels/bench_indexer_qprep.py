# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + HLO check for TPU_INDEXER_QPREP_FUSED (not a test).

bench: per call, chained clock -- each iteration's scale feeds the next
       iteration's C rows through a [T, 1] add, so nothing is loop-invariant
       and every output is carry; per-call cost = least-squares slope over M in
       M_LIST, rep 0 held out, median of reps. Arms at T in TOKENS:
         inc    the torch expression in XLA ([T,H,D] view, cat-rope, bf16 quant)
         fused  kernels.indexer_qprep.indexer_q_prep
hlo:   the torch path itself (torch.compile on the tpu device), unfused vs
       fused, T=4096, read back from the XLA dump (XLA_FLAGS must carry
       --xla_dump_to=<dir> --xla_dump_hlo_as_text; pass <dir> as argv[2]).
       Prints the ENTRY ops of the module that produces f8e4m3fn[T,32,128]
       and counts copies / transposes / fusions that read or write the
       32 MB q (bf16 [T, 4096] in any shape).
"""

import glob
import os
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels import indexer_qprep as K

H, D, R = 32, 128, 64
TOKENS = [4, 16, 64, 1024, 2048, 4096]
M_LIST = [2, 6, 10]
REPS = 6
FP8 = jnp.float8_e4m3fn


def inc(q, c, s):
    t = q.shape[0]
    x = q.reshape(t, H, D)
    h = R // 2
    rot = jnp.concatenate([x[..., h:R], x[..., :h], x[..., R:]], -1)
    y = (x * c[:, None] + rot * s[:, None]).reshape(-1, D)
    amax = jnp.max(jnp.abs(y), -1, keepdims=True)
    scale = amax * jnp.bfloat16(1 / 448)
    inv = jnp.where(scale == 0, jnp.bfloat16(0), jnp.bfloat16(1) / scale)
    q8 = jnp.clip(y * inv, -448, 448).astype(FP8)
    return q8.reshape(t, H, D), scale.astype(jnp.float32).reshape(t, H)


def fused(q, c, s):
    return K.indexer_q_prep(q, c, s, n_head=H, rope_dim=R)


def chained(fn, m):
    def f(q, c, s):
        def body(_, carry):
            c_, _q8, sc = carry
            c_ = c_ + (sc[:, :1] * 0).astype(c_.dtype)
            q8, sc = fn(q, c_, s)
            return c_, q8, sc

        q8, sc = fn(q, c, s)
        return jax.lax.fori_loop(1, m, body, (c, q8, sc))

    return jax.jit(f)


def inputs(t):
    k = jax.random.split(jax.random.PRNGKey(t), 3)
    q = (jax.random.normal(k[0], (t, H * D)) * 3).astype(jnp.bfloat16)
    ang = jax.random.uniform(k[1], (t, R // 2)) * 6.28
    cos, sin = jnp.cos(ang).astype(jnp.bfloat16), jnp.sin(ang).astype(jnp.bfloat16)
    c = jnp.concatenate([cos, cos, jnp.ones((t, D - R), jnp.bfloat16)], -1)
    s = jnp.concatenate([-sin, sin, jnp.zeros((t, D - R), jnp.bfloat16)], -1)
    return q, c, s


def bench():
    arms = {"inc": inc, "fused": fused}
    for t in TOKENS:
        args = inputs(t)
        fns = {
            (a, m): chained(f, m).lower(*args).compile()
            for a, f in arms.items()
            for m in M_LIST
        }
        times = {k: [] for k in fns}
        for _ in range(REPS):
            for k, c in fns.items():
                jax.block_until_ready(args)
                t0 = time.perf_counter()
                jax.block_until_ready(c(*args))
                times[k].append(time.perf_counter() - t0)
        res = {}
        for a in arms:
            meds = [statistics.median(times[(a, m)][1:]) * 1e6 for m in M_LIST]
            mbar, tbar = statistics.mean(M_LIST), statistics.mean(meds)
            slope = sum((m - mbar) * (x - tbar) for m, x in zip(M_LIST, meds)) / sum(
                (m - mbar) ** 2 for m in M_LIST
            )
            res[a] = slope
            print(
                f"BENCH T={t} arm={a} per_call_us={slope:.2f} "
                f"T(M)={[round(x, 1) for x in meds]}",
                flush=True,
            )
        print(
            f"BENCH-SUMMARY T={t} inc={res['inc']:.2f} fused={res['fused']:.2f} "
            f"delta={res['fused'] - res['inc']:+.2f}us",
            flush=True,
        )


BIG = re.compile(
    r"(bf16\[4096,4096\]|bf16\[4096,32,128\]|bf16\[131072,1?,?128\]|bf16\[131072,128\]|"
    r"bf16\[4096,32,64\]|bf16\[4096,32,32\]|f8e4m3fn\[4096,32,128\]|bf16\[131072\]|"
    r"bf16\[131072,1,128\])"
)


def hlo(dump_dir):
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from test_indexer_qprep import _compiled, _inputs, _Rope, unfused  # noqa: PLC0415
    from test_indexer_qprep import fused as tfused
    from vllm.config import VllmConfig, set_current_vllm_config  # noqa: PLC0415

    with set_current_vllm_config(VllmConfig()):
        rope = _Rope()
        q, pos, w = _inputs(4096, 1)
        for name, fn in (("unfused", unfused), ("fused", tfused)):
            before = set(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")))
            out = _compiled(fn)(q, pos, w, rope)
            [o.cpu() for o in out]
            torch.tpu.synchronize() if hasattr(torch, "tpu") else None
            new = sorted(
                set(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")))
                - before
            )
            print(f"HLO arm={name} new_modules={len(new)}")
            for path in new:
                with open(path) as fh:
                    text = fh.read()
                if "f8e4m3fn[4096,32,128]" not in text:
                    continue
                entry = text[text.index("\nENTRY") :]
                entry = entry[: entry.index("\n}") + 2]
                n_big = 0
                kinds = {}
                for line in entry.splitlines()[1:]:
                    m = re.match(r"\s*(ROOT )?(\S+) = (\S+) (\w[\w-]*)\(", line)
                    if not m:
                        continue
                    nm, shape, op = m.group(2), m.group(3), m.group(4)
                    if op in (
                        "parameter",
                        "constant",
                        "get-tuple-element",
                        "tuple",
                        "bitcast",
                    ):
                        continue
                    big = bool(BIG.search(line))
                    n_big += big
                    kinds[op] = kinds.get(op, 0) + 1
                    kn = re.search(r'kernel_name=([\w.-]+)|"name":"([^"]+)"', line)
                    calls = re.search(r"calls=([\w.%-]+)", line)
                    extra = (calls.group(1) if calls else "") + (
                        f" kernel={kn.group(1) or kn.group(2)}" if kn else ""
                    )
                    print(
                        f"HLO {name} {'BIG ' if big else '    '}{op:14s} {nm:44s} "
                        f"{shape[:70]:70s} {extra}"
                    )
                print(
                    f"HLO-SUMMARY arm={name} module={os.path.basename(path)} "
                    f"ops_touching_q={n_big} kinds={kinds}"
                )


def variants():
    """Each kernel body variant x block size: bit-exact vs `reference`, and
    the chained per-call time."""
    for t in (1024, 2048, 4096):
        args = inputs(t)
        r8, rsc = K.reference(*args, n_head=H, rope_dim=R)
        fns, ok = {}, {}
        for v in ("f32", "bf16", "tok"):
            for bt in (128, 256, 512):
                fn = lambda q, c, s, v=v, bt=bt: K.indexer_q_prep(  # noqa: E731
                    q, c, s, n_head=H, rope_dim=R, variant=v, block_tokens=bt
                )
                try:
                    a8, asc = jax.jit(fn)(*args)
                    mm = (
                        int((a8.view(jnp.uint8) != r8.view(jnp.uint8)).sum()),
                        int((asc != rsc).sum()),
                    )
                    fns[(v, bt)] = {
                        m: chained(fn, m).lower(*args).compile() for m in M_LIST
                    }
                    ok[(v, bt)] = mm
                except Exception as e:  # noqa: BLE001
                    print(
                        f"VAR T={t} variant={v} block={bt} FAILED "
                        f"{type(e).__name__}: {str(e)[:300]}",
                        flush=True,
                    )
        times = {(k, m): [] for k in fns for m in M_LIST}
        for _ in range(REPS):
            for k, d in fns.items():
                for m, c in d.items():
                    jax.block_until_ready(args)
                    t0 = time.perf_counter()
                    jax.block_until_ready(c(*args))
                    times[(k, m)].append(time.perf_counter() - t0)
        for k in fns:
            meds = [statistics.median(times[(k, m)][1:]) * 1e6 for m in M_LIST]
            mbar, tbar = statistics.mean(M_LIST), statistics.mean(meds)
            slope = sum((m - mbar) * (x - tbar) for m, x in zip(M_LIST, meds)) / sum(
                (m - mbar) ** 2 for m in M_LIST
            )
            print(
                f"VAR T={t} variant={k[0]} block={k[1]} per_call_us={slope:.2f} "
                f"mismatch_q8/scale={ok[k]}",
                flush=True,
            )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bench"
    print("devices:", jax.devices()[:1], "jax", jax.__version__)
    if what == "hlo":
        hlo(sys.argv[2])
    elif what == "variants":
        variants()
    else:
        bench()
