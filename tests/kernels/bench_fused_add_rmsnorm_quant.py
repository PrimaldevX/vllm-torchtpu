# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + HLO check for TPU_FUSED_ADD_RMSNORM_QUANT (not a test).

Per call, chained clock (each iteration's new residual is the next one's
input; every output is loop carry, so nothing is hoisted or dead), per-call
cost = least-squares slope over M in M_LIST, rep 0 held out, median of reps.

Arms, at T in {512, 1024, 2048, 4096} rows x 6144 (GLM-5.2 hidden), bf16:
  inc_nq    incumbent: Pallas fused_add_rms_norm -> XLA quantize_tensor (2 passes)
  fused_nq  fused norm+quant, no bf16 y written (q_a input of a non-indexer layer)
  fused_nqy fused norm+quant + bf16 y (indexer layers: wk_weights_proj reads y)
  inc_mm    inc_nq + xla_quantized_matmul to n=2624 (fused_qkv_a_proj) -- whole call
  fused_mm  fused_nq + xla_prequantized_matmul                       -- whole call

HLO: the ENTRY computation of inc_mm / fused_mm at T=4096 (one call, no loop),
with the ops that read bf16[T,6144] or write f8[T,6144] flagged.
"""

import functools
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels import fused_add_rmsnorm as K
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)

EPS = 1e-5
H = 6144
N_QA = 2624
FP8 = jnp.float8_e4m3fn
ROWS = [512, 1024, 2048, 4096]
M_LIST = [2, 6, 10]
REPS = 6


def step(arm, x, r, w, wq, ws):
    """One call of `arm`; returns (new_residual, *outputs)."""
    if arm in ("inc_nq", "inc_mm"):
        y, res = K.fused_add_rms_norm(x, r, w, eps=EPS)
        if arm == "inc_nq":
            q, s = quantize_tensor(y, FP8)
            return res, q, s
        return res, xla_quantized_matmul(y, wq, ws)
    emit_y = arm == "fused_nqy"
    y, q, s, res = K.fused_add_rms_norm_quant(x, r, w, eps=EPS, emit_y=emit_y)
    if arm == "fused_mm":
        return res, xla_prequantized_matmul(q, s, wq, ws)
    return (res, q, s, y) if emit_y else (res, q, s)


def chained(arm, m):
    def f(x, r, w, wq, ws):
        init = step(arm, x, r, w, wq, ws)

        def body(_, carry):
            return step(arm, x, carry[0], w, wq, ws)

        return jax.lax.fori_loop(1, m, body, init)

    return jax.jit(f)


def inputs(rows):
    k = jax.random.split(jax.random.PRNGKey(rows), 5)
    x = (jax.random.normal(k[0], (rows, H)) * 0.05).astype(jnp.bfloat16)
    r = (jax.random.normal(k[1], (rows, H)) * 4).astype(jnp.bfloat16)
    w = (1 + 0.2 * jax.random.normal(k[2], (H,))).astype(jnp.bfloat16)
    wq = (jax.random.normal(k[3], (H, N_QA)) * 64).astype(FP8)
    ws = jax.random.uniform(k[4], (N_QA,), jnp.float32, 1e-4, 1e-3)
    return x, r, w, wq, ws


def count_custom_calls(compiled):
    return len(re.findall(r"custom-call\(", compiled.as_text()))


def bench():
    arms = ["inc_nq", "fused_nq", "fused_nqy", "inc_mm", "fused_mm"]
    results = {}
    for rows in ROWS:
        args = inputs(rows)
        fns = {}
        for arm in arms:  # M1: compile everything before timing
            for m in M_LIST:
                t0 = time.perf_counter()
                c = chained(arm, m).lower(*args).compile()
                cc = count_custom_calls(c)
                fns[(arm, m)] = c
                print(
                    f"COMPILE rows={rows} arm={arm} M={m} "
                    f"{time.perf_counter() - t0:.1f}s"
                    f" custom_calls={cc}",
                    flush=True,
                )
        times = {k: [] for k in fns}
        for _ in range(REPS):  # interleaved arms, same chip
            for k, c in fns.items():
                jax.block_until_ready(args)
                t0 = time.perf_counter()
                jax.block_until_ready(c(*args))
                times[k].append(time.perf_counter() - t0)
        for arm in arms:
            meds = [statistics.median(times[(arm, m)][1:]) * 1e6 for m in M_LIST]
            mbar = statistics.mean(M_LIST)
            tbar = statistics.mean(meds)
            slope = sum((m - mbar) * (t - tbar) for m, t in zip(M_LIST, meds)) / sum(
                (m - mbar) ** 2 for m in M_LIST
            )
            spread = max(
                (max(times[(arm, m)][1:]) - min(times[(arm, m)][1:]))
                / statistics.median(times[(arm, m)][1:])
                for m in M_LIST
            )
            results[(rows, arm)] = slope
            print(
                f"BENCH rows={rows} arm={arm} per_call_us={slope:.2f} "
                f"intercept_us={tbar - slope * mbar:.1f} "
                f"T(M)={[round(t, 1) for t in meds]} rep_spread={spread * 100:.1f}%",
                flush=True,
            )
    print("SUMMARY rows | inc_nq fused_nq (d%) fused_nqy (d%) | inc_mm fused_mm (d%)")
    for rows in ROWS:

        def g(a, rows=rows):
            return results[(rows, a)]

        def d(a, b, g=g):
            return (g(b) - g(a)) / g(a) * 100

        print(
            f"SUMMARY {rows} | {g('inc_nq'):.1f} {g('fused_nq'):.1f} "
            f"({d('inc_nq', 'fused_nq'):+.1f}%) {g('fused_nqy'):.1f} "
            f"({d('inc_nq', 'fused_nqy'):+.1f}%) | {g('inc_mm'):.1f} "
            f"{g('fused_mm'):.1f} ({d('inc_mm', 'fused_mm'):+.1f}%)"
        )


def hlo():
    rows = 4096
    args = inputs(rows)
    pat_big = re.compile(rf"(bf16|f8e4m3fn)\[{rows},{H}\]")
    for arm in ("inc_mm", "fused_mm"):
        c = jax.jit(functools.partial(step, arm)).lower(*args).compile()
        text = c.as_text()
        entry = text[text.index("\nENTRY") :]
        entry = entry[: entry.index("\n}") + 2]
        print(f"HLO-ENTRY arm={arm} custom_calls={count_custom_calls(c)}")
        n_big = 0
        for line in entry.splitlines()[1:]:
            line = line.strip()
            if not line or line == "}":
                continue
            m = re.match(r"(ROOT )?(\S+) = (\S+) (\w[\w-]*)\(", line)
            name, shape, op = (
                (m.group(2), m.group(3), m.group(4)) if m else ("?", "?", "?")
            )
            big = bool(pat_big.search(line)) and op not in ("parameter",)
            n_big += big
            flag = "BIG " if big else "    "
            calls = re.search(r"calls=(\S+?)[,\s]", line)
            cn = re.search(r'custom_call_target="([^"]+)"', line)
            kn = re.search(r'"name":"([^"]+)"', line) or re.search(
                r"kernel_name=(\S+)", line
            )
            extra = (
                (calls.group(1) if calls else "")
                + (f" target={cn.group(1)}" if cn else "")
                + (f" kernel={kn.group(1)}" if kn else "")
            )
            print(f"HLO {arm} {flag}{op:14s} {name:40s} {shape[:60]:60s} {extra}")
        print(f"HLO-SUMMARY arm={arm} ops_touching_[{rows},{H}]={n_big}")
        # Fused computations that reduce or convert over [rows, H]
        abs_red = len(re.findall(rf"abs\(bf16\[{rows},{H}\]", text)) + len(
            re.findall(rf"bf16\[{rows},{H}\][^\n]*abs\(", text)
        )
        f8_conv = len(re.findall(rf"f8e4m3fn\[{rows},{H}\][^\n]* convert\(", text))
        print(
            f"HLO-COUNTS arm={arm} abs_on_bf16[{rows},{H}]={abs_red} "
            f"convert_to_f8[{rows},{H}]={f8_conv}"
        )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("devices:", jax.devices()[:1], "jax", jax.__version__)
    if what in ("hlo", "all"):
        hlo()
    if what in ("bench", "all"):
        bench()
