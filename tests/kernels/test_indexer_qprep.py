# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_INDEXER_QPREP_FUSED: the fused indexer q prep is bit-identical to the
torch path it replaces (``VllmTPUIndexer.forward``, half-split rope branch),
compiled and eager, and the top-k it feeds is identical.

  test_numerics_grid        instrument: every ``Numerics`` combination vs the
                            compiled torch path (prints; asserts the default
                            is one of the exact ones)
  test_bit_exact            the torch op the forward calls, T in decode and
                            mixed buckets, special rows: q8 bytes, scale bits
                            and the folded weights bits, 0 mismatches
  test_topk_identical       streamindex_topk on both paths' (q8, weights)
  test_fault_control        each fault is caught by the bit-exact comparator,
                            and the head-swap fault also moves the top-k
"""

import functools
import itertools
import os
import sys

import pytest

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels import indexer_qprep as K  # noqa: E402
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as M  # noqa: E402
from vllm_torchtpu.layers.core.quantization import quantize_tensor  # noqa: E402

pytestmark = pytest.mark.tpu_test

H, D, R = 32, 128, 64  # GLM-5.2 indexer: index_n_heads, index_head_dim, rope
SOFTMAX_SCALE = D**-0.5
MAX_POS = 1 << 19
FP8 = torch.float8_e4m3fn

# The forward builds the op at rebind, outside Dynamo; do the same here.
for _m in (0, 128):
    M._get_indexer_qprep_op(H, R, _m)


class _Rope:
    """What `rope_half_split_leading` reads off a RotaryEmbedding: the cos|sin
    cache in the query dtype (vLLM's GPT-J table, base 10000)."""

    def __init__(self):
        inv = 1.0 / (10000 ** (torch.arange(0, R, 2, dtype=torch.float32) / R))
        f = torch.outer(torch.arange(MAX_POS, dtype=torch.float32), inv)
        self.cache = torch.cat([f.cos(), f.sin()], -1).to(torch.bfloat16).to("tpu")

    def _match_cos_sin_cache_dtype(self, x):
        return self.cache


def _inputs(t, seed, kind="normal"):
    g = torch.Generator().manual_seed(seed)
    mag = 10 ** (torch.rand(t, H, 1, generator=g) * 4 - 2)  # 1e-2 .. 1e2 per row
    q = torch.randn(t, H, D, generator=g) * mag
    if kind == "special":
        q[0, 3] = 0  # all-zero (token, head) row -> scale 0
        q[min(1, t - 1), 5] *= 1e3  # huge row
        q[t // 2, :, :R] = 0  # rope lanes zero
        q[t - 1, 7, 0] = 3e4  # single spike
    q = q.reshape(t, H * D).to(torch.bfloat16)
    pos = torch.randint(0, MAX_POS, (t,), generator=g, dtype=torch.int32)
    w = torch.randn(t, H, generator=g).to(torch.bfloat16)
    return q.to("tpu"), pos.to("tpu"), w.to("tpu")


def _fold(w, q_scale):
    """`VllmTPUIndexer._finish_forward`'s weight fold, verbatim."""
    wd = w.dtype
    w = w.unsqueeze(-1) * q_scale * SOFTMAX_SCALE * H**-0.5
    return w.squeeze(-1).to(wd)


def unfused(q, pos, w, rope):
    """The flag-off lines of `VllmTPUIndexer.forward`, verbatim."""
    q = q.view(-1, H, D)
    q = M.rope_half_split_leading(rope, pos, q, R)
    q = q.view(-1, D)
    q_fp8, q_scale = quantize_tensor(q, FP8, axis=-1, use_ue8m0=False)
    q_fp8 = q_fp8.view(-1, H, D)
    q_scale = q_scale.view(-1, H, 1)
    return q_fp8, q_scale, _fold(w, q_scale)


def fused(q, pos, w, rope, min_tokens=0):
    """The flag-on branch of `VllmTPUIndexer.forward`."""
    op = M._get_indexer_qprep_op(H, R, min_tokens)
    c, s = M.rope_half_split_leading_cs(rope, pos, q, D, R)
    q_fp8, q_scale = op(q.reshape(-1, H * D), c, s)
    q_scale = q_scale.view(-1, H, 1)
    return q_fp8, q_scale, _fold(w, q_scale)


def _compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, dynamic=False)


def _mismatch(a, b):
    """(q8 bytes, scale bits, weights bits) mismatch counts."""
    out = []
    for x, y in zip(a, b):
        x, y = x.cpu(), y.cpu()
        assert x.shape == y.shape, (x.shape, y.shape)
        iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()]
        out.append(int((x.view(iv) != y.view(iv)).sum()))
    return tuple(out)


def _kernel_jax(q, pos, rope, numerics, variant="f32"):
    """Run the kernel directly (no torch op) for the numerics grid (the f32
    body is the one that takes every rounding switch)."""
    c, s = M.rope_half_split_leading_cs(rope, pos, q, D, R)
    to_j = lambda t: jnp.asarray(t.cpu().float().numpy()).astype(jnp.bfloat16)  # noqa: E731
    q8, sc = K.indexer_q_prep(
        to_j(q),
        to_j(c),
        to_j(s),
        n_head=H,
        rope_dim=R,
        numerics=numerics,
        variant=variant,
    )
    q8 = torch.from_numpy(np.asarray(q8.view(jnp.uint8))).view(FP8)
    sc = torch.from_numpy(np.asarray(sc)).view(-1, H, 1)
    return q8, sc


def test_numerics_grid(vllm_config_context):
    rope = _Rope()
    q, pos, w = _inputs(1024, 7)
    ref = _compiled(unfused)(q, pos, w, rope)
    eager = unfused(q, pos, w, rope)
    print("NUMERICS eager-vs-compiled", _mismatch(eager[:2], ref[:2]))
    exact = []
    for rb, yb, sm, ib, pb in itertools.product(
        (False, True),
        (False, True),
        ("b16r16", "b16r32", "f32"),
        (False, True),
        (False, True),
    ):
        nm = K.Numerics(
            rope_bf16=rb, y_bf16=yb, scale_mode=sm, inv_bf16=ib, prod_bf16=pb
        )
        got = _kernel_jax(q, pos, rope, nm)
        mm = _mismatch(got, ref[:2])
        mme = _mismatch(got, eager[:2])
        print(f"NUMERICS {nm.tag} vs_compiled={mm} vs_eager={mme}")
        if mm == (0, 0):
            exact.append(nm.tag)
    print("NUMERICS exact:", exact, "default:", K.DEFAULT_NUMERICS.tag)
    assert K.DEFAULT_NUMERICS.tag in exact


CASES = [
    (t, k) for t in (4, 8, 16, 64, 1024, 2048, 4096) for k in ("normal", "special")
]


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("t,kind", CASES)
def test_bit_exact(vllm_config_context, t, kind, compiled):
    rope = _Rope()
    q, pos, w = _inputs(t, 100 + t, kind)
    # Like with like: the weight fold itself rounds differently eager vs
    # compiled (unrelated to this change), so eager is held against eager.
    ref = (_compiled(unfused) if compiled else unfused)(q, pos, w, rope)
    f = _compiled(fused) if compiled else fused
    got = f(q, pos, w, rope)
    mm = _mismatch(got, ref)
    print(f"BITEXACT T={t} kind={kind} compiled={compiled} q8/scale/weights={mm}")
    assert mm == (0, 0, 0)


@pytest.mark.parametrize("t", [4, 16, 64])
def test_min_tokens_xla_path_bit_exact(vllm_config_context, t):
    """TPU_INDEXER_QPREP_FUSED_MIN_TOKENS: buckets below it run `reference`
    (plain XLA) inside the same op; it must be exact too."""
    rope = _Rope()
    q, pos, w = _inputs(t, 300 + t, "special")
    ref = _compiled(unfused)(q, pos, w, rope)
    got = _compiled(functools.partial(fused, min_tokens=128))(q, pos, w, rope)
    mm = _mismatch(got, ref)
    print(f"MINTOK T={t} q8/scale/weights={mm}")
    assert mm == (0, 0, 0)


@pytest.mark.parametrize("variant", ["f32", "bf16", "tok"])
@pytest.mark.parametrize("t", [1024, 4096])
def test_variants_bit_exact(vllm_config_context, variant, t):
    """Every kernel body variant reproduces the compiled torch path."""
    rope = _Rope()
    q, pos, w = _inputs(t, 500 + t, "special")
    ref = _compiled(unfused)(q, pos, w, rope)
    c, s = M.rope_half_split_leading_cs(rope, pos, q, D, R)
    to_j = lambda x: jnp.asarray(x.cpu().float().numpy()).astype(jnp.bfloat16)  # noqa: E731
    q8, sc = K.indexer_q_prep(
        to_j(q), to_j(c), to_j(s), n_head=H, rope_dim=R, variant=variant
    )
    q8 = torch.from_numpy(np.asarray(q8.view(jnp.uint8))).view(FP8)
    sc = torch.from_numpy(np.asarray(sc)).view(-1, H, 1)
    mm = _mismatch((q8, sc), ref[:2])
    print(f"VARIANT {variant} T={t} q8/scale={mm}")
    assert mm == (0, 0)


# --- top-k identity -----------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))


def _topk(q8, wts, seed=0):
    from test_dsv4_streamindex_topk import (  # noqa: PLC0415
        _HAS,
        _chunk_call_kwargs,
        _chunk_inputs,
        _fragmented_pages,
    )

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

    t = q8.shape[0]
    T_list, S_list = (t,), (2048,)
    inputs = _chunk_inputs(T_list, _fragmented_pages(1, pages_per_seq=16), 128, H, D)
    kw = _chunk_call_kwargs(inputs, T_list, S_list, 512, 1, 8, 2, _HAS, None)
    kw["q"] = jnp.asarray(q8.cpu().view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    kw["indexer_weights"] = jnp.asarray(wts.cpu().float().numpy()).astype(jnp.bfloat16)
    return np.asarray(streamindex_topk(**kw))


def test_topk_identical(vllm_config_context):
    rope = _Rope()
    q, pos, w = _inputs(256, 11)
    pos = (torch.arange(256, dtype=torch.int32) + 2048 - 256).to(
        "tpu"
    )  # last 256 of 2048
    ref = _compiled(unfused)(q, pos, w, rope)
    got = _compiled(fused)(q, pos, w, rope)
    a, b = _topk(ref[0], ref[2]), _topk(got[0], got[2])
    n = int((a != b).sum())
    print(
        f"TOPK identical: mismatching index entries {n} of {a.size}; "
        f"valid={int((a >= 0).sum())}"
    )
    assert n == 0


# --- fault control --------------------------------------------------------------


def _fault(name, q, pos, w, rope):
    """The fused path with one injected fault."""
    if name == "head_swap":
        q8, sc, wt = fused(q, pos, w, rope)
        perm = list(range(H))
        perm[0], perm[1] = 1, 0
        q8 = q8.view(torch.uint8)[:, perm].view(FP8)
        return q8, sc, wt
    if name == "no_rope":
        op = M._get_indexer_qprep_op(H, R, 0)
        c, s = M.rope_half_split_leading_cs(rope, pos, q, D, R)
        q8, sc = op(q, torch.ones_like(c), torch.zeros_like(s))
        sc = sc.view(-1, H, 1)
        return q8, sc, _fold(w, sc)
    # Numerics faults (y_bf16=False is not one: with per-op rope rounding y is
    # already bf16, an equivalent formulation -- j-70160033).
    nm = {
        "inv_f32": dataclass_replace(K.DEFAULT_NUMERICS, inv_bf16=False),
        "scale_f32": K.Numerics(scale_mode="f32", inv_bf16=False, prod_bf16=False),
        "prod_f32": dataclass_replace(K.DEFAULT_NUMERICS, prod_bf16=False),
        "rope_f32": dataclass_replace(K.DEFAULT_NUMERICS, rope_bf16=False),
    }[name]
    q8, sc = _kernel_jax(q, pos, rope, nm, variant="f32")
    return q8, sc, None


def dataclass_replace(x, **kw):
    import dataclasses  # noqa: PLC0415

    return dataclasses.replace(x, **kw)


@pytest.mark.parametrize(
    "name", ["head_swap", "no_rope", "inv_f32", "scale_f32", "prod_f32", "rope_f32"]
)
def test_fault_control(vllm_config_context, name):
    rope = _Rope()
    q, pos, w = _inputs(1024, 21, "special")
    ref = _compiled(unfused)(q, pos, w, rope)
    bad = _fault(name, q, pos, w, rope)
    if bad[2] is None:  # numerics faults: the kernel's own outputs
        bad, ref = bad[:2], ref[:2]
    mm = _mismatch(bad, ref)
    print(f"FAULT {name} q8/scale/weights={mm} caught={sum(mm) > 0}")
    if name == "head_swap":
        q2, pos2, w2 = _inputs(256, 11)
        pos2 = (torch.arange(256, dtype=torch.int32) + 2048 - 256).to("tpu")
        r2 = _compiled(unfused)(q2, pos2, w2, rope)
        b2 = _fault(name, q2, pos2, w2, rope)
        n = int((_topk(r2[0], r2[2]) != _topk(b2[0], b2[2])).sum())
        print(f"FAULT head_swap topk mismatching entries {n}")
        assert n > 0
    assert sum(mm) > 0
