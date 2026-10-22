# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE decode kernel with GLM-5.2's MoE block:
the shared expert has no gate (shared_gate_weight None) and the routed
experts' output is scaled by routed_scale (GLM-5.2's routed_scaling_factor,
2.5); the shared expert is not. TP experts with GLM-5.2's expert count, top-k
and per-rank intermediate at TP16, through the kernel and through the op's
per-rank body (its kernel path and its GMM MoE fallback)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode,
    vmem_bytes,
)
from vllm_torchtpu.layers.adapter import fused_moe_decode as adapter

# GLM-5.2: 256 experts, top-8, routed and shared intermediate 128 per rank at
# TP16 (hidden 6144 there; smaller here).
N, H, INTER, INTER_S, E, TOPK = 8, 1024, 128, 128, 256, 8
ROUTED_SCALE = 2.5
F8 = jnp.float8_e4m3fn
HIGHEST = jax.lax.Precision.HIGHEST


@pytest.fixture(scope="module")
def setup():
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("requires 8 TPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)

    def gen(r):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(5), r[0, 0]), 4)

        def q(w, axis):
            s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
            return (w / s).astype(F8), s

        w1, w1_scale = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, w2_scale = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        shared_w1, shared_w1_scale = q(
            jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0
        )
        shared_w2, shared_w2_scale = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
        return (
            w1,
            w2,
            w1_scale[:, :, None, :],
            w2_scale[:, :, None, :],
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
        )

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    return mesh, sh, rank, weights


def _ffn(x, w1, w1_scale, w2, w2_scale):
    i = w2.shape[0]
    gu = jnp.dot(x, w1.astype(jnp.float32) * w1_scale, precision=HIGHEST)
    return jnp.dot(
        jax.nn.silu(gu[:, :i]) * gu[:, i:],
        w2.astype(jnp.float32) * w2_scale,
        precision=HIGHEST,
    )


def _reference(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    tw,
    ti,
    routed_scale=ROUTED_SCALE,
    shared_scale=1.0,
):
    """GLM-5.2's MoE block in f32: the ungated shared expert plus routed_scale
    times the router-weighted routed experts, summed over the ranks."""
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)
    # Every expert on every token, vectorized over the experts.
    gu = jnp.einsum(
        "th,ehf->etf", xf, w1.astype(jnp.float32) * w1_scale[:, 0], precision=HIGHEST
    )
    act = jax.nn.silu(gu[..., :INTER]) * gu[..., INTER:]
    y = jnp.einsum(
        "etf,efh->eth", act, w2.astype(jnp.float32) * w2_scale[:, 0], precision=HIGHEST
    )
    coeff = jnp.sum(
        jnp.where(
            ti[:, :, None] == jnp.arange(E), tw.astype(jnp.float32)[:, :, None], 0.0
        ),
        axis=1,
    )
    routed = jnp.einsum("te,eth->th", coeff, y, precision=HIGHEST)
    out = shared_scale * shared + routed_scale * routed
    return jax.lax.psum(out, "d").astype(x.dtype)


def _kernel(rank, x, *rest):
    """GLM-5.2's call: no shared-expert gate, the routed output scaled."""
    *weights, tw, ti = rest
    return fused_moe_decode(
        rank.reshape(-1),
        x,
        *weights,
        None,
        tw,
        ti,
        n_ranks=N,
        routed_scale=ROUTED_SCALE,
    )


def _op_body(rank, x, *rest, max_tokens):
    """The op's per-rank body as arming builds it for a GLM-5.2 layer: no gate
    (a placeholder operand), routed scale 2.5; the kernel up to `max_tokens`
    tokens, GMM MoE above."""
    *weights, tw, ti = rest
    return adapter._layer(
        jnp.concatenate([rank, rank * E], axis=1),  # [[mesh index, first expert]]
        x,
        *weights,
        jnp.zeros((1, 1), jnp.bfloat16),
        tw,
        ti,
        n_ranks=N,
        topk=TOPK,
        expert_parallel=False,
        max_tokens=max_tokens,
        gated=False,
        routed_scale=ROUTED_SCALE,
    )


_PATHS = {
    "kernel": _kernel,
    "op_kernel": functools.partial(_op_body, max_tokens=256),
    "op_gmm_fallback": functools.partial(_op_body, max_tokens=1),
}


@pytest.mark.parametrize("m", [8, 32, 64])
@pytest.mark.parametrize("path", list(_PATHS))
def test_ungated_scaled_matches_reference(setup, m, path):
    if path == "op_gmm_fallback" and m != 32:
        pytest.skip("one fallback case")
    mesh, sh, rank, weights = setup
    rng = np.random.default_rng(m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)]).astype(
        np.int32
    )
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[m - 3 :] = 0  # padded rows: zero routing weight, shared expert only
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )

    def run(fn):
        out = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

    def max_rel(a, ref):
        return (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()

    got = run(_PATHS[path])
    ref = run(_reference)
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations and a BF16 all-reduce against f32 math: a few percent per
    # row (the padded rows are the shared expert alone).
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive controls: a reference without the routed scale, or with the
    # shared expert at half weight (sigmoid(0), a gate), must fail the bound.
    unscaled = run(functools.partial(_reference, routed_scale=1.0))
    assert max_rel(got[0], unscaled[0]) > 0.08
    half_shared = run(functools.partial(_reference, shared_scale=0.5))
    assert max_rel(got[0], half_shared[0]) > 0.08


def test_vmem_bound_glm52_default_cap():
    """The arming gate's estimate for GLM-5.2's layer at TP16 (256 experts,
    hidden 6144, routed and shared intermediate 128): the default token cap
    (256) fits, 1024 does not."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    from jax.experimental.pallas import tpu as pltpu

    have = pltpu.get_tpu_info().vmem_capacity_bytes
    assert vmem_bytes(256, 6144, 128, 128, 256, 16, F8) <= have
    assert vmem_bytes(1024, 6144, 128, 128, 256, 16, F8) > have
