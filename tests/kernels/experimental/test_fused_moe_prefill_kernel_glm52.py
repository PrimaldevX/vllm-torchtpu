# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE prefill kernel with GLM-5.2's MoE block:
the shared expert has no gate (shared_gate_weight None) and the routed
experts' output is scaled by routed_scale (GLM-5.2's routed_scaling_factor,
2.5); the shared expert is not. TP experts: every rank holds all E with the
intermediate sliced, and the shared expert is a TP slice."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import fused_moe_prefill

pytestmark = pytest.mark.multichip

# GLM-5.2's per-rank intermediate at TP16 (128, routed and shared) and top-8.
N, H, INTER, INTER_S, E, TOPK = 8, 1024, 128, 128, 64, 8
ROUTED_SCALE = 2.5
F8 = jnp.float8_e4m3fn


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
            shared_w1_scale.reshape(-1),
            shared_w2,
            shared_w2_scale.reshape(-1),
        )

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    return mesh, sh, rank, weights


def _ffn(x, w1, w1_scale, w2, w2_scale):
    i = w2.shape[0]
    gu = jnp.dot(x, w1.astype(jnp.float32) * w1_scale.reshape(1, -1))
    return jnp.dot(
        jax.nn.silu(gu[:, :i]) * gu[:, i:],
        w2.astype(jnp.float32) * w2_scale.reshape(1, -1),
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
    out = shared_scale * _ffn(
        xf, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale
    )
    for e in range(E):
        coeff = jnp.sum(
            jnp.where(ti == e, tw.astype(jnp.float32), 0.0), 1, keepdims=True
        )
        out = out + routed_scale * coeff * _ffn(
            xf, w1[e], w1_scale[e], w2[e], w2_scale[e]
        )
    return jax.lax.psum(out, "d").astype(x.dtype)


def _ungated(rank, x, *rest):
    """GLM-5.2's call: no shared-expert gate, the routed output scaled."""
    *weights, tw, ti = rest
    return fused_moe_prefill(
        x, *weights, None, tw, ti, axis_name="d", routed_scale=ROUTED_SCALE
    )


def _inputs(sh, m):
    rng = np.random.default_rng(m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)]).astype(
        np.int32
    )
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[m - 5 :] = 0  # padded rows: zero routing weight, shared expert only
    return (
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )


def _run(mesh, fn, args, specs, m):
    out = jax.jit(
        jax.shard_map(fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False)
    )(*args)
    return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)


def _max_rel(a, ref):
    return (
        np.linalg.norm(a - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    ).max()


# 256: one combine chunk on the Pallas combine. 1000: a chunk the Pallas
# combine does not tile (XLA combine). 4096: two chunks, two all-reduces.
@pytest.mark.parametrize("m", [256, 1000, 4096])
def test_ungated_scaled_matches_reference(setup, m):
    mesh, sh, rank, weights = setup
    x, tw, ti = _inputs(sh, m)
    specs = (P("d"),) * 12

    def run(fn):
        return _run(mesh, fn, (rank, x, *weights, tw, ti), specs, m)

    got = run(_ungated)
    ref = run(_reference)
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations, BF16 slab rows and a BF16 all-reduce against f32 math:
    # a few percent per row (the padded rows are the shared expert alone).
    assert _max_rel(got[0], ref[0]) < 0.08
    # Positive controls: a reference without the routed scale, or with the
    # shared expert at half weight (sigmoid(0), a gate), must fail the bound.
    unscaled = run(functools.partial(_reference, routed_scale=1.0))
    assert _max_rel(got[0], unscaled[0]) > 0.08
    half_shared = run(functools.partial(_reference, shared_scale=0.5))
    assert _max_rel(got[0], half_shared[0]) > 0.08
