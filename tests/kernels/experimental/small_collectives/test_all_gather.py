# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The small all-gather kernel on the 8 cores of one host, bitwise against the
blocks laid back to back and against ``lax.all_gather(tiled=True)``, with the
core's index read from the axis or handed in as data, on a one-axis mesh and
on a (lane, axis) mesh like the DCP one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_gather as ag

from .oracle import AXIS, N, custom_call_operands_in_vmem

pytestmark = pytest.mark.multichip

LANE = "lane"


def _blocks(n, size, seed):
    """Distinct int32 words per core (high bits = core), incl. the -inf bit
    pattern the DCP scores carry."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-(2**31), 2**31 - 1, size=(n, size), dtype=np.int64).astype(
        np.int32
    )
    x[:, ::7] = np.int32(np.array(-np.inf, np.float32).view(np.int32))
    return x


def _run(mesh, body, per_core, ranks, lanes=None):
    """``body(x, rank, lane)`` per core over every mesh axis; returns [cores, out]."""
    axes = mesh.axis_names
    spec = P(axes)
    sharding = NamedSharding(mesh, spec)
    n, size = per_core.shape
    x = jax.device_put(jnp.asarray(per_core.reshape(-1)), sharding)
    r = jax.device_put(jnp.asarray(ranks, jnp.int32), sharding)
    lane = jax.device_put(
        jnp.asarray(lanes if lanes is not None else np.zeros(n), jnp.int32), sharding
    )
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    out = np.asarray(fn(x, r, lane))
    return out.reshape(n, -1)


@pytest.mark.parametrize("size", [2048, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_blocks_back_to_back(mesh, size, rank_operand):
    per_core = _blocks(N, size, seed=size)

    def body(x, r, _):
        return ag.small_all_gather(
            x, num_cores=N, axis=AXIS, rank=r if rank_operand else None
        )

    got = _run(mesh, body, per_core, range(N))
    want = ag.reference(per_core)
    for core in range(N):
        np.testing.assert_array_equal(got[core], want)


def test_matches_xla_all_gather(mesh):
    per_core = _blocks(N, 8192, seed=3)
    got = _run(
        mesh,
        lambda x, r, _: ag.small_all_gather(x, num_cores=N, axis=AXIS, rank=r),
        per_core,
        range(N),
    )
    xla = _run(
        mesh, lambda x, r, _: lax.all_gather(x, AXIS, tiled=True), per_core, range(N)
    )
    np.testing.assert_array_equal(got, xla)


def test_float_bits_pass_through(mesh):
    """f32 blocks (NaN payloads, -0.0, -inf) come back bit for bit."""
    bits = _blocks(N, 2048, seed=5)
    per_core = bits.view(np.float32)
    got = _run(
        mesh,
        lambda x, r, _: ag.small_all_gather(x, num_cores=N, axis=AXIS, rank=r),
        per_core,
        range(N),
    )
    np.testing.assert_array_equal(got.view(np.int32), np.tile(bits.reshape(-1), (N, 1)))


def test_repeated_calls_in_one_program(mesh):
    """Back-to-back calls (one per idx layer) share the barrier semaphore."""
    per_core = _blocks(N, 8192, seed=9)

    def body(x, r, _):
        a = ag.small_all_gather(x, num_cores=N, axis=AXIS, rank=r)
        b = ag.small_all_gather(x + 1, num_cores=N, axis=AXIS, rank=r)
        c = ag.small_all_gather(
            a[: x.shape[0]] ^ b[-x.shape[0] :], num_cores=N, axis=AXIS, rank=r
        )
        return jnp.concatenate([a, b, c])

    got = _run(mesh, body, per_core, range(N))
    a = per_core.reshape(-1)
    b = (per_core + 1).reshape(-1)
    c = np.tile((per_core[0] ^ (per_core[N - 1] + 1))[None], (N, 1)).reshape(-1)
    want = np.concatenate([a, b, c])
    for core in range(N):
        np.testing.assert_array_equal(got[core], want)


@pytest.mark.parametrize("lane_data", [False, True])
def test_lane_mesh_gathers_within_each_lane(lane_data):
    """(2 lanes, 4 cores): each lane is its own group, as DCP's lanes are."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    lanes, n = 2, N // 2
    mesh = Mesh(np.array(jax.devices()[:N]).reshape(lanes, n), (LANE, AXIS))
    per_core = _blocks(N, 8192, seed=21)
    ranks = np.tile(np.arange(n), lanes)
    lane_ids = np.repeat(np.arange(lanes), n)

    def body(x, r, lane):
        return ag.small_all_gather(
            x,
            num_cores=n,
            axis=AXIS,
            rank=r,
            lane_axis=LANE,
            lane=lane if lane_data else None,
        )

    got = _run(mesh, body, per_core, ranks, lane_ids)
    for core in range(N):
        lane = core // n
        want = per_core[lane * n : (lane + 1) * n].reshape(-1)
        np.testing.assert_array_equal(got[core], want)


def test_a_wrong_rank_order_fails(mesh):
    """Positive control: cores 0 and 1 told each other's index write into each
    other's slots, and the comparison has to see it."""
    per_core = _blocks(N, 8192, seed=11)
    got = _run(
        mesh,
        lambda x, r, _: ag.small_all_gather(x, num_cores=N, axis=AXIS, rank=r),
        per_core,
        [1, 0, *range(2, N)],
    )
    want = ag.reference(per_core)
    assert (got[2] != want).any()


def test_rejects_a_2d_block():
    with pytest.raises(ValueError):
        ag.small_all_gather(jnp.zeros((4, 128), jnp.int32), num_cores=8, axis=AXIS)


def test_rejects_a_block_of_partial_tiles():
    with pytest.raises(ValueError):
        ag.small_all_gather(jnp.zeros((1000,), jnp.int32), num_cores=8, axis=AXIS)


@pytest.mark.parametrize("size", [8192, 32768])
def test_hbm_operand_is_bitwise_and_not_staged_in_vmem(mesh, size):
    """`hbm_operand=True` (TPU_SMALL_COLL_HBM_OPERAND): the blocks back to back
    when the block comes out of a preceding op (as the SparseCore top-k's
    scores do), and the compiled call reads it from HBM -- no copy-start/-done
    into VMEM in front of it."""
    blocks = _blocks(N, size, seed=size + 5)

    def body(x, r, _):
        return ag.small_all_gather(
            x ^ jnp.int32(0x5A5A), num_cores=N, axis=AXIS, rank=r, hbm_operand=True
        )

    got = _run(mesh, body, blocks, range(N))
    want = ag.reference(blocks ^ np.int32(0x5A5A))
    for core in range(N):
        np.testing.assert_array_equal(got[core], want)

    spec = P(mesh.axis_names)
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    args = (
        jax.ShapeDtypeStruct((N * size,), jnp.int32),
        jax.ShapeDtypeStruct((N,), jnp.int32),
        jax.ShapeDtypeStruct((N,), jnp.int32),
    )
    lowered = fn.lower(*args)
    assert "input_memory_space_colors" in lowered.as_text()
    staged = custom_call_operands_in_vmem(
        lowered.compile().as_text(), "small_all_gather"
    )
    assert staged and not any(staged.values()), staged
