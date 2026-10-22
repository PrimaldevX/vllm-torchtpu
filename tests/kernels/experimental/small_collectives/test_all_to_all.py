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
"""The small all-to-all kernel on the 8 cores of one host, bitwise against the
blocks transposed and against ``lax.all_to_all(tiled=True)`` on every row at
or past ``row0``, at the DCP merge's int16 send-block shapes, with the core's
index read from the axis or handed in as data, on a one-axis mesh and on a
(lane, axis) mesh like the DCP one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_to_all as a2a

from .oracle import AXIS, N, custom_call_operands_in_vmem

pytestmark = pytest.mark.multichip

LANE = "lane"
V = 512


def _blocks(n, block, seed, dtype=np.int16):
    """[sender, destination, block, V] distinct words per (sender, dest)."""
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    return rng.integers(
        info.min, info.max, size=(n, n, block, V), dtype=np.int64
    ).astype(dtype)


def _run(mesh, body, per_core, ranks, lanes=None):
    """``body(x, rank, lane)`` per core; ``per_core`` [cores, rows, V]."""
    axes = mesh.axis_names
    spec = P(axes)
    sharding = NamedSharding(mesh, spec)
    n = per_core.shape[0]
    x = jax.device_put(jnp.asarray(per_core.reshape(-1, per_core.shape[-1])), sharding)
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
    return out.reshape(n, -1, per_core.shape[-1])


def _rows_from(arr, block, row0):
    """[cores, n * block, V] -> rows [row0, block) of every block."""
    c = arr.shape[0]
    return arr.reshape(c, -1, block, arr.shape[-1])[:, :, row0:]


# The DCP merge's send blocks at buckets 4 / 8 / 16 (SEND_PAD_ROWS 16 + T * 4
# partial rows) and the row0 the merge uses (8) or none.
@pytest.mark.parametrize("block", [32, 48, 80])
@pytest.mark.parametrize("row0", [0, 8])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_blocks_transposed(mesh, block, row0, rank_operand):
    blocks = _blocks(N, block, seed=block + row0)

    def body(x, r, _):
        return a2a.small_all_to_all(
            x, num_cores=N, axis=AXIS, row0=row0, rank=r if rank_operand else None
        )

    got = _run(mesh, body, blocks.reshape(N, N * block, V), range(N))
    want = a2a.reference(blocks, row0)
    np.testing.assert_array_equal(
        _rows_from(got, block, row0), _rows_from(want, block, row0)
    )


@pytest.mark.parametrize("dtype", [np.int16, np.int32])
def test_matches_xla_all_to_all(mesh, dtype):
    block, row0 = 48, 8
    per_core = _blocks(N, block, seed=3, dtype=dtype).reshape(N, N * block, V)
    got = _run(
        mesh,
        lambda x, r, _: a2a.small_all_to_all(
            x, num_cores=N, axis=AXIS, row0=row0, rank=r
        ),
        per_core,
        range(N),
    )
    xla = _run(
        mesh,
        lambda x, r, _: lax.all_to_all(x, AXIS, 0, 0, tiled=True),
        per_core,
        range(N),
    )
    np.testing.assert_array_equal(
        _rows_from(got, block, row0), _rows_from(xla, block, row0)
    )


def test_repeated_calls_in_one_program(mesh):
    """Back-to-back calls (one per layer) share the barrier semaphore; the
    third depends on the first two."""
    block, row0 = 32, 8
    blocks = _blocks(N, block, seed=9)

    def body(x, r, _):
        f = lambda y: a2a.small_all_to_all(  # noqa: E731
            y, num_cores=N, axis=AXIS, row0=row0, rank=r
        )
        a = f(x)
        b = f(x + 1)
        c = f(a ^ b)
        return jnp.concatenate([a, b, c])

    got = _run(mesh, body, blocks.reshape(N, N * block, V), range(N))
    want_a = a2a.reference(blocks, row0)
    want_b = a2a.reference((blocks + 1).astype(np.int16), row0)
    # c: core d's slot s holds core s's block d of (a ^ b); rows >= row0 of
    # (a ^ b) are (blocks ^ blocks + 1) transposed, so transposing back gives
    # the original pairing.
    ab = (blocks ^ (blocks + 1).astype(np.int16)).astype(np.int16)
    want_c = ab.copy()
    want_c[:, :, :row0] = 0
    want_c = want_c.reshape(N, N * block, V)
    rows = N * block
    np.testing.assert_array_equal(
        _rows_from(got[:, :rows], block, row0), _rows_from(want_a, block, row0)
    )
    np.testing.assert_array_equal(
        _rows_from(got[:, rows : 2 * rows], block, row0),
        _rows_from(want_b, block, row0),
    )
    np.testing.assert_array_equal(
        _rows_from(got[:, 2 * rows :], block, row0), _rows_from(want_c, block, row0)
    )


@pytest.mark.parametrize("lane_data", [False, True])
def test_lane_mesh_exchanges_within_each_lane(lane_data):
    """(2 lanes, 4 cores): each lane is its own group, as DCP's lanes are."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    lanes, n = 2, N // 2
    block, row0 = 32, 8
    mesh = Mesh(np.array(jax.devices()[:N]).reshape(lanes, n), (LANE, AXIS))
    groups = [_blocks(n, block, seed=21 + g) for g in range(lanes)]
    per_core = np.concatenate([g.reshape(n, n * block, V) for g in groups])
    ranks = np.tile(np.arange(n), lanes)
    lane_ids = np.repeat(np.arange(lanes), n)

    def body(x, r, lane):
        return a2a.small_all_to_all(
            x,
            num_cores=n,
            axis=AXIS,
            row0=row0,
            rank=r,
            lane_axis=LANE,
            lane=lane if lane_data else None,
        )

    got = _run(mesh, body, per_core, ranks, lane_ids)
    want = np.concatenate([a2a.reference(g, row0) for g in groups])
    np.testing.assert_array_equal(
        _rows_from(got, block, row0), _rows_from(want, block, row0)
    )


def test_a_wrong_rank_order_fails(mesh):
    """Positive control: cores 0 and 1 told each other's index send each other
    the wrong blocks, and the comparison has to see it."""
    block, row0 = 32, 8
    blocks = _blocks(N, block, seed=11)
    got = _run(
        mesh,
        lambda x, r, _: a2a.small_all_to_all(
            x, num_cores=N, axis=AXIS, row0=row0, rank=r
        ),
        blocks.reshape(N, N * block, V),
        [1, 0, *range(2, N)],
    )
    want = a2a.reference(blocks, row0)
    assert (_rows_from(got, block, row0)[2] != _rows_from(want, block, row0)[2]).any()


def test_rejects_unaligned_rows():
    with pytest.raises(ValueError):
        a2a.small_all_to_all(
            jnp.zeros((8 * 32, V), jnp.int16), num_cores=8, axis=AXIS, row0=4
        )
    with pytest.raises(ValueError):
        a2a.small_all_to_all(jnp.zeros((8 * 36, V), jnp.int16), num_cores=8, axis=AXIS)


@pytest.mark.parametrize("block", [32, 48])
def test_hbm_operand_is_bitwise_and_not_staged_in_vmem(mesh, block):
    """`hbm_operand=True` (TPU_SMALL_COLL_HBM_OPERAND): same rows as the
    reference when the send blocks come out of a preceding op (as the MLA
    kernel's send buffer does), and the compiled call reads them from HBM --
    no copy-start/-done into VMEM in front of it."""
    row0 = 8
    blocks = _blocks(N, block, seed=11 + block)

    def body(x, r, _):
        return a2a.small_all_to_all(
            x + jnp.int16(1),
            num_cores=N,
            axis=AXIS,
            row0=row0,
            rank=r,
            hbm_operand=True,
        )

    got = _run(mesh, body, blocks.reshape(N, N * block, V), range(N))
    want = a2a.reference((blocks + np.int16(1)).astype(np.int16), row0)
    np.testing.assert_array_equal(
        _rows_from(got, block, row0), _rows_from(want, block, row0)
    )

    spec = P(mesh.axis_names)
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    args = (
        jax.ShapeDtypeStruct((N * N * block, V), jnp.int16),
        jax.ShapeDtypeStruct((N,), jnp.int32),
        jax.ShapeDtypeStruct((N,), jnp.int32),
    )
    lowered = fn.lower(*args)
    assert "input_memory_space_colors" in lowered.as_text()
    staged = custom_call_operands_in_vmem(
        lowered.compile().as_text(), "small_all_to_all"
    )
    assert staged and not any(staged.values()), staged
