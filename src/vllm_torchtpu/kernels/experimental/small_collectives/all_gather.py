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
"""All-gather of a small flat block as one Pallas kernel (one-shot).

Every core pushes its whole ``[L]`` block straight into slot ``me`` of every
other core's ``[n * L]`` output in HBM, copies it into its own slot, and waits
for the ``n - 1`` arrivals. Nothing is staged in VMEM and no vector unit
touches the words, so the output is bitwise ``lax.all_gather(x, axis,
tiled=True)``: block r at ``[r * L, (r + 1) * L)``.

It is the DCP indexer's candidate exchange at decode (8 ranks x T * 2048
int32 score words, T <= 32), which XLA runs as a serial TensorCore
all-gather of 13-17 us on the served mirror; one round of remote DMAs moves
the same bytes without XLA's collective schedule.

As in ``all_reduce``, one barrier per call is enough: a core leaves call k
only after every peer's block of call k has landed and its own sends are
done, and a peer sends call k + 1 only after the barrier of call k + 1, which
this core enters only after it left call k. The barrier is also what makes a
peer's output buffer safe to write: the peer has entered the kernel that owns
it.

Peer addresses: inside a ``(lane, axis)`` mesh (the DCP mesh carries a lane
axis for the TP lanes DCP does not shard over) a peer is ``(my lane, j)``.
Under the torch bridge both indices arrive as data (``lax.axis_index`` lowers
to a partition-id the export path refuses).
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    MESH,
    wait_bytes_of,
)

# Barrier semaphore group, distinct from the all-reduce (5), the
# reduce-scatter (7) and the fused MoE kernels (0): this one barriers only
# the cores of one DCP group.
COLLECTIVE_ID = 9


def reference(blocks):
    """[cores, L] -> what every core gets: the blocks back to back."""
    return blocks.reshape(-1)


def small_all_gather(
    x,
    *,
    num_cores,
    axis,
    rank=None,
    lane_axis=None,
    lane=None,
    collective_id=COLLECTIVE_ID,
    hbm_operand=False,
):
    """``lax.all_gather(x, axis, tiled=True)`` of a 1-D block, inside a
    shard_map over a mesh whose ``axis`` has ``num_cores`` cores.

    Args:
      x: [L], this core's block (any 32-bit dtype; the bytes are copied).
      rank: int32 [1], this core's index on ``axis`` as data, or None to read
        it from the axis.
      lane_axis: the mesh's other axis, if it has one (at most one); a peer is
        this core's position on it and ``j`` on ``axis``.
      lane: int32 [1], this core's index on ``lane_axis`` as data, or None to
        read it from the axis. Ignored without ``lane_axis``.
      hbm_operand: constrain the block to HBM (``pltpu.with_memory_space_constraint``)
        so XLA does not stage it into VMEM right before the call.
    """
    if x.ndim != 1:
        raise ValueError(f"small_all_gather takes a flat block, got {x.shape}")
    n = num_cores
    size = x.shape[0]
    if size % (8 * LANES):
        raise ValueError(f"small_all_gather needs a multiple of {8 * LANES} words")
    # The kernel moves [rows, 128] tiles: a flat 32-bit array's (1024) tiling
    # is the same bytes as [rows, 128]'s (8, 128), so both reshapes are
    # bitcasts. (XLA still copies the SparseCore top-k's output before the
    # call, 0.8-0.9 us on the twin, with either operand shape or in_spec memory
    # space: that copy is XLA's memory-space assignment staging the pl.ANY
    # operand into VMEM; only an HBM constraint on the operand aval stops it,
    # see `hbm_operand`.)
    rows = size // LANES
    data_rank = rank is not None
    data_lane = lane_axis is not None and lane is not None

    def kernel(*refs):
        refs = list(refs)
        me = refs.pop(0)[0] if data_rank else lax.axis_index(axis)
        if lane_axis is None:
            my_lane = None
        elif data_lane:
            my_lane = refs.pop(0)[0]
        else:
            my_lane = lax.axis_index(lane_axis)
        x_ref, o_ref, send_sem, recv_sem, local_sem = refs

        def peer(j):
            ids = {axis: jnp.int32(j)}
            if lane_axis is not None:
                ids[lane_axis] = my_lane
            return ids

        sem = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(sem, 1, device_id=peer(j), device_id_type=MESH)
        pl.semaphore_wait(sem, n)

        mine = o_ref.at[pl.ds(pl.multiple_of(me * rows, rows), rows)]
        for j in range(n):  # my block into slot `me` of every other core

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=x_ref,
                    dst_ref=mine,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

        local = pltpu.make_async_copy(x_ref, mine, local_sem)
        local.start()
        others = o_ref.at[pl.ds(0, (n - 1) * rows)]
        wait_bytes_of(others, recv_sem)
        local.wait()
        wait_bytes_of(others, send_sem)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    # The block wherever XLA left it (a DMA reads VMEM as well as HBM); the
    # output in HBM, where the SparseCore merge reads it and the peers write it.
    any_ = pl.BlockSpec(memory_space=pl.ANY)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))
    block = x.reshape(rows, LANES)
    if hbm_operand:
        block = pltpu.with_memory_space_constraint(block, pltpu.HBM)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n * rows, LANES), x.dtype),
        in_specs=[smem] * len(scalars) + [hbm if hbm_operand else any_],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_gather_n{n}_l{size}",
    )(*scalars, block)
    return out.reshape(n * size)
