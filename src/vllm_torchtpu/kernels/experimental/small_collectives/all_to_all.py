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
"""All-to-all of small row blocks as one Pallas kernel (one round of remote DMAs).

``x`` is ``[n * block, V]``: block ``d`` is this core's message for core ``d``.
Every core pushes rows ``[row0, block)`` of block ``d`` straight into rows
``[me * block + row0, (me + 1) * block)`` of core ``d``'s output in HBM, copies
its own block into its own slot, and waits for the ``n - 1`` arrivals. Nothing
is staged in VMEM and no vector unit touches the words, so every row at or
past ``row0`` of every output block is bitwise ``lax.all_to_all(x, axis, 0, 0,
tiled=True)``'s; rows before ``row0`` are left unwritten (the caller never
reads them).

It is the DCP LSE merge's exchange at decode (TPU_DCP_MLA_SEND_SMALL_A2A: 8
ranks x a few-KB int16 send block per destination), which XLA runs as a
ragged all-to-all of 9-11 us on the served mirror; the same bytes in one
round of remote DMAs need about one barrier plus one 7-peer round.

As in ``all_gather``, one barrier per call is enough: a core leaves call k
only after every peer's block of call k has landed and its own sends are
done, and a peer sends call k + 1 only after the barrier of call k + 1, which
this core enters only after it left call k. The barrier is also what makes a
peer's output buffer safe to write: the peer has entered the kernel that owns
it.

Peer addresses: inside a ``(lane, axis)`` mesh a peer is ``(my lane, j)``;
under the torch bridge both indices arrive as data.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    MESH,
    wait_bytes_of,
)

# Barrier semaphore group, distinct from the fused MoE kernels (0), the
# all-reduces (5), the reduce-scatter (7) and the all-gather (9).
COLLECTIVE_ID = 11


def reference(blocks, row0=0):
    """[cores, cores, block, V] (sender, destination, ...) -> what each core
    gets, [cores, cores * block, V], with rows before ``row0`` zero."""
    out = blocks.swapaxes(0, 1).copy()
    out[:, :, :row0] = 0
    n = out.shape[0]
    return out.reshape(n, -1, out.shape[-1])


def small_all_to_all(
    x,
    *,
    num_cores,
    axis,
    row0=0,
    rank=None,
    lane_axis=None,
    lane=None,
    collective_id=COLLECTIVE_ID,
    hbm_operand=False,
):
    """``lax.all_to_all(x, axis, 0, 0, tiled=True)`` of ``[n * block, V]``
    for rows ``[row0, block)`` of every block, inside a shard_map over a mesh
    whose ``axis`` has ``num_cores`` cores.

    Args:
      x: [n * block, V], this core's blocks, destination-major (any dtype; the
        bytes are copied). ``block`` and ``row0`` multiples of 8.
      row0: the first row of each block to move (static).
      rank: int32 [1], this core's index on ``axis`` as data, or None to read
        it from the axis.
      lane_axis: the mesh's other axis, if it has one (at most one).
      lane: int32 [1], this core's index on ``lane_axis`` as data, or None to
        read it from the axis. Ignored without ``lane_axis``.
      hbm_operand: constrain ``x`` to HBM (``pltpu.with_memory_space_constraint``).
        With the operand in ``pl.ANY``, XLA's memory-space assignment stages the
        send blocks into VMEM with a copy-start/-done right before the call,
        exposed because the producer runs just before; the kernel only DMAs
        them, so HBM costs nothing. The in_spec's memory space alone does not
        stop the staging (it only types the kernel ref).
    """
    if x.ndim != 2:
        raise ValueError(f"small_all_to_all takes [n * block, V], got {x.shape}")
    n = num_cores
    total, _ = x.shape
    if total % n:
        raise ValueError(f"{total} rows do not split into {n} blocks")
    block = total // n
    if block % 8 or row0 % 8 or not 0 <= row0 < block:
        raise ValueError(
            f"block {block} / row0 {row0}: need multiples of 8, row0 < block"
        )
    rows = block - row0
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

        mine = o_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), rows)]
        for j in range(n):  # block j into slot `me` of core j

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=x_ref.at[pl.ds(j * block + row0, rows)],
                    dst_ref=mine,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

        own = x_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), rows)]
        local = pltpu.make_async_copy(own, mine, local_sem)
        local.start()
        others = o_ref.at[pl.ds(0, (n - 1) * rows)]
        wait_bytes_of(others, recv_sem)
        local.wait()
        wait_bytes_of(others, send_sem)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_ = pl.BlockSpec(memory_space=pl.ANY)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))
    if hbm_operand:
        x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[smem] * len(scalars) + [hbm if hbm_operand else any_],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_to_all_n{n}_b{block}_r{row0}",
    )(*scalars, x)
