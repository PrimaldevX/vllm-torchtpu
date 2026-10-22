# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_INDEXER_K_WRITE_COMPACT: `coalesced_kv_append_dcp(compact=True)` writes
the same cache bytes as `compact=False` on every rank, and the static bound
`owned_rows_bound` is never exceeded.

  test_bound_never_exceeded   random AgentX-shaped batches (8 seq slots, any
                              offsets / chunk splits), every rank: owned rows
                              <= bound (pure address math)
  test_compact_bit_exact      whole cache, both layouts, DCP8 interleave 4,
                              all 8 ranks, T in {16, 1024, 2048, 4096}
  test_fault_control          a bound cut below the owned count, and an
                              off-by-one compaction index, are both caught
Run as a script for the chained microbenchmark (`bench`).
"""

import functools
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
    kv_cache_utils,
)

pytestmark = pytest.mark.tpu_test

PAGE, DCP, IL, SEQS, PPS = 256, 8, 4, 8, 12
_HAS, _SAL = "head_along_sublane", "seq_along_lane"


def _batch(rng, t, kind):
    """`(seq_lens, cu_q_lens, page_indices)` of an AgentX-like step: decode
    rows (q 1-4) plus one prefill chunk filling the bucket, random offsets."""
    q = np.zeros(SEQS, np.int32)
    if kind == "decode":
        n = rng.integers(1, SEQS + 1)
        q[:n] = rng.integers(1, 5, size=n)
        q[:n] = np.minimum(q[:n], t // SEQS)
    else:
        n_dec = rng.integers(0, SEQS)
        q[:n_dec] = rng.integers(1, 5, size=n_dec)
        q[n_dec] = t - q.sum() - rng.integers(0, 8)
    ctx = rng.integers(0, PAGE * DCP * (PPS - 2), size=SEQS).astype(np.int32)
    seq_lens = np.where(q > 0, ctx + q, 0).astype(np.int32)
    seq_lens = np.minimum(seq_lens, PAGE * DCP * PPS)
    cu = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    # Distinct local pages (virtual % num_pages), random virtual rank offset.
    local = rng.permutation(SEQS * PPS)
    pages = (local + SEQS * PPS * rng.integers(0, DCP, size=local.shape)).astype(
        np.int32
    )
    return jnp.asarray(seq_lens), jnp.asarray(cu), jnp.asarray(pages.reshape(SEQS, PPS))


def _dst(t, seq_lens, cu, pages, rank, num_pages):
    page, slot = kv_cache_utils.get_dcp_page_and_slot(
        t,
        seq_lens,
        pages,
        cu,
        rank,
        page_size=PAGE,
        num_pages=num_pages,
        dcp_size=DCP,
        interleave_size=IL,
    )
    return kv_cache_utils.get_dst_rows(page, slot, PAGE)


def test_bound_never_exceeded():
    rng = np.random.default_rng(1)
    worst = 0
    for trial in range(100):
        t = int(rng.choice([16, 64, 1024, 2048, 4096]))
        sl, cu, pg = _batch(rng, t, "decode" if t <= 64 else "mixed")
        cap = kv_cache_append.owned_rows_bound(
            t, num_seqs=SEQS, dcp_size=DCP, interleave_size=IL
        )
        for r in range(DCP):
            n = int((np.asarray(_dst(t, sl, cu, pg, r, SEQS * PPS)) >= 0).sum())
            worst = max(worst, n - min(cap, t) + 1_000_000)
            assert n <= min(cap, t), (t, r, n, cap)
    print(f"BOUND worst slack ok (max owned - cap = {worst - 1_000_000})")


@functools.partial(jax.jit, static_argnames=("layout", "compact"), donate_argnums=0)
def _append(cache, k, seq_lens, cu, pages, rank, *, layout, compact):
    return kv_cache_append.coalesced_kv_append_dcp(
        cache,
        k,
        seq_lens,
        pages,
        cu,
        rank,
        dcp_size=DCP,
        interleave_size=IL,
        kv_layout=layout,
        compact=compact,
    )


def _cache(rng, layout):
    c = jnp.asarray(
        rng.integers(0, 256, size=(SEQS * PPS, PAGE // 4, 4, 256), dtype=np.uint8)
    )
    if layout == _SAL:
        c = st_lib.convert_cache_to_seq_along_lane(c, actual_head_dim=128)
    return c


@pytest.mark.parametrize("layout", [_HAS, _SAL])
@pytest.mark.parametrize("t", [16, 1024, 2048, 4096])
def test_compact_bit_exact(layout, t):
    rng = np.random.default_rng(t + (layout == _SAL))
    for trial in range(3):
        sl, cu, pg = _batch(rng, t, "decode" if t <= 64 else "mixed")
        k = jnp.asarray(rng.integers(0, 256, size=(t, 129), dtype=np.uint8))
        cache = _cache(rng, layout)
        bad = 0
        for r in range(DCP):
            a = _append(
                jnp.copy(cache),
                k,
                sl,
                cu,
                pg,
                jnp.int32(r),
                layout=layout,
                compact=False,
            )
            b = _append(
                jnp.copy(cache),
                k,
                sl,
                cu,
                pg,
                jnp.int32(r),
                layout=layout,
                compact=True,
            )
            bad += int((np.asarray(a) != np.asarray(b)).sum())
        print(f"COMPACT layout={layout} T={t} trial={trial} mismatching_bytes={bad}")
        assert bad == 0


@pytest.mark.parametrize("fault", ["bound_cut", "index_shift"])
def test_fault_control(monkeypatch, fault):
    t, layout = 4096, _SAL
    rng = np.random.default_rng(99)
    sl, cu, pg = _batch(rng, t, "mixed")
    k = jnp.asarray(rng.integers(0, 256, size=(t, 129), dtype=np.uint8))
    cache = _cache(rng, layout)
    ref = [
        _append(
            jnp.copy(cache), k, sl, cu, pg, jnp.int32(r), layout=layout, compact=False
        )
        for r in range(DCP)
    ]
    if fault == "bound_cut":
        monkeypatch.setattr(
            kv_cache_append, "owned_rows_bound", lambda n, **kw: n // DCP - 64
        )
    else:
        orig_take = jnp.take
        monkeypatch.setattr(
            kv_cache_append.jnp,
            "take",
            lambda a, i, axis=None, **kw: orig_take(
                a, jnp.minimum(i + 1, a.shape[0] - 1), axis=axis, **kw
            ),
        )
    _append.clear_cache() if hasattr(_append, "clear_cache") else None
    jax.clear_caches()
    bad = 0
    for r in range(DCP):
        b = _append(
            jnp.copy(cache), k, sl, cu, pg, jnp.int32(r), layout=layout, compact=True
        )
        bad += int((np.asarray(ref[r]) != np.asarray(b)).sum())
    print(f"FAULT {fault} mismatching_bytes={bad} caught={bad > 0}")
    assert bad > 0


def bench():
    M_LIST, REPS = [2, 6, 10], 6
    for layout in (_SAL, _HAS):
        for t in (1024, 2048, 4096):
            rng = np.random.default_rng(t)
            sl, cu, pg = _batch(rng, t, "mixed")
            k = jnp.asarray(rng.integers(0, 256, size=(t, 129), dtype=np.uint8))
            cache = _cache(rng, layout)
            res = {}
            for compact in (False, True):

                def chained(
                    m,
                    compact=compact,
                    layout=layout,
                    cache=cache,
                    k=k,
                    sl=sl,
                    cu=cu,
                    pg=pg,
                ):
                    def f(c0, k, sl, cu, pg):
                        def body(i, acc):
                            return kv_cache_append.coalesced_kv_append_dcp(
                                acc,
                                k,
                                sl,
                                pg,
                                cu,
                                jnp.int32(i % DCP),
                                dcp_size=DCP,
                                interleave_size=IL,
                                kv_layout=layout,
                                compact=compact,
                            )

                        return jax.lax.fori_loop(0, m, body, c0)

                    return (
                        jax.jit(f, donate_argnums=0)
                        .lower(cache, k, sl, cu, pg)
                        .compile()
                    )

                fns = {m: chained(m) for m in M_LIST}
                times = {m: [] for m in M_LIST}
                for _ in range(REPS):
                    for m, fn in fns.items():
                        c = jnp.copy(cache)
                        jax.block_until_ready(c)
                        t0 = time.perf_counter()
                        jax.block_until_ready(fn(c, k, sl, cu, pg))
                        times[m].append(time.perf_counter() - t0)
                meds = [statistics.median(times[m][1:]) * 1e6 for m in M_LIST]
                mbar, tbar = statistics.mean(M_LIST), statistics.mean(meds)
                slope = sum(
                    (m - mbar) * (x - tbar) for m, x in zip(M_LIST, meds)
                ) / sum((m - mbar) ** 2 for m in M_LIST)
                res[compact] = slope
            print(
                f"KWBENCH layout={layout} T={t} full={res[False]:.1f}us "
                f"compact={res[True]:.1f}us "
                f"delta={res[True] - res[False]:+.1f}us",
                flush=True,
            )


if __name__ == "__main__":
    print("devices:", jax.devices()[:1])
    bench() if (len(sys.argv) < 2 or sys.argv[1] == "bench") else None
