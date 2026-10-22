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
"""The cap rule for token-count gates (`envs.bucket_gate_edge`): every lower
gate that was `T >= min_tokens` is now `T > edge`, and on the power-of-2
bucket ladder it picks exactly the buckets it did."""

import pytest

from vllm_torchtpu import envs

# GLM-5.2 AgentX serving paddings (4..4096) plus the wider power-of-2 ladder.
LADDER = [1 << k for k in range(0, 15)]

# Served values of the converted lower gates (trial-5 stack s4) and defaults:
# merge pieces / weighted HM 2048, chunked / V-space 1024, RMSNorm 512,
# absorb / indexer a2a 64, send layout 8 (default) and 0 (s4).
SERVED_MINS = [0, 1, 8, 64, 512, 1024, 2048]


@pytest.mark.parametrize("min_tokens", SERVED_MINS + [1 << k for k in range(0, 15)])
def test_lower_gate_picks_the_same_buckets(min_tokens):
    edge = envs.bucket_gate_edge(min_tokens)
    for t in LADDER:
        assert (t > edge) == (t >= min_tokens), (min_tokens, t, edge)


@pytest.mark.parametrize("min_tokens", [1 << k for k in range(1, 15)])
def test_edge_is_a_power_of_two_bucket_edge(min_tokens):
    edge = envs.bucket_gate_edge(min_tokens)
    assert edge & (edge - 1) == 0 and edge == min_tokens // 2


@pytest.mark.parametrize("min_tokens", [3, 48, 100, 1000])
def test_non_power_of_two_keeps_its_exact_meaning(min_tokens):
    edge = envs.bucket_gate_edge(min_tokens)
    for t in range(0, 4 * min_tokens):
        assert (t > edge) == (t >= min_tokens)


@pytest.mark.parametrize("cap", [16, 32, 64])
def test_merge_chunk_gate_unchanged_on_the_ladder(cap, monkeypatch):
    """`_merge_chunks` (TPU_DCP_MERGE_CHUNKED_MIN_TOKENS=1024, chunks 8,
    SMALL=2 under the LSE_AG cap 2048): unchunked below 1024, as before."""
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    for name, value in dict(
        TPU_DCP_MERGE_CHUNKED=8,
        TPU_DCP_MERGE_CHUNKED_MIN_TOKENS=1024,
        TPU_DCP_MERGE_CHUNKED_SMALL=2,
        TPU_DCP_MERGE_CHUNKED_LSE_AG=True,
        TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS=2048,
    ).items():
        monkeypatch.setattr(cp_mla.envs, name, value, raising=False)
    want = {
        4: 1,
        8: 1,
        16: 1,
        32: 1,
        64: 1,
        128: 1,
        256: 1,
        512: 1,
        1024: 2,
        2048: 2,
        4096: 8,
    }
    assert {t: cp_mla._merge_chunks(t) for t in want} == want


def test_small_all_reduce_fold_gate_picks_the_same_buckets():
    """fold_geometry's rows test is `rows > tile_rows // 2` (cap rule): on the
    bucket ladder it is the old `rows >= tile_rows` (tile 16, bf16 on v7)."""
    from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

    hidden, cores, tile = 6144, 8, 16
    for rows in (4, 8, 16, 32, 64):
        got = ar.fold_geometry(rows, hidden, cores, tile) is not None
        own = hidden // cores
        old = not (own % ar.LANES or rows < tile) and (own // ar.LANES) >= 2
        assert got == old, rows
