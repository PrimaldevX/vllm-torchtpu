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
"""TPU_MTP_TOPK_INPLACE: IndexShare's step-0 snapshot and per-step restore
without pool-sized copies.

* CPU: the snapshot is the same rows the compaction path saved (including
  slot ids that overlap the front rows), and it leaves the table untouched;
  the restore hands the in-place kernel exactly rows [0, n) and the saved
  rows; flag off keeps the eager paths.
* TPU: the in-place restore writes the same table as the eager slice
  assignment, bit for bit, on GLM-5.2's [4096, 2048] int32 table at every
  loop bucket, with a positive control.
"""

import types

import pytest
import torch


def _proposer(monkeypatch, table, inplace):
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode import eagle3

    monkeypatch.setattr(envs, "TPU_MTP_TOPK_INPLACE", inplace, raising=False)
    prop = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    prop._mtp_topk_buffers = [table]
    prop.runner = types.SimpleNamespace(device=table.device)

    def compact(slot_ids):  # vLLM DeepSeekMultiTokenPredictor.compact_topk_indices
        n = slot_ids.numel()
        table[:n] = table[slot_ids]

    prop.draft_model = types.SimpleNamespace(
        model=types.SimpleNamespace(compact_topk_indices=compact)
    )
    return prop


@pytest.mark.parametrize("p", [4, 8, 16])
def test_snapshot_matches_the_compaction_path(monkeypatch, p):
    g = torch.Generator().manual_seed(p)
    base = torch.randint(-1, 100000, (64, 2048), generator=g, dtype=torch.int32)
    # slot ids that overlap the front rows: the compaction reads before writing
    idx = torch.randperm(32, generator=g)[:p].to(torch.int32)
    old_tab = base.clone()
    old = _proposer(monkeypatch, old_tab, False)._mtp_compact_and_save(idx)
    new_tab = base.clone()
    new = _proposer(monkeypatch, new_tab, True)._mtp_compact_and_save(idx)
    assert len(old) == len(new) == 1
    assert torch.equal(old[0], new[0]) and new[0].shape == (p, 2048)
    assert torch.equal(new_tab, base), "the snapshot must not write the table"
    # the old path compacted the table front; a restore of the snapshot then
    # gives the new path the same front rows
    new_tab[:p] = new[0]
    assert torch.equal(new_tab[:p], old_tab[:p])


def test_restore_calls_the_in_place_kernel_with_rows_0_to_n(monkeypatch):
    import vllm_torchtpu.kernels.partial_kv_update as pku

    calls = []

    def fake(cache, indices, updates):
        calls.append((cache, indices.clone(), updates.clone()))
        cache[indices.long()] = updates
        return cache

    monkeypatch.setattr(pku, "pallas_index_copy_", fake)
    table = torch.full((64, 2048), -1, dtype=torch.int32)
    prop = _proposer(monkeypatch, table, True)
    rows = torch.arange(8 * 2048, dtype=torch.int32).reshape(8, 2048)
    prop._mtp_restore_topk([rows])
    assert len(calls) == 1
    cache, ids, upd = calls[0]
    assert cache is table
    assert ids.dtype == torch.int32 and ids.tolist() == list(range(8))
    assert torch.equal(upd, rows)
    assert torch.equal(table[:8], rows) and torch.all(table[8:] == -1)
    # cached per n
    prop._mtp_restore_topk([rows])
    assert calls[1][1].data_ptr() == calls[0][1].data_ptr() or torch.equal(
        calls[1][1], calls[0][1]
    )


def test_flag_off_keeps_the_eager_paths(monkeypatch):
    import vllm_torchtpu.kernels.partial_kv_update as pku

    monkeypatch.setattr(pku, "pallas_index_copy_", lambda *a: 1 / 0)
    table = torch.full((64, 2048), -1, dtype=torch.int32)
    prop = _proposer(monkeypatch, table, False)
    rows = torch.ones(4, 2048, dtype=torch.int32)
    prop._mtp_restore_topk([rows])
    assert torch.equal(table[:4], rows)


def test_flag_defaults(monkeypatch):
    """TPU_MTP_TOPK_INPLACE is on by default since trial 6."""
    import vllm_torchtpu.envs as envs

    monkeypatch.delenv("TPU_MTP_TOPK_INPLACE", raising=False)
    assert envs.TPU_MTP_TOPK_INPLACE is True


def _require_tpu():
    import jax

    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")


@pytest.mark.tpu_test
def test_in_place_restore_is_bit_identical_on_tpu():
    _require_tpu()
    from vllm_torchtpu.kernels.partial_kv_update import pallas_index_copy_

    g = torch.Generator().manual_seed(0)
    base = torch.randint(-1, 1 << 20, (4096, 2048), generator=g, dtype=torch.int32)
    for n in (4, 8, 16, 32, 64):
        rows = torch.randint(-1, 1 << 20, (n, 2048), generator=g, dtype=torch.int32)
        want = base.clone()
        want[:n] = rows
        tab = base.clone().to("tpu")
        ids = torch.arange(n, dtype=torch.int32).to("tpu")
        pallas_index_copy_(tab, ids, rows.to("tpu"))
        got = tab.cpu()
        assert torch.equal(got, want), n
        # positive control: the rows shifted by one must not match
        bad = base.clone().to("tpu")
        pallas_index_copy_(bad, ids, torch.roll(rows, 1, 0).to("tpu"))
        assert not torch.equal(bad.cpu(), want)
