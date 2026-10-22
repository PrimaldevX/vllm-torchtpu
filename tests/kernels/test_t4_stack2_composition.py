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
"""Trial-4 stack (t4-stack2), attention half: the trial-3 final composition
(test_t3_final_composition's FLAGS_ON, row widths included) plus

* TPU_DCP_MLA_SEND_LAYOUT on the decode steps (16, 32): the kernel writes the
  packed real-rows send buffer (the op variant the layer calls when
  `send_layout_applies`), and
* TPU_DCP_MLA_MERGE_PIECES=2 (MIN_TOKENS 2048, TPU_DCP_MERGE_CHUNKED=8) on the
  2048-token mixed step: per-merge-piece output buffers,

all on vs all off (DCP-8; mixed-256, decode-16, decode-32, mixed-2048).
Every step's caches must equal the all-off run bit for bit. Mixed steps: the
partial outputs and lse too, and the threaded a2a merge (pack + real rows) of
the on partials equals the two-exchange merge of the off partials. Decode
steps: every rank's sent rows equal the packed real rows of the off partials,
and the send-layout merge (threaded ragged exchange) equals the two-exchange
merge of the off partials, bit for bit. Wiring: row widths on every capped
on-call, send_world on exactly the two decode calls, out_pieces (= the 8 merge
chunks) on exactly the 2048 call.

`T4S2_DUMP=<prefix>` writes `<prefix>.mixed.npz` (the mixed steps' off / on
partials, test_t4_stack2_merge_multichip) and `<prefix>.send.npz` (decode
steps' off partials + send buffers, test_dcp_send_layout_merge_multichip's
T5SL_PARTIALS format).
"""

import os
import sys
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_dcp_mla_send_layout as sl  # noqa: E402
import test_mla_decode_prologue as prologue  # noqa: E402
import test_t3_final_composition as t3  # noqa: E402
import test_t4_combined_composition as t4  # noqa: E402

pytestmark = pytest.mark.tpu_test

# Trial-4 flags that are on by default now but not what this file composes:
# pinned off in both arms, as they were when it was written (the on arm's
# T4S2_FLAGS_ON still turns its own on).
_TRIAL4_DEFAULTS_OFF = {
    "TPU_DCP_MLA_SEND_LAYOUT": "0",
    "TPU_DCP_MLA_SEND_LAYOUT_UNPACK": "0",
    "TPU_DCP_MLA_MERGE_PIECES": "0",
    "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS": "0",
    "TPU_DCP_MLA_MIXED_KERNEL_V2": "0",
    "TPU_DCP_MERGE_CHUNKED_SMALL": "0",
    "TPU_DCP_MERGE_WEIGHTED_HM": "0",
    "TPU_INDEXER_QPREP_FUSED": "0",
    "TPU_INDEXER_K_WRITE_COMPACT": "0",
}


@pytest.fixture(autouse=True)
def _trial4_defaults_off(monkeypatch):
    for k, v in _TRIAL4_DEFAULTS_OFF.items():
        monkeypatch.setenv(k, v)


T4S2_FLAGS_ON = dict(
    t3.T3_FLAGS_ON,
    TPU_DCP_LSE_MERGE_A2A="1",
    TPU_DCP_MERGE_CHUNKED="8",
    TPU_DCP_MLA_MERGE_PIECES="2",
    TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS="2048",
)


def _on_run(dcp, sparse_kernel, seen):
    from vllm_torchtpu.layers.core import attention_interface

    real_fn = attention_interface.sparse_mla_attention_dcp
    real_k = sparse_kernel.sparse_ragged_paged_attention

    def wrapped(ql_nope, *args, **kwargs):
        # The layer's send op: decode buckets only (send_layout_applies).
        if ql_nope.shape[0] > sparse_kernel.FUSED_PREP_MAX_TOKENS:
            return real_fn(ql_nope, *args, **kwargs)
        nope, rope, send = real_fn(ql_nope, *args, send_layout=True, **kwargs)
        lse = jnp.zeros((dcp * ql_nope.shape[0], ql_nope.shape[1]), jnp.float32)
        return nope, rope, send, lse

    def spy(*args, **kwargs):
        seen.append(
            (
                tuple(kwargs.get("decode_row_widths", ())),
                int(kwargs.get("send_world", 0)),
                int(kwargs.get("out_pieces", 0)),
            )
        )
        return real_k(*args, **kwargs)

    with (
        mock.patch.object(t4, "FLAGS_ON", T4S2_FLAGS_ON),
        mock.patch.object(attention_interface, "sparse_mla_attention_dcp", wrapped),
        mock.patch.object(sparse_kernel, "sparse_ragged_paged_attention", spy),
    ):
        return t4._attention_run(dcp, on=True)


@pytest.mark.multichip
def test_t4_stack2_attention_all_flags_on_vs_off():
    prologue._require_tpu()
    dcp = t4._dcp_size()
    if dcp < 8:
        # The on arm forces the send layout, whose 16- and 32-token decode
        # buckets of 32 heads fit only from DCP8 (V/2 partial rows per rank).
        pytest.skip(f"the send-layout on arm needs DCP8 (8 devices), have {dcp}")
    import torch

    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel

    for k in T4S2_FLAGS_ON:
        os.environ.pop(k, None)
    os.environ.update(_TRIAL4_DEFAULTS_OFF)  # MERGE_PIECES is on by default
    ref, ref_w = t4._attention_run(dcp, on=False)
    seen = []
    got, got_w = _on_run(dcp, sparse_kernel, seen)
    print("wiring off", ref_w, "on", got_w)
    print("on kernel calls (row widths, send_world, out_pieces):", seen)
    assert got_w["tc"] == 0 and got_w["kvw"] == 2 and got_w["iota"] == 0, got_w
    assert got_w["prep"] > 0, got_w
    widths = tuple(int(w) for w in t3.ROW_WIDTHS.split(","))
    assert widths in [w for w, _, _ in seen], seen
    assert {w for w, _, _ in seen} <= {(), widths}, seen
    assert [s for _, s, _ in seen if s] == [dcp, dcp], seen
    # One piece per merge chunk (TPU_DCP_MERGE_CHUNKED=8), the 2048 call only.
    assert [p for _, _, p in seen if p] == [8], seen

    mixed, send = {}, {}
    nm = ns = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope"):
            n = int((np.asarray(a[k]) != np.asarray(b[k])).sum())
            print(f"step {i} {k}: {n} differing elements")
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        real = a["real"]
        tokens = a["out"].shape[0] // dcp
        outs, lses = prologue._rank_partials(a, dcp)
        want = t4._merge_every_rank(outs, lses, on=False)
        if tokens <= sparse_kernel.FUSED_PREP_MAX_TOKENS:
            sends = sl._rank_sends(b, dcp)
            for r in range(dcp):
                nw, no = sl._diffs(
                    sends[r].numpy(),
                    outs[r].view(torch.int16).numpy(),
                    lses[r].numpy(),
                    dcp,
                    real,
                )
                assert nw == 0 and no == 0, f"step {i} rank {r}: {nw} / {no}"
            one = sl._merge_send_every_rank(sends, real, tokens)
            kind = "send layout"
            send[f"s{ns}_on_out"] = a["out"]
            send[f"s{ns}_on_lse"] = a["lse"]
            send[f"s{ns}_send"] = b["out"]
            send[f"s{ns}_real"] = np.int32(real)
            ns += 1
        else:
            for k in ("out", "lse"):
                n = int((np.asarray(a[k]) != np.asarray(b[k])).sum())
                print(f"step {i} {k}: {n} differing elements")
                np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
            one = t4._merge_every_rank(
                *prologue._rank_partials(b, dcp), on=True, real=real
            )
            kind = "a2a pack + real rows"
            for tag, r in (("off", a), ("on", b)):
                mixed[f"s{nm}_{tag}_out"] = r["out"]
                mixed[f"s{nm}_{tag}_lse"] = r["lse"]
            mixed[f"s{nm}_real"] = np.int32(real)
            nm += 1
        diff = int(
            (one[:real].view(torch.int16) != want[:real].view(torch.int16)).sum()
        )
        print(
            f"step {i} ({tokens} tokens, {real} real) merged ({kind} vs off"
            f" two-exchange): {diff} differ; padding rows zero:"
            f" {bool((one[real:] == 0).all())}"
        )
        assert diff == 0, f"step {i}"
        assert (one[real:] == 0).all(), f"step {i}"
        assert torch.isfinite(one.float()).all(), f"step {i}"
    assert ns == 2 and nm == 2, (ns, nm)
    prefix = os.environ.get("T4S2_DUMP")
    if prefix:
        np.savez(f"{prefix}.mixed.npz", dcp=np.int32(dcp), steps=np.int32(nm), **mixed)
        np.savez(f"{prefix}.send.npz", dcp=np.int32(dcp), steps=np.int32(ns), **send)
        print("saved", prefix)


@pytest.mark.multichip
def test_t4_stack2_merge_pieces_swap_fault_is_caught():
    """Positive control: with the pieces exchanged (out_pieces_fault="swap")
    the 2048 step's partials must differ from the all-off run."""
    prologue._require_tpu()
    dcp = t4._dcp_size()
    if dcp < 8:
        # The on arm forces the send layout, whose 16- and 32-token decode
        # buckets of 32 heads fit only from DCP8 (V/2 partial rows per rank).
        pytest.skip(f"the send-layout on arm needs DCP8 (8 devices), have {dcp}")
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel

    for k in T4S2_FLAGS_ON:
        os.environ.pop(k, None)
    os.environ.update(_TRIAL4_DEFAULTS_OFF)  # MERGE_PIECES is on by default
    ref, _ = t4._attention_run(dcp, on=False)
    real_k = sparse_kernel.sparse_ragged_paged_attention

    def faulty(*args, **kwargs):
        if kwargs.get("out_pieces"):
            kwargs["out_pieces_fault"] = "swap"
        return real_k(*args, **kwargs)

    with mock.patch.object(sparse_kernel, "sparse_ragged_paged_attention", faulty):
        bad, _ = _on_run(dcp, sparse_kernel, [])
    a, b = ref[-1], bad[-1]
    assert not np.array_equal(a["out"], b["out"]), "swapped pieces not caught"
