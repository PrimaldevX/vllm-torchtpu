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
"""TPU_VOCAB_EMBED_I32_BOUNDS: int32 shard bounds give the same masked ids and
mask as int64 bounds (CPU)."""

import pytest
import torch


@pytest.mark.parametrize("shard", range(16))
def test_i32_bounds_match_i64(shard):
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    import vllm_torchtpu

    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()
    vpe_fn = vpe.get_masked_input_and_mask
    assert vpe_fn.__name__ == "patched_get_masked_input_and_mask"
    per = 9680
    start, end = shard * per, (shard + 1) * per
    pad = 0
    added_start, added_end = 154880 + shard * 4, 154880 + shard * 4 + 4
    g = torch.Generator().manual_seed(shard)
    ids = torch.cat(
        [
            torch.randint(0, 154880 + 64, (4096,), generator=g),
            torch.tensor(
                [
                    start,
                    end - 1,
                    end,
                    start - 1 if start else 0,
                    added_start,
                    added_end - 1,
                    added_end,
                ]
            ),
        ]
    ).to(torch.int32)
    out = {}
    for dt in (torch.int64, torch.int32):
        b = [
            torch.tensor(v, dtype=dt) for v in (start, end, pad, added_start, added_end)
        ]
        out[dt] = vpe_fn(ids, *b)
    (m64, k64), (m32, k32) = out[torch.int64], out[torch.int32]
    assert m32.dtype == torch.int32
    assert torch.equal(m32.long(), m64.long())
    assert torch.equal(k32, k64)
    # fault control: an off-by-one start must differ
    bad = vpe_fn(
        ids,
        *[
            torch.tensor(v, dtype=torch.int32)
            for v in (start + 1, end, pad, added_start, added_end)
        ],
    )
    assert not torch.equal(bad[0].long(), m64.long())
