# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""T6 hbm-hop flags.

TPU_MOE_TP_DECODE_SHARED_META: every armed layer gets the same experts_start
tensor object (same value as before), so XLA can CSE the meta stack.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tpu_test


def test_shared_e0_buffer(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_tp_decode as m

    monkeypatch.setattr(envs, "TPU_MOE_TP_DECODE_SHARED_META", True)
    m._E0_BUFFERS.clear()
    dev = torch.device("cpu")
    a, b, c = m._e0_tensor(dev, 0), m._e0_tensor(dev, 0), m._e0_tensor(dev, 64)
    assert a is b and a is not c and int(c) == 64
    monkeypatch.setattr(envs, "TPU_MOE_TP_DECODE_SHARED_META", False)
    assert m._e0_tensor(dev, 0) is not m._e0_tensor(dev, 0)
