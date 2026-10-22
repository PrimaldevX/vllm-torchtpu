"""TPU_H2D_REUSE_STEADY: the padded sampling params reuse the previous device
copy only while their values repeat; values always equal a fresh upload."""

from types import SimpleNamespace

import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(temps, top_k, top_p):
    r = SimpleNamespace(
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.tensor(temps, dtype=torch.float32),
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
        ),
        vocab_size=1000,
        _sampler_light=False,
        _h2d_reuse={},
        device=torch.device("cpu"),
    )
    r._upload_unless_unchanged = (
        lambda slot, host, device=None: TPUModelRunner._upload_unless_unchanged(
            r, slot, host, device
        )
    )
    return r


def _call(r, n=3, padded=8, dtype=torch.float32):
    logits = torch.zeros(padded, 4, dtype=dtype)
    return TPUModelRunner._build_padded_sampling_params(
        r, 0, n, logits, padded_num_reqs=padded
    )


def test_reuse_only_while_values_repeat(monkeypatch):
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "1")
    r = _runner([0.7, 1.0, 0.5], [20, 0, 5000], [0.9, 1.0, 0.8])
    a = _call(r)
    b = _call(r)
    assert all(x is y for x, y in zip(a, b))  # steady: no new upload
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "0")  # on by default
    fresh = _call(r)
    for x, y in zip(a, fresh):
        assert x is not y and x.dtype == y.dtype and torch.equal(x, y)
    # top_k >= vocab is neutralised to 0 on both paths
    assert fresh[1][2, 0] == 0 and fresh[1][3, 0] == 0
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "1")
    r.input_batch.temperature_cpu_tensor[1] = 0.25  # a request changes
    c = _call(r)
    assert c[0] is not a[0] and float(c[0][1, 0]) == 0.25
    assert c[1] is a[1] and c[2] is a[2]  # unchanged tensors still reused
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "0")
    for x, y in zip(c, _call(r)):
        assert torch.equal(x, y)


def test_shape_change_uploads(monkeypatch):
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "1")
    r = _runner([0.7, 1.0, 0.5], [20, 0, 5], [0.9, 1.0, 0.8])
    a = _call(r, padded=8)
    b = _call(r, padded=16)
    assert a[0].shape[0] == 8 and b[0].shape[0] == 16 and a[0] is not b[0]


def test_bfloat16_logits(monkeypatch):
    """The temperatures follow the logits dtype; bf16 has no numpy dtype."""
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "1")
    r = _runner([0.7, 1.0, 0.5], [20, 0, 5], [0.9, 1.0, 0.8])
    a = _call(r, dtype=torch.bfloat16)
    b = _call(r, dtype=torch.bfloat16)
    assert a[0].dtype == torch.bfloat16 and all(x is y for x, y in zip(a, b))
    r.input_batch.temperature_cpu_tensor[0] = 0.71  # differs in bf16 too
    c = _call(r, dtype=torch.bfloat16)
    assert c[0] is not a[0]
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "0")
    for x, y in zip(c, _call(r, dtype=torch.bfloat16)):
        assert x.dtype == y.dtype and torch.equal(x, y)


def test_key_bytes_match_numpy_for_numpy_dtypes():
    for t in (
        torch.arange(9, dtype=torch.int32),
        torch.arange(6, dtype=torch.int64)[::2],
        torch.tensor([True, False]),
        torch.rand(3, 2).t(),
    ):
        assert t.contiguous().view(torch.uint8).numpy().tobytes() == t.numpy().tobytes()


def test_zero_dim_host_key():
    r = _runner([0.7], [1], [0.9])
    a = TPUModelRunner._upload_unless_unchanged(
        r, "s", torch.tensor(5, dtype=torch.int32)
    )
    b = TPUModelRunner._upload_unless_unchanged(
        r, "s", torch.tensor(5, dtype=torch.int32)
    )
    assert a is b and int(a) == 5


def test_target_device_is_part_of_the_key(monkeypatch):
    """Same values asked for on another device: a fresh copy on that device,
    never the previous upload (a CPU probe followed by device logits)."""
    monkeypatch.setenv("TPU_H2D_REUSE_STEADY", "1")
    r = _runner([0.7, 1.0, 0.5], [20, 0, 5], [0.9, 1.0, 0.8])
    host = torch.tensor([1, 2, 3], dtype=torch.int32)
    a = TPUModelRunner._upload_unless_unchanged(r, "s", host, torch.device("cpu"))
    m = TPUModelRunner._upload_unless_unchanged(
        r, "s", host.clone(), torch.device("meta")
    )
    assert a.device.type == "cpu" and m.device.type == "meta"
    m2 = TPUModelRunner._upload_unless_unchanged(
        r, "s", host.clone(), torch.device("meta")
    )
    assert m2 is m
    probe = _call(r)  # logits on cpu
    meta_logits = torch.zeros(8, 4, dtype=torch.float32, device="meta")
    p = TPUModelRunner._build_padded_sampling_params(
        r, 0, 3, meta_logits, padded_num_reqs=8
    )
    assert all(x.device.type == "meta" for x in p[:3]) and all(
        x.device.type == "cpu" for x in probe[:3]
    )
