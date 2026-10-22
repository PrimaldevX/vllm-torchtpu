"""RunnerStageTiming enabled/disabled and the runner's consumer gate."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.runner import stage_timing as st_mod


class _FakeEvent:
    records = 0
    syncs = 0

    def record(self):
        type(self).records += 1

    def synchronize(self):
        type(self).syncs += 1


@pytest.fixture
def fake_events(monkeypatch):
    _FakeEvent.records = 0
    _FakeEvent.syncs = 0
    monkeypatch.setattr(st_mod, "TpuEvent", _FakeEvent)
    return _FakeEvent


def test_disabled_records_nothing_and_starts_no_thread(monkeypatch, fake_events):
    def no_thread(*args, **kwargs):
        raise AssertionError("a disabled timer must not start its collector")

    monkeypatch.setattr(st_mod.threading, "Thread", no_thread)
    timing = st_mod.RunnerStageTiming(0, enabled=False)
    assert not timing.enabled
    ran = []
    with timing.stage("main_forward"):
        ran.append("stage")
    with timing.host_stage("output_assembly"):
        ran.append("host")
    assert ran == ["stage", "host"]
    assert fake_events.records == 0 and fake_events.syncs == 0
    assert timing.take_samples() == ([], [])


def test_disabled_stage_propagates_exceptions(fake_events):
    timing = st_mod.RunnerStageTiming(0, enabled=False)
    with pytest.raises(ValueError), timing.stage("sample"):
        raise ValueError("boom")


def test_enabled_still_records(fake_events):
    """Positive control: the default path keeps timing (2 records per device
    stage after the startup probe)."""
    timing = st_mod.RunnerStageTiming(3)
    assert timing.enabled
    probe = fake_events.records
    with timing.stage("main_forward"):
        pass
    with timing.host_stage("output_assembly"):
        pass
    assert fake_events.records == probe + 2
    cpu, _ = timing.take_samples()
    assert [s[1] for s in cpu] == ["main_forward", "output_assembly"]
    assert all(s[0] == "3" for s in cpu)


@pytest.mark.parametrize(
    "flag, kv_transfer_config, expected",
    [
        (None, None, False),  # default on (trial 6): nothing consumes them
        ("0", None, True),
        ("1", None, False),  # nothing consumes the samples
        ("1", object(), True),  # a KV connector may consume them
    ],
)
def test_runner_consumer_gate(monkeypatch, flag, kv_transfer_config, expected):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if flag is None:
        monkeypatch.delenv("TPU_STAGE_TIMING_ONLY_WITH_CONSUMER", raising=False)
    else:
        monkeypatch.setenv("TPU_STAGE_TIMING_ONLY_WITH_CONSUMER", flag)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_transfer_config)
    )
    assert TPUModelRunner._stage_timing_has_consumer(runner) is expected
