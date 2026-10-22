# Copyright 2026 The vLLM TorchTPU Authors.
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
"""CPU checks of the IndexShare variant of the fused MTP draft loop (#1517):
the program installs the chunk's saved step-0 top-k rows into the draft's
table before every unrolled step, so each step reads exactly the rows the
per-step loop would have restored on the host -- even when the draft step
itself scribbles over the table (as the next chunk's step 0 would)."""

import pytest
import torch
from torch import nn

from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop


class _FakeMtp(nn.Module):
    """Reads the top-k table every step (like the MLA layer under skip_topk),
    records what it saw, then clobbers the table so a missing install shows."""

    def __init__(self, table: torch.Tensor, vocab: int = 7):
        super().__init__()
        self.table = table
        self.vocab = vocab
        self.seen: list[torch.Tensor] = []

    def forward(
        self, input_ids, positions, hidden_states, it=None, emb=None, *, spec_step_idx=0
    ):
        p = input_ids.shape[0]
        self.seen.append(self.table[:p].clone())
        self.table.fill_(-7)
        return hidden_states + self.table[:p, :1].to(hidden_states.dtype) * 0 + 1

    def compute_logits(self, h):
        return h[:, : self.vocab]


def _loop(table, num_steps=2):
    mtp = _FakeMtp(table)
    loop = _MtpFusedLoop.__new__(_MtpFusedLoop)
    nn.Module.__init__(loop)
    loop.prefix = "t"
    loop.mtp = mtp
    loop.num_steps = num_steps
    loop.topk_indices_buffer = table
    loop.vocab_parallel_argmax = None
    return loop, mtp


@pytest.mark.parametrize("p", [1, 4, 8])
def test_installs_saved_rows_before_every_step(p):
    rows_total, k = 16, 32
    table = torch.full((rows_total, k), -1, dtype=torch.int32)
    loop, mtp = _loop(table, num_steps=2)
    saved = torch.arange(p * k, dtype=torch.int32).reshape(p, k)
    seq_lens = torch.ones(p, dtype=torch.int32)
    toks = loop(
        torch.zeros(p, dtype=torch.int32),
        torch.zeros(p, dtype=torch.int32),
        torch.zeros(p, 16),
        seq_lens,
        saved,
    )
    assert len(toks) == 2 and all(t.dtype == torch.int32 for t in toks)
    assert len(mtp.seen) == 2
    for seen in mtp.seen:
        torch.testing.assert_close(seen, saved)
    # seq_lens advanced once between the two steps, positions out of place.
    assert seq_lens.tolist() == [2] * p


def test_rows_required_exactly_when_built_with_a_table():
    table = torch.zeros(4, 8, dtype=torch.int32)
    loop, _ = _loop(table)
    with pytest.raises(ValueError):
        loop(
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, 16),
            torch.ones(2, dtype=torch.int32),
            None,
        )
    loop.topk_indices_buffer = None
    with pytest.raises(ValueError):
        loop(
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, 16),
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, 8, dtype=torch.int32),
        )


def test_negative_control_without_install_reads_clobbered_rows():
    """Fault injection: skip the install and the second step reads the rows
    the first step clobbered -- the check above would catch a broken port."""
    table = torch.full((8, 8), -1, dtype=torch.int32)
    loop, mtp = _loop(table)
    loop._install_shared_topk = lambda step, rows: None
    saved = torch.ones(2, 8, dtype=torch.int32)
    table[:2] = saved
    loop(
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 16),
        torch.ones(2, dtype=torch.int32),
        saved,
    )
    assert torch.equal(mtp.seen[0], saved)
    assert not torch.equal(mtp.seen[1], saved)


def test_flag_defaults_on(monkeypatch):
    """On by default since trial 6 (acts only with VLLM_TPU_FUSED_DRAFT_LOOP)."""
    monkeypatch.delenv("VLLM_TPU_MTP_FUSED_INDEXSHARE", raising=False)
    import vllm_torchtpu.envs as envs

    assert envs.VLLM_TPU_MTP_FUSED_INDEXSHARE is True
    monkeypatch.setenv("VLLM_TPU_MTP_FUSED_INDEXSHARE", "0")
    assert envs.VLLM_TPU_MTP_FUSED_INDEXSHARE is False


# --- Eagle3Proposer side: rows restored eagerly before a read-only program ---


class _RecordingProgram:
    """Stands in for the compiled fused loop: records the table's front rows at
    each call (what every unrolled step reads, since no step writes it) and the
    call's arguments; returns K-1 tokens."""

    def __init__(self, table, num_steps=2, clobber=False):
        self.table = table
        self.num_steps = num_steps
        self.clobber = clobber
        self.calls: list[tuple] = []

    def __call__(self, input_ids, positions, hidden_states, seq_lens, *rest):
        p = input_ids.shape[0]
        self.calls.append((self.table[:p].clone(), len(rest), seq_lens.clone()))
        if self.clobber:
            self.table.fill_(-9)
        seq_lens.add_(self.num_steps - 1)
        return tuple(
            torch.full((p,), i, dtype=torch.int32) for i in range(self.num_steps)
        )


def _proposer(monkeypatch, table, program, warm=True):
    import contextlib
    import types

    from vllm_torchtpu.spec_decode import eagle3

    # The eager slice-assignment restore (TPU_MTP_TOPK_INPLACE, on by default,
    # restores in place with a Pallas index copy on the runner's device).
    monkeypatch.setenv("TPU_MTP_TOPK_INPLACE", "0")
    for name in (
        "set_forward_context",
        "set_vllm_model_wrapper_context",
        "set_model_tag",
    ):
        monkeypatch.setattr(eagle3, name, lambda *a, **k: contextlib.nullcontext())
    prop = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    skips = []
    inner = types.SimpleNamespace(set_skip_topk=skips.append)
    prop.draft_model = types.SimpleNamespace(model=inner)
    prop._share_mtp_indices = True
    prop._mtp_fused_indexshare = True
    prop._mtp_topk_buffers = [table]
    prop._mtp_fused_loop = program
    prop._mtp_fused_loop_warm = warm
    prop._mtp_fused_loop_sizes = None
    prop._draft_md_cache = {}
    prop.draft_vllm_config = None
    prop.runner = types.SimpleNamespace(
        mesh=None, _dp_num_tokens_across_dp=lambda n: None
    )

    def md(**kw):
        seq = torch.full((8,), 5, dtype=torch.int32)
        return {"l0": types.SimpleNamespace(seq_lens=seq)}

    prop._build_draft_attn_metadata = md
    return prop, skips


def _run(prop, p, rows):
    return prop._run_mtp_fused_loop(
        chunk=object(),
        input_ids=torch.zeros(p, dtype=torch.int32),
        positions=torch.zeros(p, dtype=torch.int32),
        hidden_states=torch.zeros(p, 16),
        num_rejected_np=None,
        num_rejected_dev=None,
        loop_query_start_loc=None,
        loop_request_distribution=None,
        shared_topk_indices=rows,
    )


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "first-call"])
def test_proposer_restores_each_chunks_rows_before_its_program(monkeypatch, warm):
    p, k = 4, 32
    table = torch.full((16, k), -1, dtype=torch.int32)
    prog = _RecordingProgram(table, clobber=True)
    prop, skips = _proposer(monkeypatch, table, prog, warm=warm)
    rows_a = torch.arange(p * k, dtype=torch.int32).reshape(p, k)
    rows_b = rows_a + 1000
    out_a = _run(prop, p, rows_a)
    table.fill_(-3)  # the next chunk's step 0 overwrites the table
    out_b = _run(prop, p, rows_b)
    assert out_a is not None and out_b is not None and len(out_a) == 2
    # A first call runs the program twice (trace, then compiled); chunk B then
    # runs warm. The program clobbers the table after every call, so each call
    # seeing its chunk's rows proves a restore right before it.
    n_a = 1 if warm else 2
    assert len(prog.calls) == n_a + 1
    for seen, _, _ in prog.calls[:n_a]:
        assert torch.equal(seen, rows_a)
    assert torch.equal(prog.calls[-1][0], rows_b)
    assert all(n_rest == 0 for _, n_rest, _ in prog.calls), "no rows argument"
    assert skips[0] is True and skips[-1] is False


def test_proposer_negative_control_without_restore(monkeypatch):
    """Fault injection: drop the restore and the program reads another chunk's
    rows -- the check above would catch a missing or misplaced restore."""
    p, k = 4, 32
    table = torch.full((16, k), -1, dtype=torch.int32)
    prog = _RecordingProgram(table)
    prop, _ = _proposer(monkeypatch, table, prog)
    prop._mtp_restore_topk = lambda saved: None
    rows = torch.ones(p, k, dtype=torch.int32)
    _run(prop, p, rows)
    assert not torch.equal(prog.calls[0][0], rows)


def test_proposer_rows_that_do_not_fit_leave_the_table_alone(monkeypatch):
    p, k = 4, 32
    table = torch.full((16, k), -1, dtype=torch.int32)
    prog = _RecordingProgram(table)
    prop, _ = _proposer(monkeypatch, table, prog)
    before = table.clone()
    assert _run(prop, p, torch.ones(p + 1, k, dtype=torch.int32)) is None
    assert _run(prop, p, torch.ones(p, k, dtype=torch.int64)) is None
    assert _run(prop, p, None) is None
    assert torch.equal(table, before) and not prog.calls


def test_fused_loop_takes_the_vocab_parallel_argmax(monkeypatch):
    """With vocab_parallel_argmax set, each step's token is that reduction of
    the draft's (shard) logits, not their plain argmax."""
    import vllm_torchtpu.spec_decode.utils as su

    seen = []

    def fake_vpa(logits, info):
        seen.append((logits.clone(), info))
        return torch.full((logits.shape[0],), 42, dtype=torch.int32)

    monkeypatch.setattr(su, "vocab_parallel_argmax", fake_vpa)
    table = torch.zeros(8, 8, dtype=torch.int32)
    loop, _ = _loop(table)
    loop.topk_indices_buffer = None
    loop.vocab_parallel_argmax = "INFO"
    toks = loop(
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 16),
        torch.ones(2, dtype=torch.int32),
        None,
    )
    assert [t.tolist() for t in toks] == [[42, 42], [42, 42]]
    assert len(seen) == 2 and all(info == "INFO" for _, info in seen)
    loop.vocab_parallel_argmax = None
    toks = loop(
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 16),
        torch.ones(2, dtype=torch.int32),
        None,
    )
    assert all(t.dtype == torch.int32 for t in toks) and len(seen) == 2


# --- compile time: one program for every rank, at the loop buckets only ----


def _traced_argmax_graph(info, onehot):
    """The FX graph Dynamo traces for the fused loop's draft argmax on one
    rank (the collective is stubbed; only the traced code matters)."""
    import vllm.distributed as vd

    import vllm_torchtpu.spec_decode.utils as su

    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            vd, "tensor_model_parallel_all_gather", lambda x, dim=0: x.repeat(2, 1)
        )
        mp.setattr(vd, "tensor_model_parallel_all_reduce", lambda x: x)
        fn = torch.compile(
            lambda logits: su.vocab_parallel_argmax(logits, info),
            backend=capture,
            fullgraph=True,
        )
        fn(torch.randn(4, 256).to(torch.bfloat16 if onehot else torch.float32))
    torch._dynamo.reset()
    assert len(graphs) == 1
    return graphs[0]


@pytest.mark.parametrize("onehot", [False, True], ids=["all-gather", "onehot-ar"])
def test_fused_argmax_graph_is_the_same_on_every_rank(onehot):
    """With the shard offset as a device scalar (with_device_vocab_start) the
    two ranks trace the same graph, so TPU_PRECOMPILE_PEER_REPLAY can replay
    one rank's fused-loop artifact on the other. Control: the int offset
    bakes the rank into the graph and splits the fingerprint -- what made
    every rank compile every fused-loop bucket itself."""
    import vllm_torchtpu.spec_decode.utils as su
    from vllm_torchtpu.compilation.tpu_compiler import _graph_fingerprint

    def info(r, device_start):
        i = su.VocabParallelArgmax(
            r * 256, 256, 2, slot=su.rank_slot(r, 2, "cpu") if onehot else None
        )
        return su.with_device_vocab_start(i, "cpu") if device_start else i

    fp = {
        (r, d): _graph_fingerprint(_traced_argmax_graph(info(r, d), onehot))
        for r in (0, 1)
        for d in (False, True)
    }
    assert None not in fp.values()
    assert fp[(0, True)] == fp[(1, True)]
    assert fp[(0, False)] != fp[(1, False)]  # control: the int offset


@pytest.mark.parametrize("onehot", [False, True], ids=["all-gather", "onehot-ar"])
def test_device_vocab_start_gives_the_same_ids(monkeypatch, onehot):
    import vllm.distributed as vd

    import vllm_torchtpu.spec_decode.utils as su

    tp, width = 4, 300
    g = torch.Generator().manual_seed(3)
    dtype = torch.bfloat16 if onehot else torch.float32
    full = torch.randn(8, tp * width, generator=g).to(dtype)
    full[1, width - 1] = full[1, width + 1] = 50.0  # tie across shards
    shards = full.split(width, dim=-1)
    for device_start in (False, True):
        infos = []
        for r in range(tp):
            i = su.VocabParallelArgmax(
                r * width,
                width,
                tp,
                slot=su.rank_slot(r, tp, "cpu") if onehot else None,
            )
            infos.append(su.with_device_vocab_start(i, "cpu") if device_start else i)
        sent = []
        monkeypatch.setattr(
            vd,
            "tensor_model_parallel_all_gather",
            lambda x, dim=0, sent=sent: sent.append(x) or x.repeat(tp, 1),
        )
        monkeypatch.setattr(
            vd,
            "tensor_model_parallel_all_reduce",
            lambda x, sent=sent: sent.append(x) or x,
        )
        for r in range(tp):
            su.vocab_parallel_argmax(shards[r], infos[r])
        total = (
            sum(s.float() for s in sent).to(dtype) if onehot else torch.cat(sent, dim=0)
        )
        monkeypatch.setattr(
            vd, "tensor_model_parallel_all_gather", lambda x, dim=0, t=total: t
        )
        monkeypatch.setattr(
            vd, "tensor_model_parallel_all_reduce", lambda x, t=total: t
        )
        expected = torch.argmax(full, dim=-1).to(torch.int32)
        for r in range(tp):
            out = su.vocab_parallel_argmax(shards[r], infos[r])
            assert out.dtype == torch.int32 and torch.equal(out, expected), (
                device_start,
                r,
            )


def _bucket_runner(max_num_reqs=8):
    import types

    return types.SimpleNamespace(
        num_tokens_paddings=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        num_reqs_paddings=[1, 2, 4, 8, 16][: 1 + [1, 2, 4, 8, 16].index(max_num_reqs)],
        max_num_reqs=max_num_reqs,
        _dp_lockstep_enabled=lambda: False,
        mesh=None,
        device="cpu",
    )


@pytest.mark.parametrize("mns,expected", [(8, [4, 8]), (16, [4, 8, 16])])
def test_fused_loop_compiles_at_the_loop_buckets_only(monkeypatch, mns, expected):
    """The fused program is built with its own config whose compile_sizes are
    the loop buckets pad(num_reqs) <= pad(max_num_reqs), derived from the
    server's max_num_seqs -- [4, 8] at 8, [4, 8, 16] at 16, not the 11 token
    buckets up to 4096 -- and the draft's own config (draft and loop-step
    programs) keeps every bucket."""
    import contextlib
    import types

    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode import eagle3, mtp_fused_loop

    for name in ("set_vllm_model_wrapper_context", "set_model_tag"):
        monkeypatch.setattr(eagle3, name, lambda *a, **k: contextlib.nullcontext())
    seen = {}

    @contextlib.contextmanager
    def current(cfg):
        seen["current"] = cfg
        yield

    monkeypatch.setattr(eagle3, "set_current_vllm_config", current)

    def build(**kw):
        seen.update(kw)
        return types.SimpleNamespace(vocab_parallel_argmax=None)

    monkeypatch.setattr(mtp_fused_loop, "build_mtp_fused_loop", build)
    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_DRAFT_LOOP", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MTP_FUSED_INDEXSHARE", True, raising=False)

    prop = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    prop.runner = _bucket_runner(mns)
    prop.draft_model = types.SimpleNamespace(_dynamic_arg_dims={"input_ids": 0})
    prop.speculative_config = types.SimpleNamespace(
        method="mtp", num_speculative_tokens=3
    )
    prop._draft_kv_shared = False
    prop._share_mtp_indices = True
    prop._mtp_topk_buffers = [torch.zeros(16, 4, dtype=torch.int32)]
    prop._mtp_loop_model = object()
    all_sizes = list(prop.runner.num_tokens_paddings)
    cc = types.SimpleNamespace(compile_sizes=list(all_sizes))
    prop.draft_vllm_config = types.SimpleNamespace(compilation_config=cc)
    monkeypatch.setattr(
        eagle3.Eagle3Proposer, "advance_draft_positions", property(lambda self: True)
    )

    prop._build_mtp_fused_loop()

    assert prop._mtp_fused_loop is not None and prop._mtp_fused_indexshare
    cfg = seen["vllm_config"]
    assert cfg is seen["current"] and cfg is not prop.draft_vllm_config
    assert cfg.compilation_config.compile_sizes == expected
    assert prop._mtp_fused_loop_sizes == set(expected)
    assert cc.compile_sizes == all_sizes  # the draft's config is untouched
    # Every loop bucket a propose can pick is compiled: no chunk of up to
    # max_num_seqs requests falls back to the per-step loop.
    assert {prop._loop_bucket(n) for n in range(1, mns + 1)} <= set(expected)


def test_proposer_bucket_outside_the_compiled_sizes_falls_back(monkeypatch):
    """A token count the fused program was not compiled for never reaches it
    (the per-step loop runs instead), and the table is left alone."""
    p, k = 16, 32
    table = torch.full((32, k), -1, dtype=torch.int32)
    prog = _RecordingProgram(table)
    prop, _ = _proposer(monkeypatch, table, prog)
    prop._mtp_fused_loop_sizes = frozenset({4, 8})
    before = table.clone()
    assert _run(prop, p, torch.ones(p, k, dtype=torch.int32)) is None
    assert torch.equal(table, before) and not prog.calls
    assert _run(prop, 8, torch.ones(8, k, dtype=torch.int32)) is not None
