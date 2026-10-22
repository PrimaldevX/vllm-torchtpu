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
"""TPU_INDEXER_ROPE_NEOX_PERMUTE: half-split indexer rope == GPT-J indexer rope.

`VllmTPUIndexer.permute_rope_to_half_split` permutes each head's rope output
columns of `wq_b`, the k rope columns of `wk_weights_proj` and the matching
`k_norm` weight/bias entries by the same P; `forward` then runs the rope
NeoX-style. The streamindex_topk kernel scores

    score[t, s] = sum_h relu(q_fp8[t, h] . k[s]) * weights[t, h]

which is invariant under a common channel permutation of q and k. These tests
capture the `(q_fp8, k, weights)` triple the layer hands the kernel, with the
flag off and on, in the processed runtime layouts (bf16 unflipped / flipped,
FP8 with a blockwise scale), at decode (16) and mixed (4096) token counts, and
check: q_fp8 is exactly P of the old one per head, weights are unchanged, k
matches P of the old one to LayerNorm reassociation, and the scores match.
Two positive controls (q-only; wk without k_norm) must fail. On a TPU the
flag-on scores must select the same top-k keys.
"""

import contextlib
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter_mod
from vllm.config import CacheConfig
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

HEAD_DIM = 128
N_HEAD = 32  # GLM-5.2 index_n_heads
ROPE = 64
Q_LORA = 256
HIDDEN = 256
TOPK = 8
MAX_POS = 8192
FP8_BLOCK = 128
P = mla_op.mla_rope_half_split_perm(ROPE)


@pytest.fixture(autouse=True)
def _rope_path_without_fused_qprep(monkeypatch):
    """The rope paths this file compares; the fused q-prep kernel (on by default
    since trial 4, a TPU-only Pallas op) is covered by its own tests."""
    monkeypatch.setenv("TPU_INDEXER_QPREP_FUSED", "0")


class _StubIndexerCache(DeepseekV32IndexerCache):
    def __init__(self, head_dim, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class _Linear(torch.nn.Module):
    """A dense linear in a processed runtime layout (see the MLA test)."""

    def __init__(self, n_in, n_out, layout, gen):
        super().__init__()
        w = torch.randn(n_in, n_out, generator=gen) / math.sqrt(n_in)
        self.layout = layout
        if layout == "fp8":
            nb = n_in // FP8_BLOCK
            blocks = w.view(nb, FP8_BLOCK, n_out)
            scale = blocks.abs().amax(1) / 448.0
            q = (blocks / scale[:, None, :]).to(torch.float8_e4m3fn)
            self.weight = torch.nn.Parameter(q.view(n_in, n_out), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(
                scale[None, :, None, :].contiguous(), requires_grad=False
            )
            setattr(self, WEIGHT_FLIPPED_ATTR, True)
        elif layout == "flipped":
            self.weight = torch.nn.Parameter(w.bfloat16(), requires_grad=False)
            setattr(self, WEIGHT_FLIPPED_ATTR, True)
        else:
            self.weight = torch.nn.Parameter(
                w.t().contiguous().bfloat16(), requires_grad=False
            )

    def dense(self):
        if self.layout == "fp8":
            s = self.weight_scale[0, :, 0, :].repeat_interleave(FP8_BLOCK, 0)
            return self.weight.float() * s
        if self.layout == "flipped":
            return self.weight.float()
        return self.weight.float().t()

    def forward(self, x):
        return (x.float() @ self.dense()).to(torch.bfloat16), None


def _indexer(layout="unflipped", seed=0):
    """A real `VllmTPUIndexer` (in-tree `Indexer` + rebind, collaborators
    stubbed) whose projections are replaced by `_Linear`s in `layout`
    (wk_weights_proj is never quantized upstream: bf16 only) and whose k_norm
    has a non-trivial affine."""
    config = SimpleNamespace(
        index_topk=TOPK,
        index_n_heads=N_HEAD,
        index_head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE,
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=MAX_POS))
    with (
        patch.object(deepseek_v2, "DeepseekV32IndexerCache", _StubIndexerCache),
        patch.object(deepseek_v2, "SparseAttnIndexer", MagicMock()),
        patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        patch.object(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        _default_dtype(torch.bfloat16),
    ):
        ind = Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=HIDDEN,
            q_lora_rank=Q_LORA,
            quant_config=None,
            cache_config=CacheConfig(),
            topk_indices_buffer=torch.full((16, TOPK), -1, dtype=torch.int32),
            prefix="model.layers.0.self_attn.indexer",
        )
        ind = mla_op.VllmTPUIndexer.rebind(ind)
    gen = torch.Generator().manual_seed(seed)
    ind.wq_b = _Linear(Q_LORA, N_HEAD * HEAD_DIM, layout, gen)
    ind.wk_weights_proj = _Linear(
        HIDDEN, HEAD_DIM + N_HEAD, "flipped" if layout == "fp8" else layout, gen
    )
    with torch.no_grad():
        ind.k_norm.weight.copy_(1.0 + 0.3 * torch.randn(HEAD_DIM, generator=gen))
        ind.k_norm.bias.copy_(0.3 * torch.randn(HEAD_DIM, generator=gen))
    ind.indexer_op = MagicMock(side_effect=lambda *a: a)
    return ind


def _rope(is_neox_style=False):
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        return RotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )


def _inputs(num_tokens, seed=1):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen).bfloat16()
    qr = torch.randn(num_tokens, Q_LORA, generator=gen).bfloat16()
    positions = torch.randint(0, MAX_POS, (num_tokens,), generator=gen)
    return hidden, qr, positions


def _triple(ind, rope, hidden, qr, positions):
    with torch.no_grad():
        _, q_fp8, k, weights = ind(hidden, qr, positions, rope)
    return q_fp8, k, weights


def _scores(q_fp8, k, weights):
    per_head = torch.einsum("thd,sd->ths", q_fp8.float(), k.float()).clamp(min=0)
    return (per_head * weights.float().unsqueeze(-1)).sum(dim=1)


def _head_perm():
    """Within one head: P on the leading rope channels, identity after."""
    return torch.cat([P, torch.arange(ROPE, HEAD_DIM)])


def _relmax(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


DECODE_TOKENS = 16
MIXED_TOKENS = 4096


@pytest.mark.parametrize("layout", ["unflipped", "flipped", "fp8"])
@pytest.mark.parametrize(
    "num_tokens", [DECODE_TOKENS, MIXED_TOKENS], ids=["decode16", "mixed4096"]
)
def test_flag_on_matches_flag_off(layout, num_tokens):
    rope = _rope()
    off, on = _indexer(layout), _indexer(layout)
    on.permute_rope_to_half_split()
    assert on._rope_neox_permuted
    hidden, qr, positions = _inputs(num_tokens)
    q0, k0, w0 = _triple(off, rope, hidden, qr, positions)
    q1, k1, w1 = _triple(on, rope, hidden, qr, positions)
    hp = _head_perm()
    # weights_proj columns untouched; q per-head fp8 rows exactly P of the old
    # (same dot products, NeoX(Px) == P GPT-J(x), order-free amax).
    assert torch.equal(w1, w0)
    # (value compare: the full-head rope may turn a -0 into +0)
    assert torch.equal(q1.float(), q0[..., hp].float())
    # k: LayerNorm over permuted channels reassociates its mean/var sums.
    assert _relmax(k1, k0[..., hp]) < 2e-2, _relmax(k1, k0[..., hp])
    # Scores on a key set of the step's own tokens (cap the s axis for 4096).
    ks = slice(0, min(num_tokens, 512))
    s0, s1 = _scores(q0, k0[ks], w0), _scores(q1, k1[ks], w1)
    err = _relmax(s1, s0)
    print(f"{layout} T={num_tokens}: score rel max err {err:.2e}")
    assert err < 1e-2, err


def test_positive_control_q_only_fails():
    rope = _rope()
    off, bad = _indexer(), _indexer()
    mla_op.permute_linear_output_channels(
        bad.wq_b,
        mla_op.mla_rope_output_index(N_HEAD * HEAD_DIM, ROPE, HEAD_DIM, 0, N_HEAD),
    )
    bad._rope_neox_permuted = True
    hidden, qr, positions = _inputs(DECODE_TOKENS)
    s0 = _scores(*_triple(off, rope, hidden, qr, positions))
    s1 = _scores(*_triple(bad, rope, hidden, qr, positions))
    assert _relmax(s1, s0) > 0.05, _relmax(s1, s0)


def test_positive_control_k_norm_unpermuted_fails():
    """wq_b and wk permuted but k_norm's affine left alone: k is wrong."""
    rope = _rope()
    off, bad = _indexer(), _indexer()
    w_before = bad.k_norm.weight.detach().clone()
    b_before = bad.k_norm.bias.detach().clone()
    bad.permute_rope_to_half_split()
    with torch.no_grad():
        bad.k_norm.weight.copy_(w_before)
        bad.k_norm.bias.copy_(b_before)
    hidden, qr, positions = _inputs(DECODE_TOKENS)
    s0 = _scores(*_triple(off, rope, hidden, qr, positions))
    s1 = _scores(*_triple(bad, rope, hidden, qr, positions))
    assert _relmax(s1, s0) > 0.05, _relmax(s1, s0)


def test_permutation_is_idempotent_and_moves_scales():
    ind = _indexer("fp8")
    before_w = ind.wq_b.weight.detach().view(torch.uint8).clone()
    before_s = ind.wq_b.weight_scale.detach().clone()
    before_n = ind.k_norm.weight.detach().clone()
    ind.permute_rope_to_half_split()
    once = ind.wq_b.weight.detach().view(torch.uint8).clone()
    ind.permute_rope_to_half_split()
    assert torch.equal(ind.wq_b.weight.view(torch.uint8), once)
    idx = mla_op.mla_rope_output_index(N_HEAD * HEAD_DIM, ROPE, HEAD_DIM, 0, N_HEAD)
    assert torch.equal(once, before_w[:, idx])
    assert torch.equal(ind.wq_b.weight_scale, before_s[..., idx])
    assert torch.equal(ind.k_norm.weight, before_n[_head_perm()])


def test_rope_takes_the_full_head_path():
    """Flag on with a plain RotaryEmbedding: q and k are roped whole (no
    split/cat), on the module's own cos/sin cache; the GPT-J forward is not
    called."""
    rope = _rope()
    ind = _indexer()
    ind.permute_rope_to_half_split()
    hidden, qr, positions = _inputs(4)
    with (
        patch.object(
            mla_op, "rope_half_split_leading", wraps=mla_op.rope_half_split_leading
        ) as spy,
        patch.object(rope, "forward_native", side_effect=AssertionError),
    ):
        _triple(ind, rope, hidden, qr, positions)
    assert spy.call_count == 2


@pytest.mark.parametrize("shape", [(16, 32, 128), (4096, 32, 128), (16, 1, 64)])
def test_full_head_rope_equals_split_neox_bitwise(shape):
    """`rope_half_split_leading` == vLLM NeoX on the leading 64 channels +
    pass-through, bit for bit (up to the sign of zero) in bf16."""
    rope = _rope()
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(*shape, generator=gen).bfloat16()
    positions = torch.randint(0, MAX_POS, (shape[0],), generator=gen)
    got = mla_op.rope_half_split_leading(rope, positions, x, ROPE)
    pe, rest = x[..., :ROPE], x[..., ROPE:]
    ref_pe, _ = mla_op.rope_half_split(rope, positions, pe.clone(), pe[:, :1].clone())
    ref = torch.cat([ref_pe.reshape(pe.shape), rest], dim=-1)
    assert torch.equal(got.float(), ref.float())


def test_deepseek_scaling_rope_falls_back_to_split_path():
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        rope = mla_op.VllmDeepseekScalingRotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=False,
            scaling_factor=4.0,
            dtype=torch.bfloat16,
        )
    assert not mla_op.rope_half_split_leading_supported(rope)
    off, on = _indexer(), _indexer()
    on.permute_rope_to_half_split()
    hidden, qr, positions = _inputs(DECODE_TOKENS)
    s0 = _scores(*_triple(off, rope, hidden, qr, positions))
    s1 = _scores(*_triple(on, rope, hidden, qr, positions))
    assert _relmax(s1, s0) < 1e-2, _relmax(s1, s0)


class _Attn(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.zeros(1)


def _construct(indexer, indexer_rope):
    modules = MLAModules(
        kv_a_layernorm=torch.nn.Identity(),
        kv_b_proj=torch.nn.Identity(),
        rotary_emb=_rope(),
        o_proj=torch.nn.Identity(),
        fused_qkv_a_proj=torch.nn.Identity(),
        kv_a_proj_with_mqa=None,
        q_a_layernorm=torch.nn.Identity(),
        q_b_proj=torch.nn.Identity(),
        q_proj=None,
        indexer=indexer,
        is_sparse=True,
        topk_indices_buffer=torch.zeros((16, TOPK), dtype=torch.int32),
        indexer_rotary_emb=indexer_rope,
    )
    with patch.object(mla_op, "VllmTPUMLAAttention", return_value=_Attn()):
        return mla_op.VllmMultiHeadLatentAttentionWrapper(
            hidden_size=HIDDEN,
            num_heads=4,
            scale=1.0,
            qk_nope_head_dim=192,
            qk_rope_head_dim=ROPE,
            v_head_dim=256,
            q_lora_rank=Q_LORA,
            kv_lora_rank=512,
            mla_modules=modules,
        )


def test_flag_registers_only_the_indexer_hook(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", False)
    monkeypatch.setattr(envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", True)
    ind = _indexer()
    w = _construct(ind, _rope())
    assert w.mla_attn.post_load_hooks == [ind.permute_rope_to_half_split]
    assert not getattr(ind, "_rope_neox_permuted", False)


def test_both_flags_register_both_hooks(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", True)
    monkeypatch.setattr(envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", True)
    ind = _indexer()
    w = _construct(ind, _rope())
    assert w.mla_attn.post_load_hooks == [
        w.permute_rope_to_half_split,
        ind.permute_rope_to_half_split,
    ]


def test_flag_off_registers_nothing(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", False)
    monkeypatch.setattr(envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", False)
    w = _construct(_indexer(), _rope())
    assert not getattr(w.mla_attn, "post_load_hooks", [])


def test_flag_rejects_a_neox_indexer_rope(monkeypatch):
    monkeypatch.setenv("TPU_INDEXER_ROPE_NEOX_PERMUTE", "1")
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", False)
    monkeypatch.setattr(envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", True)
    with pytest.raises(NotImplementedError):
        _construct(_indexer(), _rope(is_neox_style=True))


def test_default_skips_a_neox_indexer_rope(monkeypatch):
    monkeypatch.delenv("TPU_INDEXER_ROPE_NEOX_PERMUTE", raising=False)
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", False)
    monkeypatch.setattr(envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", True)
    w = _construct(_indexer(), _rope(is_neox_style=True))
    assert not getattr(w.mla_attn, "post_load_hooks", [])


# --------------------------------------------------------------------------
# Kernel level (TPU): the triple through the real streamindex_topk kernel.
# --------------------------------------------------------------------------


def test_tpu_scores_select_the_same_tokens():
    """On the TPU, in bf16 with the cache-side per-row fp8 k quantization: the
    flag-on triple scores (relu-weighted, f32 accumulate) select the same
    top-64 keys as flag off, for a 16-row decode bucket and a 256-row chunk
    over 1024 keys. Not the streamindex kernel itself -- that is a function
    of the triple, which the tests above pin."""
    jax = pytest.importorskip("jax")
    if jax.devices()[0].platform != "tpu":
        pytest.skip("streamindex_topk needs a TPU")
    import jax.numpy as jnp

    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    rope = _rope()
    off, on = _indexer("fp8"), _indexer("fp8")
    on.permute_rope_to_half_split()
    n_keys = 1024
    hidden, qr, positions = _inputs(n_keys, seed=5)
    res = {}
    for mode, ind in (("off", off), ("on", on)):
        q_fp8, k, weights = _triple(ind, rope, hidden, qr, positions)
        # Cache-side quantization: one fp8 block per 128-wide k row.
        k_fp8, k_scale = quantize_tensor(k.float(), torch.float8_e4m3fn, axis=-1)
        kd = k_fp8.float() * k_scale
        qd = q_fp8.float()
        for name, rows in (
            ("decode", slice(n_keys - 16, n_keys)),
            ("chunk", slice(0, 256)),
        ):
            sc = (
                jnp.maximum(
                    jnp.einsum(
                        "thd,sd->ths",
                        jnp.asarray(qd[rows].numpy(), jnp.bfloat16),
                        jnp.asarray(kd.numpy(), jnp.bfloat16),
                        preferred_element_type=jnp.float32,
                    ),
                    0,
                )
                * jnp.asarray(weights[rows].float().numpy())[..., None]
            ).sum(1)
            res[(mode, name)] = np.asarray(jax.lax.top_k(sc, 64)[1])
    for name in ("decode", "chunk"):
        a, b = res[("off", name)], res[("on", name)]
        overlap = np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)])
        print(f"{name}: top-64 overlap off/on = {overlap:.4f}")
        assert overlap > 0.97, (name, overlap)
