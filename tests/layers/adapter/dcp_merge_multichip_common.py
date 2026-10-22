# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared pieces of the 8-rank DCP merge multichip tests: the trial-4 DCP8
stack's merge env, the synthetic partials and their lse, and ulp counting."""

WORLD_SIZE = 8
HEADS = 64
C = 512

S1_MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    "TPU_DCP_MERGE_CHUNKED": 8,
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 1024,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 2048,
    "TPU_DCP_MERGE_RS_BF16": 1,
    "TPU_DCP_MERGE_WEIGHTED_HM": True,
    "TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS": 2048,
    "TPU_DCP_MLA_MERGE_PIECES": 2,
    "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS": 2048,
    "TPU_MTP_DCP_MERGE_SIZED": False,
}


def _lses(tokens, world=WORLD_SIZE, seed=20261014):
    import torch

    g = torch.Generator().manual_seed(seed + tokens)
    lses = torch.randn(world, tokens, HEADS, generator=g) * 4 + 3
    # Ranks that own none of a row's top-k: -inf (about 6 % of entries).
    own_none = torch.rand(world, tokens, HEADS, generator=g) < 0.06
    lses[own_none] = float("-inf")
    lses[:, 3] = float("-inf")  # nobody owns token 3
    lses[1:, 2] = float("-inf")  # rank 0 owns token 2 alone
    lses[:, 0] += 200.0
    return lses


def _block(r, hb, tokens, local, lses):
    """Rank r's partial for head block hb, `[T, local, C]` bf16, NaN where
    rank r's lse is -inf on some of those rows (zero-weight dummy-key rows)."""
    import torch

    g = torch.Generator().manual_seed(1_000_003 * (r + 1) + 7919 * hb + tokens)
    o = torch.randn(tokens, local, C, generator=g).to(torch.bfloat16)
    dead = ~torch.isfinite(lses[r, :, hb * local : (hb + 1) * local])
    o[dead & (torch.arange(tokens)[:, None] % 2 == 0)] = float("nan")
    return o


def _partial(rank, tokens, lses, world=WORLD_SIZE):
    import torch

    local = HEADS // world
    return torch.cat(
        [_block(rank, hb, tokens, local, lses) for hb in range(world)], dim=1
    ).contiguous()


def _ulp(a, b):
    import torch

    def line(x):
        x = x.view(torch.int16).to(torch.int32)
        return torch.where(x < 0, -(x & 0x7FFF), x)

    return (line(a) - line(b)).abs()


def _set(cp_mla, **kw):
    """The trial-4 DCP8 stack's merge env (`S1_MERGE_ENV`) with `kw` on top."""
    env = dict(S1_MERGE_ENV)
    env.update(kw)
    for k, v in env.items():
        setattr(cp_mla.envs, k, v)


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )
