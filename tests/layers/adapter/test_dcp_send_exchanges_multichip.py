# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The send-layout exchanges over real collectives, eager and compiled.

On 8 ranks, each rank builds a synthetic kernel send buffer
(`sparse_mla_kernel.send_layout_shape`: per destination SEND_PAD_ROWS - 1 rows
of junk, the lse word row, then T * local partial rows, the padding tokens'
rows junk too) and merges it through `merge_send_layout` four ways: the
ragged exchange (TPU_DCP_MLA_SEND_LAYOUT_UNPACK, the served path), the ragged
exchange with torch views (no flag), the row-group ragged exchange
(TPU_DCP_MLA_SEND_GROUP_ROWS=8 / 16) and the Pallas small all-to-all
(TPU_DCP_MLA_SEND_SMALL_A2A). Every output element must agree bit for
bit, at decode buckets 4 / 8 / 16 with all, all-but-one and one real token(s).

Positive controls the comparison must catch: the row-group and small ops
handed every row as live (padding tokens' junk merged instead of zero), and
the send buffer with its lse word row's lo halves zeroed.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_cp_mla_merge_multichip as mm  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
HEADS = 32
VDIM = 512
CASES = [(4, 4), (4, 3), (4, 1), (8, 8), (8, 7), (8, 1), (16, 16), (16, 12), (16, 1)]
MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    "TPU_DCP_MLA_SEND_LAYOUT": True,
    # TPU_DCP_MLA_SEND_SMALL_A2A's preconditions; its cap covers every case.
    "TPU_DCP_INDEXER_SMALL_AG": True,
    "TPU_SC_TOPK_DCP_MERGE_FLAT": True,
    "TPU_DCP_RANK_OPERAND": True,
    "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS": 16,
}


def _send_buffer(rank, world, tokens, real, pad):
    """One rank's int16 `[world, pad + tokens * local, V]` send buffer."""
    import numpy as np

    local = HEADS // world
    tl = tokens * local
    rng = np.random.default_rng(7000 + 97 * rank + 13 * tokens + real)
    buf = rng.integers(-32768, 32767, size=(world, pad + tl, VDIM), dtype=np.int16)
    o = rng.standard_normal((world, tl, VDIM)).astype(np.float32)
    o[:, real * local :] *= 1e30  # padding tokens: finite junk that would show
    bf = (o.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    buf[:, pad:] = bf
    lse = (rng.standard_normal((world, tl)) * 4).astype(np.float32)
    if rank == 3:
        lse[:, ::5] = -np.inf  # rows this rank owns no position of
    bits = lse.view(np.int32)
    buf[:, pad - 1, :tl] = (bits >> 16).astype(np.int16)
    buf[:, pad - 1, tl : 2 * tl] = (bits & 0xFFFF).astype(np.uint16).view(np.int16)
    return buf


def _check(rank, world, cp_mla):
    import torch

    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS as PAD

    for k, v in MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    local = HEADS // world
    report = {}
    for tokens, real in CASES:
        tl = tokens * local
        send = torch.from_numpy(_send_buffer(rank, world, tokens, real, PAD))
        bad = send.clone()
        bad[:, PAD - 1, tl : 2 * tl] = 0
        qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
        qsl_all = torch.tensor([0, tokens], dtype=torch.int32).to("tpu")
        for how in ("eager", "compiled"):
            key = f"T{tokens}_r{real}/{how}"
            try:
                wrap = mm._compile if how == "compiled" else (lambda f: f)

                def run(s, q, unpack, group=0, tokens=tokens, wrap=wrap, small=False):
                    cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = unpack
                    cp_mla.envs.TPU_DCP_MLA_SEND_GROUP_ROWS = group
                    cp_mla.envs.TPU_DCP_MLA_SEND_SMALL_A2A = small
                    try:
                        f = wrap(lambda s, q: cp_mla.merge_send_layout(s, q, tokens))
                        return f(s.contiguous().to("tpu"), q).cpu()
                    finally:
                        cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = False
                        cp_mla.envs.TPU_DCP_MLA_SEND_GROUP_ROWS = 0
                        cp_mla.envs.TPU_DCP_MLA_SEND_SMALL_A2A = False

                want = run(send, qsl, True)  # served: ragged + unpack op
                plain = run(send, qsl, False)  # ragged, torch views
                got_g8 = run(send, qsl, True, 8)
                got_g16 = run(send, qsl, True, 16)
                ctl_g8_rows = run(send, qsl_all, True, 8)  # no padding mask
                ctl_g8_lse = run(bad, qsl, True, 8)
                # The served stack sets UNPACK + GROUP_ROWS=8 too; SMALL wins.
                got_sm = run(send, qsl, True, 8, small=True)
                ctl_sm_rows = run(send, qsl_all, True, 8, small=True)
                ctl_sm_lse = run(bad, qsl, True, 8, small=True)

                def nd(a, want=want):
                    return int((a.view(torch.int16) != want.view(torch.int16)).sum())

                report[key] = {
                    "elements": int(want.numel()),
                    "plain_n_diff": nd(plain),
                    "nonzero_real": bool((want[:real] != 0).any()),
                    "pad_rows_zero": bool((want[real:] == 0).all()),
                    "g8_n_diff": nd(got_g8),
                    "g16_n_diff": nd(got_g16),
                    "ctl_g8_rows_n_diff": nd(ctl_g8_rows),
                    "ctl_g8_lse_n_diff": nd(ctl_g8_lse),
                    "sm_n_diff": nd(got_sm),
                    "ctl_sm_rows_n_diff": nd(ctl_sm_rows),
                    "ctl_sm_lse_n_diff": nd(ctl_sm_lse),
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    return report


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="t5d_send_exchanges",
        )
        cp_mla.get_dcp_group = lambda: group
        from vllm_torchtpu.distributed import dcp as dcp_mod

        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world, SEND_PAD_ROWS - 1)
        cp_mla.prebuild_send_unpack(world)
        cp_mla.prebuild_send_group(world, 8)
        cp_mla.prebuild_send_group(world, 16)
        cp_mla.prebuild_send_small(world)
        result["check"] = _check(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_send_group_and_small_match_ragged_on_real_collectives(tmp_path):
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--result-dir",
        str(tmp_path),
    ]
    done = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=mm._prepare_worker_env(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=mm.LAUNCH_TIMEOUT_SECONDS,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-8000:],
    )
    total: dict[str, dict] = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key,
                dict(
                    elements=0,
                    plain=0,
                    nonzero=False,
                    pad_zero=True,
                    g8=0,
                    g16=0,
                    ctl_g8_rows=0,
                    ctl_g8_lse=0,
                    sm=0,
                    ctl_sm_rows=0,
                    ctl_sm_lse=0,
                ),
            )
            t["elements"] += c["elements"]
            t["plain"] += c["plain_n_diff"]
            t["nonzero"] |= c["nonzero_real"]
            t["pad_zero"] &= c["pad_rows_zero"]
            t["g8"] += c["g8_n_diff"]
            t["g16"] += c["g16_n_diff"]
            t["ctl_g8_rows"] += c["ctl_g8_rows_n_diff"]
            t["ctl_g8_lse"] += c["ctl_g8_lse_n_diff"]
            t["sm"] += c["sm_n_diff"]
            t["ctl_sm_rows"] += c["ctl_sm_rows_n_diff"]
            t["ctl_sm_lse"] += c["ctl_sm_lse_n_diff"]
    for key, t in sorted(total.items()):
        print(f"SENDEXCH {key}: {t}")
        tokens, real = (int(x[1:]) for x in key.split("/")[0].split("_"))
        assert t["plain"] == 0, ("ragged torch-view path drifted", key, t)
        assert t["g8"] == 0, ("TPU_DCP_MLA_SEND_GROUP_ROWS=8 not bitwise", key, t)
        assert t["g16"] == 0, ("TPU_DCP_MLA_SEND_GROUP_ROWS=16 not bitwise", key, t)
        assert t["ctl_g8_lse"] > 0, ("row-group lse control not caught", key, t)
        assert t["sm"] == 0, ("TPU_DCP_MLA_SEND_SMALL_A2A not bitwise", key, t)
        assert t["ctl_sm_lse"] > 0, ("small a2a lse control not caught", key, t)
        assert t["nonzero"] and t["pad_zero"], (key, t)
        if real < tokens:
            assert t["ctl_g8_rows"] > 0, ("row-group padding rows not caught", key, t)
            assert t["ctl_sm_rows"] > 0, ("small a2a padding rows not caught", key, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
