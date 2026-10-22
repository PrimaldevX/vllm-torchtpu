# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DCP_MLA_SEND_LAYOUT merge over real collectives, eager and compiled.

Reads `T5SL_PARTIALS`, the npz `tests/kernels/test_dcp_mla_send_layout.py`
writes (`T5SL_DUMP`): per decode step, every rank's incumbent partial output
and lse (prologue on) and the kernel-written send buffer of the same step. On
8 ranks with the served merge env, each rank merges the incumbent partials
through `merge_lse_partials_scatter_heads` (a2a + PACK + REAL_ROWS) and the
send buffer through `merge_send_layout` (one ragged a2a from row
SEND_PAD_ROWS - 1), eager and compiled, with the torch merge and with
TPU_DCP_MLA_SEND_LAYOUT_UNPACK (exchange + unpack as one JAX op): every row
must agree bit for bit.
Positive control: the send buffer with its lse word row's lo halves zeroed
must fail the same check.
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
MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
}


def _check(rank, cp_mla, path):
    import numpy as np
    import torch

    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    z = np.load(path)
    dcp = int(z["dcp"])
    assert dcp == WORLD_SIZE, dcp
    for k, v in MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    report = {}
    for step in range(int(z["steps"])):
        out = torch.from_numpy(z[f"s{step}_on_out"].view(np.int16).copy())
        lse = torch.from_numpy(z[f"s{step}_on_lse"].view(np.float32).copy())
        send = torch.from_numpy(z[f"s{step}_send"].view(np.int16).copy())
        real = int(z[f"s{step}_real"])
        tokens = out.shape[0] // dcp
        o = out.view(torch.bfloat16).reshape(dcp, tokens, *out.shape[1:])[rank]
        lr = lse.reshape(dcp, tokens, -1)[rank]
        s = send.reshape(dcp, dcp, *send.shape[1:])[rank]
        rows = tokens * (o.shape[1] // dcp)
        bad = s.clone()
        bad[:, SEND_PAD_ROWS - 1, rows : 2 * rows] = 0
        qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
        for how in ("eager", "compiled"):
            key = f"s{step}/{how}"
            try:

                def inc(o, lse, qsl=qsl):
                    return cp_mla.merge_lse_partials_scatter_heads(
                        o, lse, query_start_loc=qsl
                    )

                def new(s, qsl=qsl, tokens=tokens):
                    return cp_mla.merge_send_layout(s, qsl, tokens)

                wrap = mm._compile if how == "compiled" else (lambda f: f)
                want = wrap(inc)(
                    o.contiguous().to("tpu"), lr.contiguous().to("tpu")
                ).cpu()
                got = wrap(new)(s.contiguous().to("tpu")).cpu()
                ctl = wrap(new)(bad.contiguous().to("tpu")).cpu()
                # TPU_DCP_MLA_SEND_LAYOUT_UNPACK: exchange + unpack as one JAX
                # op, torch merge.
                cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = True
                got_u = wrap(new)(s.contiguous().to("tpu")).cpu()
                ctl_u = wrap(new)(bad.contiguous().to("tpu")).cpu()
                cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = False
                report[key] = {
                    "tokens": tokens,
                    "real": real,
                    "elements": int(want.numel()),
                    "n_diff": int(
                        (got.view(torch.int16) != want.view(torch.int16)).sum()
                    ),
                    "nonzero_real": bool((want[:real] != 0).any()),
                    "control_n_diff": int(
                        (ctl.view(torch.int16) != want.view(torch.int16)).sum()
                    ),
                    "unpack_n_diff": int(
                        (got_u.view(torch.int16) != want.view(torch.int16)).sum()
                    ),
                    "unpack_control_n_diff": int(
                        (ctl_u.view(torch.int16) != want.view(torch.int16)).sum()
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    return report


def _run_worker(result_dir: Path, path: str) -> None:
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
            group_name="t5sl_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        from vllm_torchtpu.distributed import dcp as dcp_mod

        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        cp_mla.prebuild_ragged_all_to_all(world, SEND_PAD_ROWS - 1)
        cp_mla.prebuild_send_unpack(world)
        result["check"] = _check(rank, cp_mla, path)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_send_layout_merge_on_real_collectives(tmp_path):
    path = os.environ.get("T5SL_PARTIALS")
    if not path or not os.path.exists(path):
        pytest.skip("T5SL_PARTIALS (npz from test_dcp_mla_send_layout) unset")
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
        "--partials",
        path,
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
    total = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key,
                dict(
                    elements=0,
                    n_diff=0,
                    ctl=0,
                    nonzero=False,
                    unpack=0,
                    unpack_ctl=0,
                ),
            )
            t["unpack"] += c["unpack_n_diff"]
            t["unpack_ctl"] += c["unpack_control_n_diff"]
            t["elements"] += c["elements"]
            t["n_diff"] += c["n_diff"]
            t["ctl"] += c["control_n_diff"]
            t["nonzero"] |= c["nonzero_real"]
    for key, t in sorted(total.items()):
        print(f"SENDMERGE {key}: {t}")
        assert t["n_diff"] == 0, (key, t)
        assert t["nonzero"], (key, t)
        assert t["ctl"] > 0, ("dropped lse lo halves not caught", key, t)
        assert t["unpack"] == 0, ("TPU_DCP_MLA_SEND_LAYOUT_UNPACK not bitwise", key, t)
        assert t["unpack_ctl"] > 0, ("unpack control not caught", key, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--partials", required=True)
    a = p.parse_args()
    _run_worker(a.result_dir, a.partials)
