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
"""The MTP drafter's sized DCP merge with the Pallas small all-to-all
(`TPU_MTP_DCP_MERGE_SMALL_A2A`) beside the sized merge over XLA's all-to-all.

On one host (8 ranks), compiled:

  1. Bitwise: at T <= the small cap (4, 8) the small-a2a op's output equals the
     sized op's bit for bit, including -inf rows, a rank owning nothing and NaN
     partials of weight 0; above the cap (16, 24) the op keeps XLA's path (also
     bitwise).
  2. Positive control: every received block shifted by one row inside the
     small exchange are not bitwise (and not within 1 ulp of the f64 merge).
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cp_mla_merge_multichip as base  # noqa: E402

WORLD_SIZE = base.WORLD_SIZE
HEADS = base.HEADS
CAP = 32
SMALL_CAP = 8


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _partials(world, tokens):
    # base._global_partials needs >= 5 tokens (its special rows 0-4); take the
    # first `tokens` rows of an 8-token draw below that.
    outputs, lses = base._global_partials(world, max(tokens, 8))
    return outputs[:, :tokens].contiguous(), lses[:, :tokens].contiguous()


def _arm(cp_mla, world, small):
    cp_mla.envs.TPU_MTP_DCP_MERGE_SMALL_A2A = small
    cp_mla._SIZED_MERGE_OPS.pop(world, None)
    cp_mla.prebuild_sized_merge(world)
    return lambda o, lse: cp_mla.merge_lse_partials_scatter_heads(
        o, lse, is_backbone=False
    )


def _check(rank, world, cp_mla):
    import torch

    report = {}
    local = HEADS // world
    for tokens in (4, 8, 16, 24):
        outputs, lses = _partials(world, tokens)
        poisoned = outputs.clone()
        poisoned[~torch.isfinite(lses)] = float("nan")
        ref = base._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
        o_dev, p_dev = outputs[rank].to("tpu"), poisoned[rank].to("tpu")
        l_dev = lses[rank].to("tpu")
        got = {}
        for arm, small in (("sized", False), ("small", True)):
            key = f"{arm}@{tokens}"
            try:
                fn = _compile(_arm(cp_mla, world, small))
                y, yp = fn(o_dev, l_dev).cpu(), fn(p_dev, l_dev).cpu()
                got[arm] = (y, yp)
                c = {"shape": list(y.shape), **base._error_stats(y, ref)}
                c["nan_input_bitwise"] = bool(
                    torch.equal(yp.view(torch.int16), y.view(torch.int16))
                )
                if arm == "small":
                    c["bitwise_vs_sized"] = bool(
                        torch.equal(
                            y.view(torch.int16), got["sized"][0].view(torch.int16)
                        )
                    )
                    c["nan_bitwise_vs_sized"] = bool(
                        torch.equal(
                            yp.view(torch.int16), got["sized"][1].view(torch.int16)
                        )
                    )
                report[key] = c
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    # Positive control: rotate the received blocks inside the small exchange.
    real = cp_mla._small_exchange

    def rotated(send, **kw):
        import jax.numpy as jnp

        return jnp.roll(real(send, **kw), 1, axis=1)

    cp_mla._small_exchange = rotated
    try:
        outputs, lses = _partials(world, 4)
        ref = base._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
        sized = _compile(_arm(cp_mla, world, False))(
            outputs[rank].to("tpu"), lses[rank].to("tpu")
        ).cpu()
        y = _compile(_arm(cp_mla, world, True))(
            outputs[rank].to("tpu"), lses[rank].to("tpu")
        ).cpu()
        c = base._error_stats(y, ref)
        c["bitwise_vs_sized"] = bool(
            torch.equal(y.view(torch.int16), sized.view(torch.int16))
        )
        report["control_rotated/compiled@4"] = c
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report["control_rotated/compiled@4"] = {"error": repr(exc)[:600]}
    finally:
        cp_mla._small_exchange = real
    return report


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "world": world}
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
        from vllm_torchtpu.distributed import dcp
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="mtp_dcp_merge_small_test",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp.get_dcp_group = lambda: group
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A = False
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS = CAP
        cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = True
        cp_mla.envs.TPU_DCP_RANK_OPERAND = True
        cp_mla.envs.TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS = SMALL_CAP
        cp_mla.envs.TPU_SMALL_COLL_HBM_OPERAND = (
            os.environ.get("TPU_SMALL_COLL_HBM_OPERAND", "1") == "1"
        )
        result["check"] = _check(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2))
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
    if "error" in result:
        raise SystemExit(1)


def assert_results(results):
    errors = [r for r in results if "error" in r]
    assert not errors, errors
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            if key.startswith("control"):
                assert not c["bitwise_vs_sized"], (
                    "control not caught",
                    r["rank"],
                    key,
                    c,
                )
                assert not c["within_1ulp"], ("control not caught", r["rank"], key, c)
                continue
            assert c["within_1ulp"] and c["nan_input_bitwise"], (r["rank"], key, c)
            if key.startswith("small"):
                assert c["bitwise_vs_sized"] and c["nan_bitwise_vs_sized"], (
                    r["rank"],
                    key,
                    c,
                )


def test_small_a2a_sized_merge_bitwise(tmp_path):
    result_dir = tmp_path / "small"
    result_dir.mkdir()
    import subprocess

    cmd = base._worker_command(result_dir, [])
    cmd[cmd.index(str(Path(base.__file__).resolve()))] = str(Path(__file__).resolve())
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=base._prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=base.LAUNCH_TIMEOUT_SECONDS,
    )
    results = base._results(result_dir, WORLD_SIZE)
    print(json.dumps(results[0], indent=1))
    if any("error" in r for r in results):
        print(completed.stdout[-12000:])
    assert_results(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--bench-tokens", default="")
    a = p.parse_args()
    a.result_dir.mkdir(parents=True, exist_ok=True)
    _run_worker(a.result_dir)
