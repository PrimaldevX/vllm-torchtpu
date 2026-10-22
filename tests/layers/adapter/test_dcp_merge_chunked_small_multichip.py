# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DCP_MERGE_CHUNKED_SMALL (chunk count of the gathered-once buckets)
over real collectives, 8 ranks, on the trial-4 DCP8 stack's merge env
(dcp_merge_multichip_common.S1_MERGE_ENV), eager and compiled.

Arms: off (8 chunks everywhere); s4 / s2 (SMALL=4 / 2: T <= 2048 splits into
4 / 2 chunks); ctl (s4 with every rank's lse rows rolled by one row before
the gather: must differ). T=1024 and 2048 take SMALL; T=4096 and 16 must be
bit for bit off (untouched). Splitting the token axis changes no per-row
arithmetic; the reduce-scatter only sees larger operands. Pass: 0 diffs
eager; compiled, each arm's max error against the f64 merge at most
ERR_SLACK x off's. Compiled bits are not compared: the bf16 reduce-scatter's
summation order follows the compiled schedule
(test_dcp_merge_rs_pipeline_multichip), and at T=1024 off's 128-row chunks
and s4 / s2's 256 / 512-row ones round about 16 % of the elements
differently, all within the same error of the f64 merge. The drafter
(is_backbone False) at 2048 never chunks: bit for bit.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import dcp_merge_multichip_common as vm  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
TOKENS = (1024, 2048, 4096, 16)
ARMS = {"s4": 4, "s2": 2}
LAUNCH_TIMEOUT = 3000
# Compiled: an arm's max abs error vs the f64 merge, over off's (measured
# 1.03 at most on TPU).
ERR_SLACK = 1.5


def _set(cp_mla, small=0):
    vm._set(cp_mla, TPU_DCP_MERGE_CHUNKED_SMALL=small)


def _reference(rank, world, tokens, lses):
    """This rank's heads of the merge in f64: sum_r w_r o_r, zero-weight
    (possibly NaN) partials dropped."""
    import torch

    local = vm.HEADS // world
    heads = slice(rank * local, (rank + 1) * local)
    merged = torch.logsumexp(lses.double(), dim=0)
    acc = None
    for r in range(world):
        o = vm._partial(r, tokens, lses, world)[:, heads].double()
        w = torch.exp(lses[r].double() - merged)[:, heads].unsqueeze(-1)
        x = torch.where(w > 0, o * w, torch.zeros_like(o))
        acc = x if acc is None else acc + x
    return acc


def _max_err(got, ref):
    return float((got.double() - ref).abs().nan_to_num(float("inf")).max())


def _check(rank, world, cp_mla):
    import torch

    report = {}
    real_gather = cp_mla._gather_lse

    def rolled(lse_f32, group):
        return real_gather(torch.roll(lse_f32, 1, dims=0), group)

    for tokens in TOKENS:
        lses = vm._lses(tokens, world)
        o = vm._partial(rank, tokens, lses, world).to("tpu")
        lse = lses[rank].contiguous().to("tpu")
        ref = _reference(rank, world, tokens, lses)
        qsl = torch.tensor([0, tokens], dtype=torch.int32).to("tpu")

        def fn(o, lse, qsl=qsl, backbone=True):
            return cp_mla.merge_lse_partials_scatter_heads(
                o, lse, is_backbone=backbone, query_start_loc=qsl
            )

        for how in ("compiled", "eager"):
            key = f"T{tokens}/{how}"
            try:
                comp = how == "compiled"

                def run(f, comp=comp):
                    return vm._compile(f) if comp else f

                _set(cp_mla)
                off = run(fn)(o, lse).cpu()
                c = {"tokens": tokens, "elements": int(off.numel())}
                c["off_err"] = _max_err(off, ref)
                for arm, small in ARMS.items():
                    _set(cp_mla, small)
                    c[f"{arm}_chunks"] = cp_mla._merge_chunks(tokens)
                    got = run(fn)(o, lse).cpu()
                    u = vm._ulp(got, off)
                    c[f"{arm}_ndiff"] = int((u > 0).sum())
                    c[f"{arm}_maxulp"] = int(u.max())
                    c[f"{arm}_err"] = _max_err(got, ref)
                if 16 < tokens <= 2048:
                    _set(cp_mla, 4)
                    cp_mla._gather_lse = rolled
                    try:
                        bad = run(fn)(o, lse).cpu()
                    finally:
                        cp_mla._gather_lse = real_gather
                    c["ctl_ndiff"] = int((vm._ulp(bad, off) > 0).sum())
                if tokens == 2048 and comp:

                    def drafter(o, lse):
                        return fn(o, lse, backbone=False)

                    _set(cp_mla)
                    d_off = vm._compile(drafter)(o, lse).cpu()
                    _set(cp_mla, 4)
                    d_on = vm._compile(drafter)(o, lse).cpu()
                    c["drafter_ndiff"] = int((vm._ulp(d_on, d_off) > 0).sum())
                report[key] = c
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
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="chunked_small_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        result["check"] = _check(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_dcp_merge_chunked_small_on_real_collectives(tmp_path):
    import subprocess

    import test_cp_mla_merge_multichip as mm

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
        timeout=LAUNCH_TIMEOUT,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-6000:],
        (done.stderr or "")[-6000:],
    )
    total = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(key, {})
            for k, v in c.items():
                if k.endswith("ndiff") or k == "elements":
                    t[k] = t.get(k, 0) + v
                elif k.endswith(("maxulp", "_err")):
                    t[k] = max(t.get(k, 0), v)
                else:
                    t[k] = v
    failures = []
    for key, t in sorted(total.items()):
        print(f"CHUNKSMALL {key}: {json.dumps(t)}")
        if key.endswith("/compiled"):
            # The bf16 reduce-scatter's summation order follows the compiled
            # schedule (test_dcp_merge_rs_pipeline_multichip): held to the f64
            # merge, not to off's bits.
            ok = all(t[f"{a}_err"] <= ERR_SLACK * t["off_err"] for a in ARMS)
        else:
            ok = all(t[f"{a}_ndiff"] == 0 for a in ARMS)
        if 16 < t["tokens"] <= 2048:
            ok &= t["s4_chunks"] == 4 and t["s2_chunks"] == 2
            ok &= t["ctl_ndiff"] > 1e-3 * t["elements"]
        else:
            ok &= all(t[f"{a}_ndiff"] == 0 for a in ARMS)  # untouched buckets
        if "drafter_ndiff" in t:
            ok &= t["drafter_ndiff"] == 0
        if not ok:
            failures.append((key, t))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
