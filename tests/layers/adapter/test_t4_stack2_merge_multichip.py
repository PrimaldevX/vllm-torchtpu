# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Trial-4 stack (t4-stack2), merge half: the backbone DCP merge of real
all-off vs all-on attention partials over real collectives (8 ranks), eager
and compiled, with merge pipelining composed onto the bf16 wire.

Reads `T4S2_PARTIALS`, the `<T4S2_DUMP>.mixed.npz` test_t4_stack2_composition
writes (mixed-256 and mixed-2048; the on partials there come from the merge
pieces kernel). With the s25 merge env (a2a <= 64 tokens, chunk8 + collapsed
lse gather <= 2048), arms per step:

* off: off partials; PACK / REAL_ROWS / RS_BF16 / WEIGHTED_HM / PIECES off.
* t3: on partials; PACK + REAL_ROWS + TPU_DCP_MERGE_RS_BF16=1 (trial 3).
* mp: on partials; PACK + REAL_ROWS + WEIGHTED_HM + MERGE_PIECES=2, f32 wire
  (merge pipelining alone): bit for bit the off arm.
* on: t3 + WEIGHTED_HM + MERGE_PIECES=2 (the t4-stack2 merge): the bf16
  heads-major operand feeding the bf16 reduce-scatter.
* ctl: on with every rank's lse rolled by one token.

The unchunked step (256) must be bit for bit off in every arm. The chunked
step (2048): `on` is held to RS_BF16's own bound (7 bf16 scale-ulps over the
off arm's error vs f64, as test_t3_final_merge_multichip), and its
differing-element count against `t3` is reported (same bf16 operand values;
0 expected unless the heads-major reduce-scatter sums in another order).
`ctl` must break the bound.
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
import test_t3_final_merge_multichip as t3m  # noqa: E402
import test_t4_combined_merge_multichip as t4m  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
ARMS = {
    # name: (partials tag, RS_BF16, WEIGHTED_HM + PIECES, pack + real rows)
    "off": ("off", 0, False, False),
    "t3": ("on", 1, False, True),
    "mp": ("on", 0, True, True),
    "on": ("on", 1, True, True),
}


def _set(cp_mla, rs_bf16, mp, pack):
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_PACK = pack
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS = pack
    cp_mla.envs.TPU_DCP_MERGE_RS_BF16 = rs_bf16
    cp_mla.envs.TPU_DCP_MERGE_WEIGHTED_HM = mp
    cp_mla.envs.TPU_DCP_MLA_MERGE_PIECES = 2 if mp else 0
    cp_mla.envs.TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS = 2048
    cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = False


def _check(rank, cp_mla, path):
    import numpy as np
    import torch

    z = np.load(path)
    dcp = int(z["dcp"])
    assert dcp == WORLD_SIZE, dcp
    for k, v in t3m.S25_MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    merge = cp_mla.merge_lse_partials_scatter_heads
    report = {}
    for step in range(int(z["steps"])):
        real = int(z[f"s{step}_real"])
        for how in ("eager", "compiled"):
            key = f"s{step}/{how}"
            try:
                qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")

                def fn0(o, lse, qsl=qsl):
                    return merge(o, lse, query_start_loc=qsl)

                res = {}
                for name, (tag, rs, mp, pack) in [
                    *ARMS.items(),
                    ("ctl", ("on", 1, True, True)),
                ]:
                    _set(cp_mla, rs, mp, pack)
                    fn = fn0 if how == "eager" else mm._compile(fn0)
                    o, lse = t4m._partials(z, step, tag, rank, dcp, name == "ctl")
                    res[name] = fn(o.to("tpu"), lse.to("tpu")).cpu()[:real]
                tokens = int(real)
                outs, lses = t3m._all_partials(z, step, "off", dcp, real)
                c = {
                    "tokens": tokens,
                    "elements": int(res["off"].numel()),
                    "chunked": bool(
                        cp_mla._merge_chunks(int(z[f"s{step}_off_out"].shape[0]) // dcp)
                        > 1
                    ),
                }
                for name in ("t3", "mp", "on", "ctl"):
                    u = t4m._ulp(res[name], res["off"])
                    c[f"{name}_vs_off_ndiff"] = int((u > 0).sum())
                    c[f"{name}_vs_off_maxulp"] = int(u.max()) if u.numel() else 0
                u = t4m._ulp(res["on"], res["t3"])
                c["on_vs_t3_ndiff"] = int((u > 0).sum())
                c["on_vs_t3_maxulp"] = int(u.max()) if u.numel() else 0
                for name in ("off", "t3", "on", "ctl"):
                    c[f"{name}_scale"] = t3m._scale_err(
                        res[name], outs, lses, rank, dcp
                    )["max_ulp"]
                report[key] = c
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:400]}
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
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="t4s2_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        result["check"] = _check(rank, cp_mla, path)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_t4_stack2_merge_on_real_collectives(tmp_path):
    path = os.environ.get("T4S2_PARTIALS")
    if not path or not os.path.exists(path):
        pytest.skip("T4S2_PARTIALS (npz from test_t4_stack2_composition) unset")
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
            t = total.setdefault(key, {"tokens": c["tokens"], "chunked": c["chunked"]})
            for k, v in c.items():
                if k in ("tokens", "chunked"):
                    continue
                if k.endswith("ndiff") or k == "elements":
                    t[k] = t.get(k, 0) + v
                else:
                    t[k] = max(t.get(k, 0), v)
    failures = []
    for key, t in sorted(total.items()):
        print(f"T4S2MERGE {key}: {json.dumps(t)}")
        ok = t["mp_vs_off_ndiff"] == 0
        if t["chunked"]:
            bound = t["off_scale"] + t3m.RS_BF16_MAX_SCALE_ULP
            ok &= t["on_scale"] <= bound and t["ctl_scale"] > bound
        else:
            # The new flags only touch the chunked merge.
            ok &= t["on_vs_t3_ndiff"] == 0 and t["ctl_vs_off_ndiff"] > 0
            if key.endswith("eager"):
                ok &= t["on_vs_off_ndiff"] == 0
            else:
                ok &= t["on_vs_off_maxulp"] <= 1
        if not ok:
            failures.append((key, t))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--partials", required=True)
    a = p.parse_args()
    _run_worker(a.result_dir, a.partials)
