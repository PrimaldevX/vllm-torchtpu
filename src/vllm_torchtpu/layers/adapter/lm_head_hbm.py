# SPDX-License-Identifier: Apache-2.0
"""TPU_LM_HEAD_HBM: the LM head's logits from the HBM-streaming kernel.

vLLM computes logits as `lm_head.quant_method.apply(lm_head, x)`, which for an
unquantized vocab-parallel head is `F.linear(x, W)` with W [vocab/tp, hidden]
row-major. `HbmLmHeadMethod` replaces that call with a torch op over
`kernels.lm_head_hbm.lm_head_logits`, which runs the kernel at <= MAX_M logits rows
and XLA's dot otherwise (the row count is static per compiled program, so the choice
is made per program when JAX traces the op, never by a guard on the torch side).
Every logits path goes through `quant_method.apply` (`LogitsProcessor._apply_head`,
`get_top_tokens`, a draft's `compute_logits`); a logits processor whose `head_dtype`
differs from the hidden states' dtype bypasses it with its own F.linear, unchanged.

Modes (`install_lm_head_hbm`, run once after the drafter has bound its heads):
- "all": the target's head module gets the method. An MTP draft whose shared heads
  are that module (spec_decode/utils.py `maybe_share_lm_head`) follows.
- "drafter": the target's head is left alone. Every draft submodule slot that holds
  the target's head is rebound to a shallow copy of it -- same parameters, same
  weight tensor, its own `quant_method`. Draft tokens only change where a logit
  sits within a few bf16 ulp of the top; acceptance decides, the target distribution
  is untouched.
The kernel's logits are not bit-identical to XLA's (see kernels/lm_head_hbm.py), so
"all" needs an accuracy check before it ships.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MODES = ("", "drafter", "all")
_OP: Any = None


def get_lm_head_op() -> Any:
    """The torch op `x [m, k] bf16, w [n, k] bf16 -> [m, n]` over
    `lm_head_hbm.lm_head_logits`. Built at load: jax_op takes a Lock."""
    global _OP
    if _OP is None:
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels import lm_head_hbm

        op = pallas.jax_op("pallas::lm_head_hbm_logits", lm_head_hbm.lm_head_logits)
        op.register_fake(
            lambda x, w: torch.empty(
                x.shape[0], w.shape[0], dtype=x.dtype, device=x.device
            )
        )
        _OP = op
    return _OP


class HbmLmHeadMethod(UnquantizedEmbeddingMethod):
    """`UnquantizedEmbeddingMethod` whose logits come from the HBM-streaming kernel
    (decode row counts) or the same XLA dot as before (other row counts). The
    embedding lookup is inherited unchanged."""

    lm_head_hbm = True

    def __init__(self, op: Any) -> None:
        super().__init__()
        self._op = op

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = layer.weight
        if bias is not None or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return super().apply(layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        out = self._op(x2, w)
        return out.reshape(*x.shape[:-1], out.shape[-1])


def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    get_lm = getattr(model, "get_language_model", None)
    return get_lm() if callable(get_lm) else model


def _refusal(head: Any) -> str | None:
    if head is None:
        return "no lm_head"
    if type(head.quant_method) is not UnquantizedEmbeddingMethod:
        return f"quant method {type(head.quant_method).__name__}"
    w: Any = getattr(head, "weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return "weight is not a 2-D tensor"
    if w.dtype != torch.bfloat16:
        return f"weight dtype {w.dtype}"
    if getattr(head, "bias", None) is not None:
        return "the head has a bias"
    return None


def _rebind_draft_heads(
    draft_model: torch.nn.Module, head: torch.nn.Module, proxy: torch.nn.Module
) -> int:
    """Point every draft submodule slot holding `head` at `proxy`; returns the count."""
    n = 0
    for module in list(draft_model.modules()):
        for name, child in list(module._modules.items()):
            if child is head:
                setattr(module, name, proxy)
                n += 1
    return n


def install_lm_head_hbm(
    model: torch.nn.Module,
    draft_model: torch.nn.Module | None,
    mode: str,
    op: Any = None,
) -> int:
    """Install `HbmLmHeadMethod` per `mode` (see the module docstring). Returns the
    number of head slots that now run the kernel path (0: nothing installed; the
    reason is logged)."""
    if mode not in MODES:
        raise ValueError(f"TPU_LM_HEAD_HBM={mode!r}: expected one of {MODES}")
    if not mode:
        return 0
    head = getattr(_language_model(model), "lm_head", None)
    why = _refusal(head)
    if why is not None:
        logger.info("TPU_LM_HEAD_HBM=%s: not installed (%s).", mode, why)
        return 0
    assert head is not None  # _refusal(None) is a reason
    op = op if op is not None else get_lm_head_op()
    n, k = head.weight.shape
    if mode == "all":
        head.quant_method = HbmLmHeadMethod(op)
        shared = (
            0
            if draft_model is None
            else sum(
                1
                for m in draft_model.modules()
                for c in m._modules.values()
                if c is head
            )
        )
        logger.info(
            "Applied TPU patch: TPU_LM_HEAD_HBM=all - lm_head [%d, %d] logits from the "
            "HBM-streaming kernel at decode row counts (target sampler + %d draft "
            "slot(s)).",
            n,
            k,
            shared,
        )
        return 1 + shared
    if draft_model is None:
        logger.info("TPU_LM_HEAD_HBM=drafter: no draft model; not installed.")
        return 0
    proxy = copy.copy(head)  # shares _parameters / _buffers / _modules: same weight
    proxy.quant_method = HbmLmHeadMethod(op)
    count = _rebind_draft_heads(draft_model, head, proxy)
    if count == 0:
        logger.info(
            "TPU_LM_HEAD_HBM=drafter: the draft does not share the target's lm_head "
            "module; not installed."
        )
        return 0
    logger.info(
        "Applied TPU patch: TPU_LM_HEAD_HBM=drafter - %d draft head slot(s) on a copy "
        "of lm_head [%d, %d] (same weight) with logits from the HBM-streaming kernel; "
        "the target sampler keeps XLA's dot.",
        count,
        n,
        k,
    )
    return count
