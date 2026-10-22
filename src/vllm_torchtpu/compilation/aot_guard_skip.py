# SPDX-License-Identifier: Apache-2.0
"""Skip the per-call guard check of guard-free AOT-compiled forwards
(TPU_AOT_SKIP_GUARD_CHECK).

vLLM compiles model forwards with ``guard_filter_fn=skip_all_guards_unsafe``
unless ``dynamic_shapes_config.evaluate_guards`` is set (or the mode is stock
torch.compile), and its loader calls ``disable_guard_check()`` on an AOT
artifact read from disk under that condition. A freshly compiled artifact
keeps ``_guard_check_enabled`` and so runs ``prepare_f_locals`` plus an empty
guard-manager check on every call (~25 us per forward on the serving thread).

``install()`` wraps ``TorchCompileWithNoGuardsWrapper.aot_compile`` so every
fresh AOT compile -- the target model at warmup and the drafter forwards that
compile lazily on their first propose -- gets the loader's treatment.
``disable(fn)`` does the same for one already-compiled function. ``FNS``
records every function switched off (for the in-run A/B).
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

FNS: list[Any] = []
_installed = False


def guards_dropped(compilation_config: Any) -> bool:
    """Whether vLLM compiled with every guard filtered out."""
    from vllm.config import CompilationMode

    return not compilation_config.dynamic_shapes_config.evaluate_guards and (
        compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE
    )


def disable(fn: Any) -> bool:
    if fn is None or not hasattr(fn, "disable_guard_check"):
        return False
    fn.disable_guard_check()
    if all(f is not fn for f in FNS):
        FNS.append(fn)
    return True


def install() -> bool:
    """Idempotent. True if the wrapper is patched."""
    global _installed
    if _installed:
        return True
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W

    original = W.aot_compile

    def aot_compile(self, *args: Any, **kwargs: Any) -> Any:
        fn = original(self, *args, **kwargs)
        if guards_dropped(self.vllm_config.compilation_config) and disable(fn):
            logger.info(
                "TPU_AOT_SKIP_GUARD_CHECK: guard check off for %s",
                type(self).__name__,
            )
        return fn

    aot_compile._h5b_original = original  # type: ignore[attr-defined]
    W.aot_compile = aot_compile
    _installed = True
    return True
