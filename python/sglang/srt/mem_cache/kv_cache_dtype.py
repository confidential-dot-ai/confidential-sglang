import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.platform_kv_cache_dtype import (
    registered_kv_cache_storage_dtype,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

_is_hip = is_hip()

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}

# Quantization recipes of the NPU DSA quantized indexer cache. Unlike the
# quant_lightning_indexer (v2) FP8/MXFP8/FP4 quant modes, the INT8 recipe
# runs the v1 npu_quant_lightning_indexer op, where the quantization is
# implied by the INT8 key dtype (its quant_mode args stay 0).
QUANT_MODE_TOKEN_INT8 = 0

_INDEXER_KV_CACHE_DTYPE_TO_QUANT_MODE = {
    "int8": QUANT_MODE_TOKEN_INT8,
}


def dflash_draft_kv_cache_dtype(
    *,
    kv_cache_dtype: torch.dtype,
    model_dtype: torch.dtype,
    speculative_draft_attention_backend: Optional[str],
    kv_cache_is_fp4: Optional[bool] = None,
) -> torch.dtype:
    """The KV dtype a DFLASH-family draft pool actually uses.

    Single source of truth for the draft-dtype override, shared between
    configure_kv_cache_dtype (pool allocation on the draft worker) and
    DefaultPoolConfigurator (target-side sizing that budgets for the draft
    pool): fa4 draft attention needs K.dtype == Q.dtype, and FP4 target KV was
    never validated for any draft backend, so both force the draft pool back
    to the compute dtype.
    """
    if kv_cache_is_fp4 is None:
        kv_cache_is_fp4 = (
            hasattr(torch, "float4_e2m1fn_x2")
            and kv_cache_dtype == torch.float4_e2m1fn_x2
        )
    needs_draft_dtype_override = (
        speculative_draft_attention_backend == "fa4" or kv_cache_is_fp4
    )
    if needs_draft_dtype_override and kv_cache_dtype != model_dtype:
        return model_dtype
    return kv_cache_dtype


def configure_kv_cache_dtype(
    *,
    server_args_kv_cache_dtype: str,
    quant_config: Optional["QuantizationConfig"],
    model_dtype: torch.dtype,
    is_draft_worker: bool,
    is_dflash_family: bool,
    speculative_draft_attention_backend: str,
    speculative_draft_kv_cache_dtype: Optional[str] = None,
) -> tuple[str, torch.dtype]:
    """(label, torch dtype) of a runner's KV cache; a pool in the compute
    dtype is labelled "auto", which backends gate FP8 casts and descales on."""
    if is_draft_worker and speculative_draft_kv_cache_dtype is not None:
        server_args_kv_cache_dtype = speculative_draft_kv_cache_dtype
    kv_cache_dtype_str = server_args_kv_cache_dtype
    if server_args_kv_cache_dtype == "auto":
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            kv_cache_dtype = fp8_dtype if _is_hip else torch.float8_e4m3fn
            kv_cache_dtype_str = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
        else:
            kv_cache_dtype = model_dtype
    elif server_args_kv_cache_dtype == "fp8_e5m2":
        if current_platform.is_cpu():
            raise ValueError("--kv-cache-dtype fp8_e5m2 is not supported on CPU.")
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e5m2
    elif server_args_kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype in ("bf16", "bfloat16"):
        kv_cache_dtype = torch.bfloat16
    elif server_args_kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
        if hasattr(torch, "float4_e2m1fn_x2"):
            kv_cache_dtype = torch.float4_e2m1fn_x2
            logger.warning(
                "%s KV Cache might lead to an accuracy drop!",
                server_args_kv_cache_dtype.upper(),
            )
        else:
            raise ValueError(
                f"--kv-cache-dtype={server_args_kv_cache_dtype} requires "
                "torch.float4_e2m1fn_x2 support. Please use PyTorch 2.8.0+ "
                "with CUDA 12.8+."
            )
    elif (
        storage_dtype := registered_kv_cache_storage_dtype(server_args_kv_cache_dtype)
    ) is not None:
        kv_cache_dtype = storage_dtype
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {server_args_kv_cache_dtype}.")

    # DFLASH: fa4 draft attention can't read the target's fp8 KV (needs K.dtype == Q.dtype),
    # so give the fa4 draft its own compute-dtype KV. fp8-capable backends keep the target
    # dtype. FP4 (fp4_e2m1) draft KV was never validated for ANY draft backend -- FlashInfer's
    # own paged-prefill wrapper enforces k.dtype == kv_data_type at plan/run time, and the
    # draft model's own K/V arrive in model_dtype (bf16, not FP4), so every draft backend
    # needs its own compute-dtype KV when the target's KV cache is FP4, not just fa4.
    # Policy lives in dflash_draft_kv_cache_dtype (shared with pool sizing).
    if is_draft_worker and is_dflash_family:
        draft_kv_dtype = dflash_draft_kv_cache_dtype(
            kv_cache_dtype=kv_cache_dtype,
            model_dtype=model_dtype,
            speculative_draft_attention_backend=speculative_draft_attention_backend,
        )
        if draft_kv_dtype != kv_cache_dtype:
            logger.info(
                "DFLASH-family %s draft: overriding KV cache dtype %s -> %s "
                "(draft attention backend cannot read the target's quantized KV).",
                speculative_draft_attention_backend,
                kv_cache_dtype,
                draft_kv_dtype,
            )
            kv_cache_dtype = draft_kv_dtype
            # "auto" is the tag for an unquantized pool; backends gate descale on it.
            kv_cache_dtype_str = "auto"

    return kv_cache_dtype_str, kv_cache_dtype


def resolve_indexer_quant_mode(
    indexer_kv_cache_dtype: Optional[str],
    main_kv_cache_dtype: torch.dtype,
) -> Optional[int]:
    """Map --indexer-kv-cache-dtype to a quantized-indexer recipe for the
    NPU DSA indexer cache.

    None leaves the quantized indexer at its default: an FP8 main KV cache
    keeps the existing FP8 indexer (the DSA FP8 packed store), any other
    main dtype leaves the quantized indexer disabled (the DSA indexer then
    runs the legacy bf16 npu_lightning_indexer path). The INT8 recipe only
    exists on top of a non-FP8 main KV cache; the FP8 main cache already
    stores the indexer cache quantized.
    """
    if indexer_kv_cache_dtype is None:
        return None
    quant_mode = _INDEXER_KV_CACHE_DTYPE_TO_QUANT_MODE.get(indexer_kv_cache_dtype)
    if quant_mode is None:
        raise ValueError(
            f"Unsupported indexer_kv_cache_dtype: {indexer_kv_cache_dtype}."
        )
    if main_kv_cache_dtype == torch.float8_e4m3fn:
        raise ValueError(
            f"--indexer-kv-cache-dtype={indexer_kv_cache_dtype} requires a "
            "non-FP8 main KV cache (--kv-cache-dtype auto or bf16), "
            f"got {main_kv_cache_dtype}."
        )
    return quant_mode
