from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import INDEX_K_SLOT_BYTES
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    _ceil_ue8m0_exp,
    _has_native_e2m1x2,
    _pack_e2m1x2,
    _select_group_value,
    quantize_fp4_indexer_tensor,
)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


@triton.jit
def _quantize_store_fp4_index_k_cache_kernel(
    x,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
    USE_NATIVE_E2M1X2: tl.constexpr,
):
    """Quantize one index row and scatter it directly into the paged cache."""
    token_id = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_N // 2)
    offs0 = pair_offsets * 2
    offs1 = offs0 + 1
    raw0 = tl.load(x + token_id * BLOCK_N + offs0).to(tl.float32)
    raw1 = tl.load(x + token_id * BLOCK_N + offs1).to(tl.float32)

    # Each adjacent FP4 pair lies wholly inside one 32-value scale group. Reduce
    # the pair maxima to derive the same four scales, then reuse raw0/raw1 for
    # packing instead of loading the entire row a second time.
    pair_group = pair_offsets // (GROUP_N // 2)
    pair_abs_max = tl.maximum(tl.abs(raw0), tl.abs(raw1))
    amax0 = tl.max(tl.where(pair_group == 0, pair_abs_max, 0.0), axis=0)
    amax1 = tl.max(tl.where(pair_group == 1, pair_abs_max, 0.0), axis=0)
    amax2 = tl.max(tl.where(pair_group == 2, pair_abs_max, 0.0), axis=0)
    amax3 = tl.max(tl.where(pair_group == 3, pair_abs_max, 0.0), axis=0)

    exp0 = _ceil_ue8m0_exp(tl.maximum(amax0 / 6.0, 1.0e-4))
    exp1 = _ceil_ue8m0_exp(tl.maximum(amax1 / 6.0, 1.0e-4))
    exp2 = _ceil_ue8m0_exp(tl.maximum(amax2 / 6.0, 1.0e-4))
    exp3 = _ceil_ue8m0_exp(tl.maximum(amax3 / 6.0, 1.0e-4))

    scale_exp = _select_group_value(pair_group, exp0, exp1, exp2, exp3)
    scale = (scale_exp << 23).to(tl.float32, bitcast=True)
    v0 = raw0 / scale
    v1 = raw1 / scale
    packed = _pack_e2m1x2(v0, v1, RNE, USE_NATIVE_E2M1X2)

    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size
    page_base = cache + page * cache_stride
    tl.store(page_base + page_offset * (BLOCK_N // 2) + pair_offsets, packed)

    packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (packed_sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        page_base
        + page_size * (BLOCK_N // 2)
        + page_offset * 4
        + sf_offsets,
        sf_bytes,
    )


def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = True,
) -> None:
    assert input.shape[-1] == 128
    input = input.contiguous()
    n_tokens = input.numel() // input.shape[-1]
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES

    if n_tokens == 0:
        return
    _quantize_store_fp4_index_k_cache_kernel[(n_tokens,)](
        input,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK_N=128,
        GROUP_N=32,
        RNE=rne,
        USE_NATIVE_E2M1X2=_has_native_e2m1x2(input.device),
    )


def _store_fp4_index_k_cache_unfused(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = True,
) -> None:
    """Measured control for the fused writer; not used by serving."""
    assert input.shape[-1] == 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    assert cache.shape[1] == page_size * (64 + 4)
    if n_tokens == 0:
        return
    _store_fp4_index_k_cache_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK=64,
    )


@triton.jit
def _gather_fp4_index_k_kernel(
    cache,
    cache_stride,
    slots,
    payload,
    scales,
    n,
    page_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # int32 view of a page: page_size * 16 payload words, then page_size scale words.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n
    slot = tl.load(slots + rows, valid, 0).to(tl.int64)
    page = slot // page_size
    offset = slot - page * page_size
    page_base = cache + page * cache_stride
    words = tl.arange(0, 16)
    payload_words = tl.load(
        page_base[:, None] + offset[:, None] * 16 + words[None, :], valid[:, None]
    )
    tl.store(
        payload + rows[:, None].to(tl.int64) * 16 + words[None, :],
        payload_words,
        valid[:, None],
    )
    tl.store(scales + rows, tl.load(page_base + page_size * 16 + offset, valid), valid)


def gather_fp4_index_k(
    cache: torch.Tensor, slots: torch.Tensor, *, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed fp4 rows at ``slots`` of the index-K ``cache`` (``[pages, page_size *
    68]`` uint8, ``store_fp4_index_k_cache``'s layout): (payload int8 ``[n, 64]``,
    ue8m0 scales packed int32 ``[n]``), the layout ``quantize_fp4_indexer_tensor``
    returns."""
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES and cache.stride(1) == 1
    # The kernel addresses slots as `slots + rows`, so a strided view would be read
    # as if its elements were packed.
    assert slots.is_contiguous()
    words = cache.view(torch.int32)
    n = slots.shape[0]
    payload = torch.empty((n, 16), device=cache.device, dtype=torch.int32)
    scales = torch.empty((n,), device=cache.device, dtype=torch.int32)
    if n > 0:
        block_rows = 64
        _gather_fp4_index_k_kernel[(triton.cdiv(n, block_rows),)](
            words,
            words.stride(0),
            slots,
            payload,
            scales,
            n,
            page_size,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
    return payload.view(torch.int8), scales
