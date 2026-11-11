"""Oracle decoding: a request's completion is the tokens it supplies.

A request with ``SamplingParams.forced_token_ids`` still runs every forward pass
it would normally run (prefill, DSpark draft and target verify), but the tokens
it emits are the forced ones. A benchmark replaying a recorded conversation then
produces the exact next-turn history, so prefix-cache hits and output lengths
are properties of the dataset and runs are comparable. Benchmark only: enabled
with SGLANG_ENABLE_ORACLE_DECODE, and the emitted KV belongs to whatever tokens
the forward pass consumed, so output quality is meaningless.

The forced tokens live in a device pool indexed by request slot
(``req.req_pool_idx``), written once when a request first lands in a slot. Each
forward step then reads its window with a gather; no per-step host copies.

Positions follow the sampler: a token produced after consuming position ``p``
lands at ``p + 1``, and forced token ``i`` sits at position
``len(origin_input_ids) + i``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def oracle_enabled() -> bool:
    return envs.SGLANG_ENABLE_ORACLE_DECODE.get()


class OracleTokenPool:
    def __init__(self, num_slots: int, max_tokens: int, device):
        self.max_tokens = int(max_tokens)
        self.tokens = torch.zeros((num_slots, self.max_tokens), dtype=torch.int64, device=device)
        self.start = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.valid = torch.zeros(num_slots, dtype=torch.int64, device=device)
        # slot -> (rid, prompt length) of the forced request written there
        self._owner: dict[int, tuple] = {}

    def sync(self, reqs: Iterable["Req"]) -> bool:
        """Write newly placed forced requests into their slots and clear slots
        reused by unforced ones. True if any request in ``reqs`` is forced."""
        active = False
        for req in reqs:
            slot = getattr(req, "req_pool_idx", None)
            if slot is None:
                continue
            forced = req.sampling_params.forced_token_ids
            if forced:
                active = True
                key = (req.rid, len(req.origin_input_ids))
                if self._owner.get(slot) != key:
                    if len(forced) > self.max_tokens:
                        raise ValueError(
                            f"forced_token_ids has {len(forced)} tokens; "
                            f"SGLANG_ORACLE_MAX_TOKENS is {self.max_tokens}"
                        )
                    row = torch.tensor(forced, dtype=torch.int64)
                    self.tokens[slot, : len(forced)].copy_(row, non_blocking=False)
                    self.start[slot] = len(req.origin_input_ids)
                    self.valid[slot] = len(forced)
                    self._owner[slot] = key
            elif slot in self._owner:
                self.valid[slot] = 0
                del self._owner[slot]
        return active

    def apply(
        self,
        next_token_ids: torch.Tensor,
        positions: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Replace sampled tokens with forced ones where a forced token exists
        at ``positions + 1``; other rows pass through."""
        n = next_token_ids.shape[0]
        slots = req_pool_indices[:n].to(torch.int64)
        idx = positions[:n].to(torch.int64) + 1 - self.start[slots]
        in_range = (idx >= 0) & (idx < self.valid[slots])
        forced = self.tokens[slots, idx.clamp(0, self.max_tokens - 1)]
        return torch.where(in_range, forced.to(next_token_ids.dtype), next_token_ids)

    def window(
        self,
        req_pool_indices: torch.Tensor,
        first_positions: torch.Tensor,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forced tokens at ``first_positions + j`` for j < width, and whether
        each exists: ``(tokens[bs, width], in_range[bs, width])``."""
        slots = req_pool_indices.to(torch.int64)
        idx = (first_positions.to(torch.int64) - self.start[slots]).unsqueeze(1) + torch.arange(
            width, device=slots.device
        )
        in_range = (idx >= 0) & (idx < self.valid[slots].unsqueeze(1))
        tokens = self.tokens[slots.unsqueeze(1), idx.clamp(0, self.max_tokens - 1)]
        return tokens, in_range


_POOL: Optional[OracleTokenPool] = None


def get_oracle_pool(req_to_token_pool, device) -> OracleTokenPool:
    """The process's pool, with one row per request slot of
    ``req_to_token_pool`` (slot 0 is the padding slot)."""
    global _POOL
    if _POOL is None:
        table = getattr(req_to_token_pool, "req_to_token", None)
        num_slots = (
            table.shape[0]
            if table is not None
            else req_to_token_pool.size + 1 + getattr(req_to_token_pool, "pre_alloc_size", 0)
        )
        _POOL = OracleTokenPool(num_slots, envs.SGLANG_ORACLE_MAX_TOKENS.get(), device)
    return _POOL


def oracle_commit(
    *,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    window: torch.Tensor,
    in_range: torch.Tensor,
    max_correct_len: torch.Tensor,
    fixed_commit_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DSpark accept for Oracle rows. ``window[:, j]`` is the forced token for
    out position j (the verify drafts were replaced by ``window[:, :gamma]``).
    Commit ``fixed_commit_len`` tokens, or the target's own greedy acceptance of
    the forced drafts when it is 0, bounded by the verify cap and the forced
    tokens left; at least one. The bonus is the forced token at that length.
    Rows with no forced token left keep the served result."""
    avail = in_range.to(torch.int64).cumprod(dim=1).sum(dim=1)
    active = avail > 0
    c = correct_len.to(torch.int64)
    if fixed_commit_len > 0:
        c = torch.full_like(c, fixed_commit_len - 1)
    c = torch.minimum(c, max_correct_len.to(torch.int64))
    c = torch.minimum(c, avail - 1).clamp_min(0)
    forced_bonus = window.gather(1, c.unsqueeze(1)).squeeze(1)
    return (
        torch.where(active, c.to(correct_len.dtype), correct_len),
        torch.where(active, forced_bonus.to(bonus.dtype), bonus),
        torch.where(active, torch.zeros_like(cap_trim_lens), cap_trim_lens),
    )
