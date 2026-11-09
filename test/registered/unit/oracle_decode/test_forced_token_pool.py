from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.oracle_decode.forced_token_pool import OracleTokenPool, oracle_commit
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def req(rid, slot, prompt_len, forced):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=slot,
        origin_input_ids=[0] * prompt_len,
        sampling_params=SimpleNamespace(forced_token_ids=forced),
    )


def test_sync_writes_forced_rows_and_clears_reused_slots():
    pool = OracleTokenPool(num_slots=4, max_tokens=8, device="cpu")
    a = req("a", 1, prompt_len=5, forced=[11, 12, 13])
    assert pool.sync([a, req("b", 2, 3, None)])
    assert pool.tokens[1, :3].tolist() == [11, 12, 13]
    assert pool.start[1] == 5 and pool.valid[1] == 3
    # Slot 1 is reused by an unforced request: its row stops applying.
    assert not pool.sync([req("c", 1, 4, None)])
    assert pool.valid[1] == 0
    # The forced request comes back to the same slot (e.g. after a retraction).
    assert pool.sync([a])
    assert pool.valid[1] == 3


def test_sync_rejects_forced_runs_longer_than_the_pool():
    pool = OracleTokenPool(num_slots=2, max_tokens=2, device="cpu")
    with pytest.raises(ValueError, match="SGLANG_ORACLE_MAX_TOKENS"):
        pool.sync([req("a", 1, 3, [1, 2, 3])])


def test_apply_forces_by_position_and_passes_other_rows_through():
    pool = OracleTokenPool(num_slots=4, max_tokens=8, device="cpu")
    pool.sync([req("a", 1, prompt_len=5, forced=[11, 12, 13]), req("b", 2, 3, None)])
    sampled = torch.tensor([99, 98, 97])
    # Row 0: prefill's last prompt position (4) -> forced[0]. Row 1: unforced.
    # Row 2: position 7 -> index 3, past the 3 forced tokens -> sampled.
    out = pool.apply(sampled, torch.tensor([4, 2, 7]), torch.tensor([1, 2, 1]))
    assert out.tolist() == [11, 98, 97]
    out = pool.apply(sampled[:1], torch.tensor([6]), torch.tensor([1]))
    assert out.tolist() == [13]


def test_window_reads_forced_tokens_from_a_position():
    pool = OracleTokenPool(num_slots=4, max_tokens=8, device="cpu")
    pool.sync([req("a", 1, prompt_len=5, forced=[11, 12, 13, 14])])
    tokens, in_range = pool.window(torch.tensor([1]), torch.tensor([6]), width=4)
    assert tokens[0, :3].tolist() == [12, 13, 14]
    assert in_range.tolist() == [[True, True, True, False]]


def commit(correct, window, in_range, fixed, max_correct=4):
    bs = len(correct)
    return oracle_commit(
        correct_len=torch.tensor(correct, dtype=torch.int32),
        bonus=torch.full((bs,), -1, dtype=torch.int64),
        cap_trim_lens=torch.ones(bs, dtype=torch.int32),
        window=torch.tensor(window),
        in_range=torch.tensor(in_range),
        max_correct_len=torch.full((bs,), max_correct),
        fixed_commit_len=fixed,
    )


def test_fixed_commit_length_emits_the_forced_tokens():
    w = [[1, 2, 3, 4, 5]]
    c, bonus, trim = commit([0], w, [[True] * 5], fixed=3)
    # Commit 3 = 2 accepted drafts + the bonus, which is the third forced token.
    assert c.tolist() == [2] and bonus.tolist() == [3] and trim.tolist() == [0]


def test_target_acceptance_is_used_when_no_fixed_length():
    c, bonus, _ = commit([1], [[1, 2, 3, 4, 5]], [[True] * 5], fixed=0)
    assert c.tolist() == [1] and bonus.tolist() == [2]


def test_commit_is_bounded_by_tokens_left_and_the_verify_cap():
    # Two forced tokens left: commit at most 2 even with a fixed length of 4.
    c, bonus, _ = commit([0], [[7, 8, 0, 0, 0]], [[True, True, False, False, False]], fixed=4)
    assert c.tolist() == [1] and bonus.tolist() == [8]
    c, bonus, _ = commit([0], [[1, 2, 3, 4, 5]], [[True] * 5], fixed=5, max_correct=1)
    assert c.tolist() == [1] and bonus.tolist() == [2]


def test_exhausted_rows_keep_the_served_accept():
    c, bonus, trim = commit([2], [[0] * 5], [[False] * 5], fixed=3)
    assert c.tolist() == [2] and bonus.tolist() == [-1] and trim.tolist() == [1]


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_forced_text_is_tokenized_by_the_server_and_sets_the_length(monkeypatch):
    from sglang.srt.sampling.sampling_params import SamplingParams

    monkeypatch.delenv("SGLANG_ENABLE_ORACLE_DECODE", raising=False)
    with pytest.raises(ValueError, match="SGLANG_ENABLE_ORACLE_DECODE"):
        SamplingParams(forced_text="ab").normalize(CharTokenizer())
    monkeypatch.setenv("SGLANG_ENABLE_ORACLE_DECODE", "1")
    sp = SamplingParams(forced_text="abc", max_new_tokens=1)
    sp.normalize(CharTokenizer())
    sp.verify(200)
    assert sp.forced_token_ids == [97, 98, 99]
    assert sp.max_new_tokens == 3
    with pytest.raises(ValueError, match="outside"):
        sp.verify(98)
    with pytest.raises(ValueError, match="non-empty"):
        SamplingParams(forced_text="").normalize(CharTokenizer())


def test_token_ids_cannot_be_sent_directly(monkeypatch):
    from sglang.srt.sampling.sampling_params import SamplingParams

    monkeypatch.setenv("SGLANG_ENABLE_ORACLE_DECODE", "1")
    with pytest.raises(ValueError, match="internal"):
        SamplingParams(forced_token_ids=[1, 2]).normalize(CharTokenizer())
    with pytest.raises(ValueError, match="internal"):
        SamplingParams(forced_token_ids=[1, 2]).verify(100)
