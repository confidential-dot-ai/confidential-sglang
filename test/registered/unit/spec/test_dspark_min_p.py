"""Sampling parameters under DSpark: min_p filters the verify distribution the way
the sampler does, every temperature-only fast path steps aside, and the
frequency/presence/repetition penalties reach the verify logits."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    build_speculative_verify_target_probs,
    min_p_renorm_prob,
)
from sglang.srt.speculative.dspark_components import dspark_verify


def _info(bs, vocab, *, temperature=0.7, top_k=None, top_p=1.0, min_p=0.0):
    min_ps = torch.as_tensor(min_p, dtype=torch.float32).expand(bs).clone()
    return SimpleNamespace(
        temperatures=torch.full((bs, 1), temperature),
        top_ks=torch.full((bs,), vocab if top_k is None else top_k, dtype=torch.int32),
        top_ps=torch.full((bs,), top_p),
        min_ps=min_ps,
        need_top_k_sampling=top_k is not None,
        need_top_p_sampling=top_p < 1.0,
        need_min_p_sampling=bool((min_ps > 0).any()),
        is_all_greedy=False,
        is_any_greedy=False,
        sampling_seed=None,
    )


class TestMinPRenorm(unittest.TestCase):
    def test_matches_the_sampler_filter(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(6, 50) * 2, dim=-1)
        min_ps = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.3, 1.0])
        ours = min_p_renorm_prob(probs, min_ps)
        _, sorted_weights, sorted_ids, _ = (None, *top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            torch.full((6,), 50, dtype=torch.int32),
            torch.ones(6),
            min_ps,
            True,
            None,
            torch.zeros(6, dtype=torch.int64),
            return_filtered_probs=True,
        )[1:])
        ref = torch.zeros_like(probs).scatter_(1, sorted_ids, sorted_weights)
        ref = ref / ref.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(ours, ref)

    def test_keeps_the_top_token_and_sums_to_one(self):
        probs = torch.softmax(torch.randn(4, 30), dim=-1)
        out = min_p_renorm_prob(probs, torch.full((4,), 1.0))
        self.assertTrue(torch.equal(out.argmax(-1), probs.argmax(-1)))
        torch.testing.assert_close(out.sum(-1), torch.ones(4))
        self.assertTrue(((out > 0).sum(-1) >= 1).all())

    def test_zero_min_p_is_identity(self):
        probs = torch.softmax(torch.randn(3, 20), dim=-1)
        torch.testing.assert_close(min_p_renorm_prob(probs, torch.zeros(3)), probs)


class TestVerifyTargetProbs(unittest.TestCase):
    def _probs(self, info, logits, rows):
        return build_speculative_verify_target_probs(
            next_token_logits=logits,
            sampling_info=info,
            draft_token_num=rows,
            bs=info.temperatures.shape[0],
        )

    def test_min_p_filters_every_verify_row(self):
        torch.manual_seed(1)
        bs, rows, vocab = 2, 4, 64
        logits = torch.randn(bs * rows, vocab) * 3
        info = _info(bs, vocab, min_p=[0.0, 0.2])
        out = self._probs(info, logits, rows).view(bs * rows, vocab)
        plain = torch.softmax(logits / 0.7, dim=-1)
        torch.testing.assert_close(out[:rows], plain[:rows])
        expected = min_p_renorm_prob(plain[rows:], torch.full((rows,), 0.2))
        torch.testing.assert_close(out[rows:], expected)
        self.assertTrue((out[rows:][plain[rows:] < 0.2 * plain[rows:].amax(-1, keepdim=True)] == 0).all())

    def test_min_p_composes_with_top_k_and_top_p(self):
        torch.manual_seed(2)
        bs, rows, vocab = 3, 2, 40
        logits = torch.randn(bs * rows, vocab) * 2
        info = _info(bs, vocab, top_k=10, top_p=0.9, min_p=0.1)
        out = self._probs(info, logits, rows).view(bs * rows, vocab)
        base = _info(bs, vocab, top_k=10, top_p=0.9)
        filtered = self._probs(base, logits, rows).view(bs * rows, vocab)
        torch.testing.assert_close(out, min_p_renorm_prob(filtered, torch.full((bs * rows,), 0.1)))
        self.assertTrue(((out > 0).sum(-1) <= 10).all())


class TestFastPathsStepAside(unittest.TestCase):
    def test_sharded_and_blockwise_accepts_exclude_min_p(self):
        info = _info(2, 16, min_p=0.1)
        block = SimpleNamespace(vocab_shard=object(), greedy_draft=False)
        self.assertFalse(dspark_verify._accepts_on_shards(block, info, False))
        self.assertFalse(
            dspark_verify._blockwise_rejection_applies(
                sampling_info=info, rejection_noise=object(), block_verification=False
            )
        )
        plain = _info(2, 16)
        self.assertTrue(dspark_verify._accepts_on_shards(block, plain, False))
        self.assertTrue(
            dspark_verify._blockwise_rejection_applies(
                sampling_info=plain, rejection_noise=object(), block_verification=False
            )
        )

    def test_split_accept_kernel_is_not_used_for_min_p(self):
        # The split kernel folds a temperature-only softmax; min-p must reach
        # build_speculative_verify_target_probs instead.
        seen = []
        orig = dspark_accept.build_speculative_verify_target_probs

        def spy(**kw):
            seen.append(kw["sampling_info"])
            raise RuntimeError("stop")

        dspark_accept.build_speculative_verify_target_probs = spy
        try:
            info = _info(1, 16, min_p=0.1)
            with self.assertRaisesRegex(RuntimeError, "stop"):
                dspark_accept._accept_sampling_core(
                    candidates=torch.zeros(1, 4, dtype=torch.int64),
                    target_logits=torch.randn(4, 16),
                    draft_probs=torch.full((1, 3, 16), 1 / 16),
                    sampling_info=info,
                    draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
                    gamma=3,
                    verify_num_draft_tokens=4,
                    cutoff_verify_lens=None,
                    block_verification=False,
                )
        finally:
            dspark_accept.build_speculative_verify_target_probs = orig
        self.assertEqual(seen, [info])


@unittest.skipUnless(torch.cuda.is_available(), "the chain accept kernel is Triton")
class TestSampledAcceptHonorsMinPOnGpu(unittest.TestCase):
    def test_first_committed_token_follows_the_filtered_target(self):
        torch.manual_seed(3)
        dev = "cuda"
        vocab, gamma, n = 12, 3, 20000
        rows = gamma + 1
        row_logits = torch.randn(rows, vocab, device=dev) * 1.5
        info = _info(n, vocab, temperature=1.0, min_p=0.15)
        info = SimpleNamespace(**{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in vars(info).items()})
        # A uniform draft proposes every token, including ones min-p forbids.
        draft = torch.randint(0, vocab, (n, gamma), device=dev)
        candidates = torch.cat([torch.zeros(n, 1, dtype=torch.int64, device=dev), draft], dim=1)
        correct_len, bonus, _ = dspark_accept.AcceptSampling.execute(
            candidates=candidates,
            target_logits=row_logits.repeat(n, 1),
            draft_probs=torch.full((n, gamma, vocab), 1.0 / vocab, device=dev),
            sampling_info=info,
            draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
            gamma=gamma,
            verify_num_draft_tokens=rows,
        )
        first = torch.where(correct_len > 0, draft[:, 0], bonus.view(-1))
        target = min_p_renorm_prob(torch.softmax(row_logits[:1], -1), torch.tensor([0.15], device=dev))[0]
        counts = torch.bincount(first, minlength=vocab).float() / n
        self.assertEqual(int(counts[target == 0].sum().item() * n), 0)
        torch.testing.assert_close(counts, target, atol=0.015, rtol=0)


class _PenaltyInfo:
    """What verify receives under overlap scheduling: accumulated penalties, no
    live penalizer (SamplingBatchInfo.copy_for_forward)."""

    def __init__(self, bs, *, additive=None, scaling=None):
        self.acc_additive_penalties = additive
        self.acc_scaling_penalties = scaling
        self.penalizer_orchestrator = None
        self.grammar_mask = None
        self.logit_bias = None
        self.has_custom_logit_processor = False
        self._bs = bs

    def __len__(self):
        return self._bs


class TestPenaltiesReachVerify(unittest.TestCase):
    # Verify read a field that no longer exists, so penalties shaped only the
    # prefill-sampled first token and greedy steps took the penalty-free graph accept.
    def test_accumulated_penalties_are_applied_like_the_sampler(self):
        torch.manual_seed(4)
        bs, rows, vocab = 2, 3, 16
        logits = torch.randn(bs * rows, vocab) * 3
        additive = torch.zeros(bs, vocab)
        additive[0, 3] = -2.0
        scaling = torch.ones(bs, vocab)
        scaling[1, 5] = 1.5
        out = logits.clone()
        apply_dflash_verify_logits_adjustments(
            next_token_logits=out,
            sampling_info=_PenaltyInfo(bs, additive=additive, scaling=scaling),
            draft_token_num=rows,
        )
        # The sampler's order: add, then scale (_apply_pre_grammar_logits_transforms).
        ref = logits.view(bs, rows, vocab) + additive[:, None]
        ref = torch.where(ref < 0, ref * scaling[:, None], ref / scaling[:, None])
        torch.testing.assert_close(out.view(bs, rows, vocab), ref)

    def test_penalties_keep_verify_off_the_noop_paths(self):
        noop = dspark_verify.verify_logits_adjustments_are_noop
        self.assertFalse(noop(_PenaltyInfo(1, additive=torch.zeros(1, 8))))
        self.assertFalse(noop(_PenaltyInfo(1, scaling=torch.ones(1, 8))))
        self.assertTrue(noop(_PenaltyInfo(1)))


if __name__ == "__main__":
    unittest.main()
