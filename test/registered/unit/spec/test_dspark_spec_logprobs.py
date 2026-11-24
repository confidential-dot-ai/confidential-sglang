"""Logprobs under DSpark (DFLASH family) speculative decoding, PD and not.

The decode verify step returns the target's logprobs for every committed token
(compute_spec_logprobs on the (bs, verify width) chain layout), the scheduler
appends them per accepted token, a PD decode server prepends the first token's
logprob from the prefill metadata buffer, and the output streamer cuts them at
the stop like the output ids. These tests run each stage on CPU tensors.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_greedy
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    build_out_tokens,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.io_struct import RequestOutputStats
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 50
WIDTH = 4  # verify width: the current token + 3 drafts
GAMMA = WIDTH - 1


def _batch(temperatures, top_logprobs_nums, token_ids_logprobs=None, top_ps=None):
    bs = len(temperatures)
    return SimpleNamespace(
        seq_lens=torch.zeros(bs, dtype=torch.int64),
        sampling_info=SimpleNamespace(
            is_all_greedy=all(t == 0 for t in temperatures),
            # Greedy rows carry temperature 1 (SamplingBatchInfo).
            temperatures=torch.tensor(
                [[t if t > 0 else 1.0] for t in temperatures], dtype=torch.float32
            ),
            top_ps=torch.tensor(top_ps or [1.0] * bs),
            need_top_p_sampling=top_ps is not None,
        ),
        top_logprobs_nums=top_logprobs_nums,
        token_ids_logprobs=token_ids_logprobs or [None] * bs,
    )


def _non_spec_logprobs(logits, temperature):
    """What the non-speculative Sampler returns: log of the temperature-scaled
    softmax, before top-p/top-k/min-p (sampler.py: torch.log(probs))."""
    if temperature == 0:
        return torch.log_softmax(logits, dim=-1)
    return torch.log(torch.softmax(logits / temperature, dim=-1))


def _greedy_step(logits, bs, matched):
    """A real greedy accept: draft j matches the target argmax for j < matched[b]."""
    predict = logits.argmax(-1).view(bs, WIDTH)
    drafts = predict[:, :GAMMA].clone()
    for b, m in enumerate(matched):
        if m < GAMMA:
            drafts[b, m] = (drafts[b, m] + 1) % VOCAB
    candidates = torch.cat([torch.zeros(bs, 1, dtype=torch.int64), drafts], dim=1)
    correct_len, bonus, _ = accept_greedy(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=WIDTH,
    )
    out_tokens = build_out_tokens(
        draft_tokens=drafts,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=WIDTH,
        gamma=GAMMA,
    )
    return out_tokens, (correct_len + 1).tolist()


class TestComputeSpecLogprobs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_greedy_accepted_tokens_are_each_rows_top_logprob(self):
        bs = 3
        logits = torch.randn(bs * WIDTH, VOCAB) * 3
        out_tokens, commit_lens = _greedy_step(logits, bs, matched=[0, 2, 3])
        self.assertEqual(commit_lens, [1, 3, 4])
        batch = _batch([0, 0, 0], [2, 2, 2])
        out = SimpleNamespace(next_token_logits=logits.clone())

        compute_spec_logprobs(batch, out, out_tokens.reshape(-1), chain_stride=WIDTH)

        self.assertEqual(tuple(out.next_token_logprobs.shape), (bs, WIDTH))
        for b in range(bs):
            for j in range(commit_lens[b]):
                row = b * WIDTH + j
                ref = torch.log_softmax(logits[row], -1)
                # The committed token is the argmax of its own logits row ...
                self.assertAlmostEqual(
                    out.next_token_logprobs[b, j].item(), ref.max().item(), places=5
                )
                # ... and so the head of that row's top logprobs.
                self.assertEqual(
                    out.next_token_top_logprobs_idx[row][0].item(),
                    out_tokens[b, j].item(),
                )
                self.assertEqual(len(out.next_token_top_logprobs_val[row]), 2)

    def test_sampled_rows_match_non_spec_semantics_before_top_p(self):
        bs = 2
        temps = [0.7, 1.3]
        logits = torch.randn(bs * WIDTH, VOCAB) * 2
        out_tokens = torch.randint(0, VOCAB, (bs, WIDTH))
        # top_p does not change the returned logprobs, as in Sampler.forward.
        batch = _batch(temps, [5, 0], top_ps=[0.5, 0.9])
        out = SimpleNamespace(next_token_logits=logits.clone())

        compute_spec_logprobs(batch, out, out_tokens.reshape(-1), chain_stride=WIDTH)

        for b in range(bs):
            for j in range(WIDTH):
                ref = _non_spec_logprobs(logits[b * WIDTH + j], temps[b])
                tok = out_tokens[b, j].item()
                self.assertAlmostEqual(
                    out.next_token_logprobs[b, j].item(), ref[tok].item(), places=4
                )
        # Top logprobs per (request, chain position), k per request.
        rows = out.next_token_top_logprobs_val
        self.assertEqual(len(rows), bs * WIDTH)
        for j in range(WIDTH):
            ref = _non_spec_logprobs(logits[j], temps[0])
            torch.testing.assert_close(rows[j], ref.topk(5).values, atol=1e-4, rtol=0)
            self.assertEqual(len(rows[WIDTH + j]), 0)

    def test_mixed_greedy_and_sampled_batch(self):
        temps = [0.0, 0.5]
        logits = torch.randn(2 * WIDTH, VOCAB)
        out_tokens = torch.randint(0, VOCAB, (2, WIDTH))
        out = SimpleNamespace(next_token_logits=logits.clone())

        compute_spec_logprobs(
            _batch(temps, [0, 0]), out, out_tokens.reshape(-1), chain_stride=WIDTH
        )

        for b, t in enumerate(temps):
            ref = _non_spec_logprobs(logits[b * WIDTH], t)
            self.assertAlmostEqual(
                out.next_token_logprobs[b, 0].item(),
                ref[out_tokens[b, 0]].item(),
                places=4,
            )

    def test_token_ids_logprobs_per_chain_position(self):
        logits = torch.randn(WIDTH, VOCAB)
        out_tokens = torch.randint(0, VOCAB, (1, WIDTH))
        out = SimpleNamespace(next_token_logits=logits.clone())

        compute_spec_logprobs(
            _batch([0], [0], token_ids_logprobs=[[3, 7]]),
            out,
            out_tokens.reshape(-1),
            chain_stride=WIDTH,
        )

        for j in range(WIDTH):
            ref = torch.log_softmax(logits[j], -1)[[3, 7]]
            got = out.next_token_token_ids_logprobs_val[j]
            got = got.tolist() if torch.is_tensor(got) else got
            self.assertEqual(len(got), 2)
            for g, r in zip(got, ref.tolist()):
                self.assertAlmostEqual(g, r, places=5)

    def test_masked_vocab_gives_finite_logprobs(self):
        # Grammar masks write -inf in place before the logprobs are taken.
        logits = torch.randn(WIDTH, VOCAB)
        logits[:, 10:] = float("-inf")
        out_tokens = torch.zeros(1, WIDTH, dtype=torch.int64)
        out = SimpleNamespace(next_token_logits=logits)

        compute_spec_logprobs(
            _batch([0], [3]), out, out_tokens.reshape(-1), chain_stride=WIDTH
        )

        self.assertTrue(torch.isfinite(out.next_token_logprobs).all())


class _Proc:
    """The SchedulerBatchResultProcessor state _apply_decode_logprobs reads."""

    logprob_result_processor = SchedulerLogprobResultProcessor


def _req(top_logprobs_num, max_new_tokens=64, stop_token_ids=None):
    sp = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0,
        stop_token_ids=stop_token_ids,
    )
    sp.normalize(None)
    return Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        return_logprob=True,
        top_logprobs_num=top_logprobs_num,
    )


def _host_logits_output(logits, out_tokens, temps, top_nums):
    """Run the verify-step logprobs, then the host copies the scheduler makes."""
    out = SimpleNamespace(next_token_logits=logits)
    compute_spec_logprobs(
        _batch(temps, top_nums),
        out,
        out_tokens.reshape(-1),
        chain_stride=WIDTH,
    )
    out.next_token_top_logprobs_val = [
        v.tolist() for v in out.next_token_top_logprobs_val
    ]
    out.next_token_top_logprobs_idx = [
        v.tolist() for v in out.next_token_top_logprobs_idx
    ]
    out.next_token_token_ids_logprobs_val = None
    out.next_token_token_ids_logprobs_idx = None
    return out, out.next_token_logprobs.tolist()


class TestSpecDecodeLogprobAssembly(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_every_accepted_token_gets_its_rows_logprob(self):
        bs = 2
        logits = torch.randn(bs * WIDTH, VOCAB) * 3
        out_tokens, commit_lens = _greedy_step(logits, bs, matched=[1, 3])
        reqs = [_req(2), _req(2)]
        out, flat = _host_logits_output(logits, out_tokens, [0, 0], [2, 2])
        batch = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)

        for i, req in enumerate(reqs):
            accepted = out_tokens[i, : commit_lens[i]].tolist()
            SchedulerBatchResultProcessor._apply_decode_logprobs(
                _Proc(),
                req=req,
                i=i,
                batch=batch,
                next_token_id=accepted,
                next_token_logprobs=flat,
                logits_output=out,
            )

        for i, req in enumerate(reqs):
            n = commit_lens[i]
            self.assertEqual(
                req.logprob.output_token_logprobs_idx, out_tokens[i, :n].tolist()
            )
            self.assertEqual(len(req.logprob.output_token_logprobs_val), n)
            self.assertEqual(len(req.logprob.output_top_logprobs_val), n)
            for j in range(n):
                ref = torch.log_softmax(logits[i * WIDTH + j], -1)
                self.assertAlmostEqual(
                    req.logprob.output_token_logprobs_val[j],
                    ref.max().item(),
                    places=5,
                )
                self.assertEqual(
                    req.logprob.output_top_logprobs_idx[j][0],
                    req.logprob.output_token_logprobs_idx[j],
                )

    def test_pd_first_token_then_spec_steps_then_stop_cut(self):
        """PD decode: the prefill's first token and logprob arrive through the
        metadata buffer, the verify steps append theirs, and an EOS inside an
        accepted run cuts the logprobs where it cuts the output."""
        eos = 9
        req = _req(3, stop_token_ids=[eos])
        proc = SimpleNamespace(
            logprob_result_processor=SchedulerLogprobResultProcessor,
            model_config=SimpleNamespace(
                think_end_ids=None, self_opened_think_start_ids=None
            ),
        )
        proc._maybe_update_reasoning_tokens = lambda r, t: (
            SchedulerBatchResultProcessor._maybe_update_reasoning_tokens(proc, r, t)
        )
        # decode.py _commit_transfer_to_req reads these from the metadata row.
        first_top_val = torch.tensor([-0.1, -2.5, -3.0, 0.0, 0.0])
        first_top_idx = torch.tensor([5, 6, 7, 0, 0], dtype=torch.int32)
        SchedulerBatchResultProcessor.commit_handoff_token(
            proc,
            req,
            5,
            logprob_token_id=5,
            logprob=-0.1,
            top_logprobs_val=first_top_val[: req.logprob.top_logprobs_num].tolist(),
            top_logprobs_idx=first_top_idx[: req.logprob.top_logprobs_num].tolist(),
        )
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1])
        self.assertEqual(req.logprob.output_top_logprobs_idx, [[5, 6, 7]])

        batch = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)
        # Step 1 commits 4 tokens; step 2 commits [11, eos, 12]: EOS mid-run.
        for step_tokens in ([20, 21, 22, 23], [11, eos, 12]):
            logits = torch.randn(WIDTH, VOCAB)
            out_tokens = torch.zeros(1, WIDTH, dtype=torch.int64)
            out_tokens[0, : len(step_tokens)] = torch.tensor(step_tokens)
            out, flat = _host_logits_output(logits, out_tokens, [0], [3])
            req.output_ids.extend(step_tokens)
            req.update_finish_state(len(step_tokens))
            SchedulerBatchResultProcessor._apply_decode_logprobs(
                proc,
                req=req,
                i=0,
                batch=batch,
                next_token_id=step_tokens,
                next_token_logprobs=flat,
                logits_output=out,
            )

        self.assertTrue(req.finished())
        through_stop = list(req.output_ids_through_stop)
        self.assertEqual(through_stop, [5, 20, 21, 22, 23, 11, eos])
        # One logprob per committed token, the over-accepted 12 included ...
        self.assertEqual(len(req.logprob.output_token_logprobs_idx), 8)
        self.assertEqual(req.logprob.output_token_logprobs_idx, list(req.output_ids))

        # ... and the streamer sends exactly the tokens through the stop.
        acc = _accumulator()
        acc.accept(req=req)
        sent_idx = acc.output_token_logprobs_idx[0]
        self.assertEqual(sent_idx, through_stop)
        self.assertEqual(len(acc.output_token_logprobs_val[0]), len(through_stop))
        self.assertEqual(len(acc.output_top_logprobs_val[0]), len(through_stop))
        self.assertTrue(all(len(r) == 3 for r in acc.output_top_logprobs_val[0]))
        self.assertTrue(
            all(v <= 0 and math.isfinite(v) for v in acc.output_token_logprobs_val[0])
        )


_STATS = RequestOutputStats(
    prompt_tokens=0,
    cached_tokens=0,
    cached_tokens_details=None,
    retraction_count=0,
    image_tokens=0,
    audio_tokens=0,
    video_tokens=0,
    time_stats=None,
)


def _accumulator():
    return _GenerationStreamAccumulator(
        return_logprob=True,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        disaggregation_mode=DisaggregationMode.DECODE,
        default_stream_interval=1,
        default_force_stream_interval=1,
        build_request_stats=lambda req: _STATS,
        output_sink=OutputSink.DETOKENIZER,
        is_request_level_customized_info_key=lambda key: False,
        current_weight_version=None,
    )


class TestDflashAdmitsLogprobs(unittest.TestCase):
    def test_return_logprob_is_admitted(self):
        req = SimpleNamespace(
            return_logprob=True,
            return_hidden_states=False,
            sampling_params=SimpleNamespace(
                json_schema=None, regex=None, ebnf=None, structural_tag=None
            ),
        )
        for masks in (True, False):
            self.assertIsNone(
                validate_dflash_request(
                    req, True, "xgrammar", masks_grammar_in_verify=masks
                )
            )


if __name__ == "__main__":
    unittest.main()
