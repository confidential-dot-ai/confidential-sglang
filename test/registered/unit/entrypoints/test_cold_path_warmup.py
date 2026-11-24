import asyncio
import multiprocessing
import os
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.entrypoints import warmup
from sglang.srt.entrypoints.warmup import COLD_PATH_SAMPLING, cold_path_plan
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GRAMMAR = ("json_schema", "regex", "ebnf", "structural_tag")


class TestColdPathPlan(unittest.TestCase):
    def test_no_speculation_runs_every_path_with_logprobs(self):
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.NONE)
        self.assertEqual(sampling, COLD_PATH_SAMPLING)
        self.assertTrue(logprobs)

    def test_dspark_drops_min_p_keeps_logprobs_and_the_rest(self):
        # The scheduler aborts min_p under any speculative algorithm; DSpark
        # serves logprobs (compute_spec_logprobs), so the warmup runs that path
        # and a first logprob request does not compile it; DSpark masks
        # grammars in verify.
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.DSPARK)
        self.assertTrue(logprobs)
        self.assertFalse(any("min_p" in s for s in sampling))
        self.assertIn({"temperature": 0.7, "top_p": 0.9}, sampling)
        # Joint top-k/top-p runs a torch.compile'd top-p rescale in DSpark verify.
        self.assertIn({"temperature": 0.7, "top_k": 40, "top_p": 0.9}, sampling)
        self.assertIn({"temperature": 0.7, "top_k": 1, "top_p": 0.9}, sampling)
        self.assertTrue(any("json_schema" in s for s in sampling))
        self.assertTrue(any("repetition_penalty" in s for s in sampling))
        self.assertIn({}, sampling)
        self.assertEqual(len(sampling), len(COLD_PATH_SAMPLING))

    def test_dflash_also_drops_grammar(self):
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.DFLASH)
        self.assertTrue(logprobs)
        self.assertFalse(any(k in s for s in sampling for k in GRAMMAR + ("min_p",)))

    def test_eagle_keeps_logprobs_without_min_p(self):
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.EAGLE)
        self.assertTrue(logprobs)
        self.assertFalse(any("min_p" in s for s in sampling))

    def test_uno_drops_penalties_grammar_and_logprobs(self):
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.UNO)
        self.assertFalse(logprobs)
        self.assertFalse(any(k in s for s in sampling
                             for k in GRAMMAR + ("min_p", "repetition_penalty",
                                                 "frequency_penalty", "presence_penalty")))
        self.assertEqual(len(sampling), len({repr(sorted(s.items())) for s in sampling}))

    def test_pd_decode_drops_grammar_paths(self):
        # A decode warmup request's first token comes from the fake bootstrap,
        # not from a grammar-constrained prefill.
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.DSPARK, "decode")
        self.assertFalse(any(k in s for s in sampling for k in GRAMMAR + ("min_p",)))
        self.assertTrue(logprobs)
        prefill, _ = cold_path_plan(SpeculativeAlgorithm.DSPARK, "prefill")
        self.assertTrue(any("json_schema" in s for s in prefill))
        plain, logprobs = cold_path_plan(SpeculativeAlgorithm.NONE, "decode")
        self.assertTrue(logprobs)
        self.assertFalse(any("json_schema" in s for s in plain))


class TestLongPrompts(unittest.TestCase):
    def test_lengths_follow_what_the_worker_admits(self):
        # 1M context: ~80K and nearly the longest admitted prompt, with room for
        # the follow-up and its new tokens.
        lengths = warmup.long_prompt_lengths(1048576 - 6)
        self.assertEqual(lengths[0], warmup.LONG_PROMPT_TOKENS)
        self.assertLessEqual(lengths[-1] + warmup.LONG_PROMPT_FOLLOW_UP + 16, 1048576 - 6)
        self.assertGreater(lengths[-1], 1048576 - 1024 - 8192)
        # Both end in an eager 8K-chunk tail past the 65536 prefill graph cap:
        # 128..255 tokens (an MXFP8 GEMM tactic bucket) and 1024..8191 tokens,
        # neither a multiple of 16.
        short_tail, long_tail = (n % 8192 for n in lengths)
        self.assertTrue(128 <= short_tail < 256 and short_tail % 16)
        self.assertTrue(1024 <= long_tail < 8192 and long_tail % 16)
        self.assertGreater(lengths[0] - short_tail, 65536)
        # A pool smaller than the context bounds it, not the context.
        self.assertLess(warmup.long_prompt_lengths(700_000)[-1], 700_000)
        # Short-context servers skip the step.
        self.assertEqual(warmup.long_prompt_lengths(32768), ())
        self.assertEqual(warmup.long_prompt_lengths(None), ())

    def test_the_follow_up_extends_the_long_prompt(self):
        seen = []

        class Manager:
            max_req_input_len = 262144 - 6
            context_len = 262144

            async def generate_request(self, req, _):
                seen.append(req)
                yield {}

        spec = mock.Mock(speculative_algorithm="DSPARK")
        with mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec):
            asyncio.run(warmup.confidential_cold_path("prefill", Manager()))
        long = [r for r in seen if len(r.input_ids) > 32768]
        self.assertEqual([len(r.input_ids) for r in long],
                         [82120, 82120 + 64, 258955, 258955 + 64])
        for first, follow_up in (long[0:2], long[2:4]):
            self.assertEqual(follow_up.input_ids[:len(first.input_ids)], first.input_ids)
            self.assertNotIn("min_p", follow_up.sampling_params)
            self.assertIsNotNone(first.bootstrap_room)


def _tokenizer_process(server_key, log, ready):
    async def fake(*_):
        with open(log, "a") as f:
            f.write(f"start {os.getpid()} {time.time()}\n")
        await asyncio.sleep(0.5)
        with open(log, "a") as f:
            f.write(f"end {os.getpid()} {time.time()}\n")

    with mock.patch.object(warmup, "_run_warmups", fake), \
            mock.patch.object(warmup, "_warmup_lock_dir", lambda: os.path.dirname(log)):
        asyncio.run(warmup.execute_warmups("decode", ["x"], None, server_key=server_key))
    with open(ready, "a") as f:
        f.write(f"{time.time()}\n")  # this process would now finish startup


class TestWarmupsOnce(unittest.TestCase):
    def test_four_tokenizer_processes_warm_up_once_and_none_starts_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, ready = os.path.join(tmp, "log"), os.path.join(tmp, "ready")
            ctx = multiprocessing.get_context("fork")
            procs = [ctx.Process(target=_tokenizer_process, args=(str(os.getpid()), log, ready)) for _ in range(4)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(30)
                self.assertEqual(p.exitcode, 0)
            lines = open(log).read().split()
            self.assertEqual(lines.count("start"), 1)
            end = float(lines[lines.index("end") + 2])
            readies = [float(x) for x in open(ready).read().split()]
            self.assertEqual(len(readies), 4)
            self.assertTrue(all(t >= end for t in readies))

    def test_single_tokenizer_runs_directly(self):
        calls = []

        async def fake(*args):
            calls.append(args)

        with mock.patch.object(warmup, "_run_warmups", fake):
            self.assertTrue(asyncio.run(warmup.execute_warmups("null", ["x"], None, server_key=None)))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
