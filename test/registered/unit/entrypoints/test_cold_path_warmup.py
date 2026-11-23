import unittest

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

    def test_dspark_drops_min_p_and_logprobs_keeps_the_rest(self):
        # The scheduler aborts min_p under any speculative algorithm and
        # return_logprob under DSpark; DSpark masks grammars in verify.
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.DSPARK)
        self.assertFalse(logprobs)
        self.assertFalse(any("min_p" in s for s in sampling))
        self.assertIn({"temperature": 0.7, "top_p": 0.9}, sampling)
        self.assertTrue(any("json_schema" in s for s in sampling))
        self.assertTrue(any("repetition_penalty" in s for s in sampling))
        self.assertIn({}, sampling)
        self.assertEqual(len(sampling), len(COLD_PATH_SAMPLING))

    def test_dflash_also_drops_grammar(self):
        sampling, logprobs = cold_path_plan(SpeculativeAlgorithm.DFLASH)
        self.assertFalse(logprobs)
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


if __name__ == "__main__":
    unittest.main()
