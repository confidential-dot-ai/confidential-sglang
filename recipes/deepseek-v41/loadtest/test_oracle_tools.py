import argparse
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import aiohttp  # noqa: F401  (oracle_replay imports it)

    import oracle_replay
    import oracle_scorecard
except ImportError:  # host python without aiohttp
    oracle_replay = oracle_scorecard = None



@unittest.skipIf(oracle_replay is None, "needs aiohttp")
class ForcedTextTest(unittest.TestCase):
    def test_realmix_takes_the_next_turn_text_and_decodes_the_rest(self):
        convs = [
            [
                {"messages": [{"role": "user", "content": "q"}], "forced_token_ids": [1], "max_new_tokens": 1},
                {
                    "messages": [{"role": "assistant", "content": "abc"}, {"role": "user", "content": "q2"}],
                    "forced_token_ids": [2],
                    "max_new_tokens": 1,
                },
            ]
        ]
        # The last step has no following text and no decoder: refuse.
        with self.assertRaises(SystemExit):
            oracle_replay.attach_forced_text(convs, "")
        self.assertEqual(convs[0][0]["_forced_text"], "abc")

    def test_tracelab_steps_need_the_source_tokenizer(self):
        convs = [[{"messages": [], "workload": {}, "reset_history": False, "forced_token_ids": [5]}]]
        with self.assertRaises(SystemExit):
            oracle_replay.attach_forced_text(convs, "")


@unittest.skipIf(oracle_replay is None, "needs aiohttp")
class ScorecardTest(unittest.TestCase):
    def summary(self, **over):
        s = {
            "valid_window": True,
            "window": {
                "window_s": 600,
                "traffic": {"output_share": 0.02},
                "cache": {"prefix_hit": 0.95},
                "preemption": {"retracted_req_s": 0.0},
                "health": {"kv_transfer_failed_s": 0.0},
            },
            "client": {"failed": 0, "ttft_s": {50: 1.0}},
        }
        for k, v in over.items():
            s[k] = v
        return s

    def args(self, **over):
        a = argparse.Namespace(cache_hit_band=(0.9, 0.97), output_share_band=(0.01, 0.03), ttft_p50_max=3.0)
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def test_passing_window_has_no_failures(self):
        self.assertEqual(oracle_scorecard.gates(self.summary(), self.args()), [])

    def test_each_gate_fails_on_its_own(self):
        s = self.summary()
        s["window"]["cache"]["prefix_hit"] = 0.5
        s["window"]["preemption"]["retracted_req_s"] = 0.01
        s["client"]["failed"] = 2
        s["client"]["ttft_s"][50] = 9.0
        s["valid_window"] = False
        failures = oracle_scorecard.gates(s, self.args())
        self.assertEqual(len(failures), 5, failures)

    def test_band_parsing(self):
        self.assertEqual(oracle_scorecard.band("0.94,0.96"), (0.94, 0.96))
        with self.assertRaises(argparse.ArgumentTypeError):
            oracle_scorecard.band("0.9,1.5")

    def test_load_attaches_forced_text_only_in_oracle_mode(self):
        steps = [
            {"messages": [{"role": "user", "content": "q"}], "forced_token_ids": [1], "max_new_tokens": 1},
            {"messages": [{"role": "assistant", "content": "a"}], "forced_token_ids": [2], "max_new_tokens": 1},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"metadata": {}, "conversations": [steps]}, f)
        try:
            args = oracle_replay.parser().parse_args([f.name, "--max-steps", "1"])
            self.assertNotIn("_forced_text", oracle_replay.load(args)[0][0])
            args.oracle = True
            with self.assertRaises(SystemExit):  # last step needs a decoder
                oracle_replay.load(args)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
