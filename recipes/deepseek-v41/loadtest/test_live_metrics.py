import math
import unittest

import live_metrics as lm

PREFILL_0 = """
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="dsv41",tp_rank="0"} 1000.0
sglang:prompt_tokens_created{model_name="dsv41"} 1.7e9
sglang:cached_tokens_total{model_name="dsv41"} 800.0
sglang:token_usage{model_name="dsv41"} 0.5
sglang:evicted_tokens_total{model_name="dsv41"} 0
sglang:hicache_backup_tokens_total{model_name="dsv41"} 0
sglang:load_back_tokens_total{model_name="dsv41"} 0
sglang:num_running_reqs{model_name="dsv41"} 2
sglang:num_queue_reqs{model_name="dsv41"} 1
sglang:queue_time_seconds_bucket{le="0.1"} 0
sglang:queue_time_seconds_bucket{le="1.0"} 0
sglang:queue_time_seconds_bucket{le="+Inf"} 0
"""

PREFILL_1 = """
sglang:prompt_tokens_total{model_name="dsv41",tp_rank="0"} 21000.0
sglang:cached_tokens_total{model_name="dsv41"} 19800.0
sglang:token_usage{model_name="dsv41"} 0.9
sglang:evicted_tokens_total{model_name="dsv41"} 500
sglang:hicache_backup_tokens_total{model_name="dsv41"} 1000
sglang:load_back_tokens_total{model_name="dsv41"} 2000
sglang:num_running_reqs{model_name="dsv41"} 4
sglang:num_queue_reqs{model_name="dsv41"} 3
sglang:queue_time_seconds_bucket{le="0.1"} 5
sglang:queue_time_seconds_bucket{le="1.0"} 10
sglang:queue_time_seconds_bucket{le="+Inf"} 10
"""

DECODE_0 = """
sglang:prompt_tokens_total{model_name="dsv41"} 1000.0
sglang:generation_tokens_total{model_name="dsv41"} 100
sglang:num_requests_total{model_name="dsv41"} 10
sglang:num_retracted_requests_total{model_name="dsv41"} 0
sglang:token_usage{model_name="dsv41"} 0.2
sglang:time_to_first_token_seconds_bucket{le="0.5"} 0
sglang:time_to_first_token_seconds_bucket{le="1.0"} 0
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 0
"""

DECODE_1 = """
sglang:prompt_tokens_total{model_name="dsv41"} 21000.0
sglang:generation_tokens_total{model_name="dsv41"} 300
sglang:num_requests_total{model_name="dsv41"} 20
sglang:num_retracted_requests_total{model_name="dsv41"} 3
sglang:token_usage{model_name="dsv41"} 0.4
sglang:time_to_first_token_seconds_bucket{le="0.5"} 50
sglang:time_to_first_token_seconds_bucket{le="1.0"} 100
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 100
"""

ROUTER_0 = """
smg_http_responses_total{status="200"} 10
smg_http_responses_total{status="429"} 1
smg_admission_rejected_total 1
smg_admission_inflight 3
smg_http_requests_total{method="POST",path="/v1/chat/completions"} 11
"""

ROUTER_1 = """
smg_http_responses_total{status="200"} 20
smg_http_responses_total{status="429"} 6
smg_http_responses_total{status="503"} 4
smg_admission_rejected_total 6
smg_admission_inflight 7
smg_http_requests_total{method="POST",path="/v1/chat/completions"} 30
"""


def snap(t, p, d, r=None):
    return lm.Snapshot(t, lm.parse(p), lm.parse(d), None if r is None else lm.parse(r))


class LiveMetricsTest(unittest.TestCase):
    def test_parse_sums_labels_and_skips_created_and_comments(self):
        s = lm.parse(PREFILL_0)
        self.assertEqual(s.values["sglang:prompt_tokens_total"], 1000.0)
        self.assertNotIn("sglang:prompt_tokens_created", s.values)
        self.assertEqual(s.buckets["sglang:queue_time_seconds"][math.inf], 0)

    def test_pd_window_takes_input_from_prefill_and_output_from_decode(self):
        rec = lm.window(snap(10, PREFILL_1, DECODE_1, ROUTER_1), snap(0, PREFILL_0, DECODE_0, ROUTER_0), gpus=4)
        t, c = rec["traffic"], rec["cache"]
        # 20000 prompt tokens in 10 s: prefill only, not prefill + decode.
        self.assertEqual(t["input_tok_s"], 2000)
        self.assertEqual(t["cached_input_tok_s"], 1900)
        self.assertEqual(t["uncached_input_tok_s"], 100)
        self.assertEqual(t["output_tok_s"], 20)
        self.assertEqual(t["requests_per_s"], 1)
        self.assertAlmostEqual(t["served_tpm_per_gpu"], 2020 * 60 / 4)
        self.assertAlmostEqual(c["prefix_hit"], 0.95)
        self.assertEqual(c["prefill_kv_use"], 0.9)
        self.assertEqual(c["decode_kv_use"], 0.4)
        self.assertEqual(c["evicted_tok_s"], 50)
        self.assertEqual(c["hicache_writeback_tok_s"], 100)
        self.assertEqual(c["hicache_loadback_tok_s"], 200)
        self.assertAlmostEqual(rec["preemption"]["retracted_req_s"], 0.3)
        self.assertEqual(rec["queues"]["prefill_waiting"], 3)
        self.assertEqual(rec["queues"]["router_inflight"], 7)
        self.assertAlmostEqual(rec["queues"]["router_rejected_s"], 0.5)
        # 5 more 429s and 4 new 503s.
        self.assertAlmostEqual(rec["health"]["router_non2xx_s"], 0.9)

    def test_histogram_window_quantiles_interpolate_within_buckets(self):
        rec = lm.window(snap(10, PREFILL_1, DECODE_1), snap(0, PREFILL_0, DECODE_0), gpus=0)
        ttft = rec["latency_s"]["ttft"]
        self.assertAlmostEqual(ttft["p50"], 0.5)
        self.assertAlmostEqual(ttft["p90"], 0.5 + 0.5 * (90 - 50) / 50)
        self.assertIsNone(rec["traffic"]["served_tpm_per_gpu"])
        self.assertIsNone(rec["queues"]["router_inflight"])

    def test_quantile_past_last_finite_bucket_reports_that_bound(self):
        q = lm.histogram_quantiles({1.0: 1, math.inf: 10}, {1.0: 0, math.inf: 0})
        self.assertEqual(q["p99"], 1.0)

    def test_empty_histogram_window_has_no_quantiles(self):
        q = lm.histogram_quantiles({1.0: 5, math.inf: 5}, {1.0: 5, math.inf: 5})
        self.assertEqual(q, {"p50": None, "p90": None, "p99": None})

    def test_counter_reset_counts_from_zero(self):
        self.assertEqual(lm.counter_delta(5, 100), 5)
        self.assertEqual(lm.counter_delta(105, 100), 5)
        self.assertIsNone(lm.counter_delta(None, 100))

    def test_colocated_worker_supplies_both_sides(self):
        both = PREFILL_1 + DECODE_1.replace("sglang:prompt_tokens_total", "x_prompt").replace(
            "sglang:token_usage", "x_usage"
        )
        both0 = PREFILL_0 + DECODE_0.replace("sglang:prompt_tokens_total", "x_prompt").replace(
            "sglang:token_usage", "x_usage"
        )
        new, old = lm.parse(both), lm.parse(both0)
        rec = lm.window(lm.Snapshot(10, new, new, None), lm.Snapshot(0, old, old, None), gpus=4)
        self.assertEqual(rec["traffic"]["input_tok_s"], 2000)
        self.assertEqual(rec["traffic"]["output_tok_s"], 20)
        # Colocated: one process, so counters are not summed twice.
        self.assertEqual(rec["cache"]["evicted_tok_s"], 50)

    def test_merge_sums_workers_of_one_role(self):
        m = lm.merge([lm.parse(PREFILL_1), lm.parse(PREFILL_1)])
        self.assertEqual(m.values["sglang:prompt_tokens_total"], 42000)
        self.assertEqual(m.buckets["sglang:queue_time_seconds"][1.0], 20)

    def test_missing_worker_metrics_are_reported(self):
        s = lm.Snapshot(0, lm.parse(""), lm.parse(""), None)
        self.assertIn("prefill:sglang:prompt_tokens_total", lm.check_enabled(s))
        self.assertEqual(lm.check_enabled(snap(0, PREFILL_0, DECODE_0)), [])

    def test_scraper_needs_pd_pair_or_colocated_workers(self):
        with self.assertRaises(ValueError):
            lm.Scraper(["p"], [], [], "", 1)
        with self.assertRaises(ValueError):
            lm.Scraper(["p"], ["d"], ["w"], "", 1)

    def test_status_line_renders_missing_values(self):
        rec = lm.window(snap(10, PREFILL_1, DECODE_1), snap(0, PREFILL_0, DECODE_0), gpus=0)
        text = lm.line(rec)
        self.assertIn("hit 95.0%", text)
        self.assertIn("TPM/GPU -", text)

    def test_router_only_window_reports_router_side_and_nothing_else(self):
        empty = lm.Scrape()
        new = lm.Snapshot(10, empty, empty, lm.parse(ROUTER_1))
        old = lm.Snapshot(0, empty, empty, lm.parse(ROUTER_0))
        rec = lm.window(new, old, gpus=4)
        self.assertAlmostEqual(rec["traffic"]["router_requests_per_s"], 1.9)
        self.assertIsNone(rec["traffic"]["input_tok_s"])
        self.assertIsNone(rec["latency_s"]["ttft"]["p50"])
        text = lm.router_line(rec)
        self.assertIn("router req/s 1.90", text)
        self.assertIn("in flight 7", text)

    def test_router_only_needs_a_router(self):
        with self.assertRaises(ValueError):
            lm.Scraper([], [], [], "", 1, router_only=True)
        s = lm.Scraper(["p"], ["d"], [], "http://r", 1, router_only=True)
        self.assertEqual((s.prefill, s.decode), ([], []))

    def test_worker_without_metrics_endpoint_exits_with_a_hint(self):
        def boom(url, timeout):
            raise lm.MetricsUnavailable(url + "/metrics returned 404")

        orig, lm.fetch = lm.fetch, boom
        try:
            self.assertEqual(lm.main(["--prefill", "http://p", "--decode", "http://d"]), 2)
        finally:
            lm.fetch = orig

    def test_unreachable_endpoint_exits_2(self):
        self.assertEqual(lm.main(["--prefill", "http://127.0.0.1:1", "--decode", "http://127.0.0.1:2"]), 2)

    def _rec(self):
        return lm.window(
            snap(10, PREFILL_1, DECODE_1, ROUTER_1), snap(0, PREFILL_0, DECODE_0, ROUTER_0), gpus=4
        )

    def test_exposition_has_gauges_quantiles_labels_and_skips_missing(self):
        text = lm.exposition(self._rec(), {"topology": "1P1D"})
        self.assertIn('inference_up{topology="1P1D"} 1', text)
        self.assertIn('inference_traffic_input_tok_s{topology="1P1D"} 2000.0', text)
        self.assertIn('inference_cache_prefix_hit{topology="1P1D"} 0.95', text)
        self.assertIn('inference_latency_seconds{quantile="0.5",stage="ttft",topology="1P1D"} 0.5', text)
        # TPM per GPU present with --gpus; spec accept length missing -> omitted
        self.assertIn("inference_traffic_served_tpm_per_gpu", text)
        self.assertNotIn("spec_accept_length", text)
        # every sample line parses back with this module's own parser
        back = lm.parse(text)
        self.assertEqual(back.values["inference_up"], 1)
        for ln in text.splitlines():
            self.assertTrue(ln.startswith("# TYPE ") or lm._SAMPLE.match(ln), ln)

    def test_label_values_are_escaped_and_bad_labels_rejected(self):
        text = lm.exposition(None, {"note": 'a"b\\c'}, up=False)
        self.assertIn('inference_up{note="a\\"b\\\\c"} 0', text)
        self.assertEqual(lm.parse_labels(["topology=4x", "gpu=B200"]), {"topology": "4x", "gpu": "B200"})
        for bad in (["novalue"], ["1x=a"], ["quantile=1"]):
            with self.assertRaises(ValueError):
                lm.parse_labels(bad)

    def test_exporter_serves_latest_window_and_flags_failed_scrapes(self):
        import urllib.request

        ex = lm.Exporter("127.0.0.1", 0, {"topology": "t"})
        try:
            url = f"http://127.0.0.1:{ex.port}/metrics"
            get = lambda: urllib.request.urlopen(url, timeout=5).read().decode()
            self.assertIn('inference_up{topology="t"} 0', get())
            ex.update(self._rec(), up=True)
            body = get()
            self.assertIn('inference_up{topology="t"} 1', body)
            self.assertIn("inference_traffic_output_tok_s", body)
            ex.update(None, up=False)
            body = get()
            self.assertIn('inference_up{topology="t"} 0', body)
            self.assertIn("inference_traffic_output_tok_s", body)  # last values kept
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://127.0.0.1:{ex.port}/other", timeout=5)
        finally:
            ex.close()

    def test_first_snapshot_waits_for_metrics_when_serving(self):
        good = snap(0, PREFILL_0, DECODE_0)
        calls = []

        class Flaky:
            router_only = False

            def snapshot(self):
                calls.append(1)
                if len(calls) == 1:
                    raise ConnectionRefusedError("down")
                if len(calls) == 2:
                    return lm.Snapshot(0, lm.parse(""), lm.parse(""), None)  # metrics off
                return good

        self.assertIs(lm.first_snapshot(Flaky(), 0.001), good)
        self.assertEqual(len(calls), 3)
        calls.clear()
        with self.assertRaises(ConnectionRefusedError):
            lm.first_snapshot(Flaky(), 0)

    def test_metrics_unavailable_is_a_scrape_failure(self):
        self.assertTrue(issubclass(lm.MetricsUnavailable, OSError))


if __name__ == "__main__":
    unittest.main()
