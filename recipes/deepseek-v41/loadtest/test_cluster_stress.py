"""The cluster stress tools against a fake gateway (no GPU).

The fake requires a Bearer key, refuses past its capacity with 429, can be taken
down for a few seconds (503), answers needle prompts right or wrong on demand,
and serves a Prometheus query API.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

try:
    from aiohttp import web
except ImportError:  # host python without aiohttp
    web = None

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "sk-test-not-a-real-key-7f3a"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeGateway:
    def __init__(self):
        self.capacity, self.inflight = 1000, 0
        self.down_until, self.wrong_needles, self.over_capacity_status = 0.0, False, 429
        self.unauthorized = 0
        self.port = free_port()
        self.counters = {"prompt": 0.0, "cached": 0.0, "generated": 0.0}

    def app(self):
        app = web.Application()
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/_down", self.down)
        app.router.add_get("/api/v1/query", self.prom_query)
        return app

    async def models(self, request):
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"data": [{"id": "fake-model"}]})

    async def down(self, request):
        self.down_until = time.time() + float(request.query["s"])
        return web.json_response({"ok": True})

    def _authorized(self, request):
        ok = request.headers.get("authorization") == f"Bearer {KEY}"
        self.unauthorized += not ok
        return ok

    async def chat(self, request):
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        if time.time() < self.down_until:
            return web.json_response({"error": "no upstream"}, status=503)
        if self.inflight >= self.capacity:
            return web.json_response({"error": "busy"}, status=self.over_capacity_status)
        self.inflight += 1
        try:
            body = await request.json()
            await asyncio.sleep(0.02)
            return await self._answer(request, body)
        finally:
            self.inflight -= 1

    async def _answer(self, request, body):
        if "min_p" in body:  # as DSpark does: a 400, or for a stream a 200 with an error event
            err = {"object": "error", "message": "min_p is not supported with DSPARK", "code": 400}
            if not body.get("stream"):
                return web.json_response({"error": err}, status=400)
            resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(f"data: {json.dumps({'error': err})}\n\ndata: [DONE]\n\n".encode())
            return resp
        content = body["messages"][-1]["content"]
        m = re.search(r"access code is (\d{6})", content)
        text = ("000000" if self.wrong_needles else m.group(1)) if m else "fine, thanks"
        prompt_tokens = max(1, len(content) // 4)
        self.counters["prompt"] += prompt_tokens
        self.counters["cached"] += prompt_tokens // 2
        self.counters["generated"] += 5
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 5, "total_tokens": prompt_tokens + 5}
        if body.get("stream"):
            resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await resp.prepare(request)
            for word in ("one", "two", "three"):
                chunk = {"choices": [{"index": 0, "delta": {"content": word}}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n".encode())
            return resp
        msg = {"role": "assistant", "content": text}
        if body.get("tools") and body.get("tool_choice") != "none":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]}
        return web.json_response({"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage})

    async def prom_query(self, request):
        q = request.query["query"]
        role = "prefill" if 'role="prefill"' in q else "decode" if 'role="decode"' in q else "router"
        series = []
        if role == "prefill":
            for rank in ("0", "1"):
                series += [
                    ("sglang:prompt_tokens_total", {"tp_rank": rank}, self.counters["prompt"] / 2),
                    ("sglang:cached_tokens_total", {"tp_rank": rank}, self.counters["cached"] / 2),
                    ("sglang:token_usage", {"tp_rank": rank}, 0.5),
                ]
        elif role == "decode":
            series += [("sglang:generation_tokens_total", {"tp_rank": "0"}, self.counters["generated"]),
                       ("sglang:token_usage", {"tp_rank": "0"}, 0.3)]
        result = [{"metric": {"__name__": n, "role": role, **lab}, "value": [time.time(), str(v)]}
                  for n, lab, v in series]
        return web.json_response({"status": "success", "data": {"resultType": "vector", "result": result}})


def start(fake):
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def serve():
        runner = web.AppRunner(fake.app())
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", fake.port).start()
        ready.set()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)
    return fake


def tool(script, *argv, env_key=True, timeout=120, nofile=None):
    env = dict(os.environ)
    if env_key:
        env["STRESS_API_KEY"] = KEY
    else:
        env.pop("STRESS_API_KEY", None)
    def limit():
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))

    return subprocess.run([sys.executable, os.path.join(HERE, script), *argv], capture_output=True, text=True,
                          env=env, timeout=timeout, cwd=HERE, preexec_fn=limit if nofile else None)


@unittest.skipIf(web is None, "needs aiohttp")
class ClusterStressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = start(FakeGateway())
        cls.url = f"http://127.0.0.1:{cls.fake.port}"

    def setUp(self):
        self.fake.capacity, self.fake.wrong_needles, self.fake.over_capacity_status = 1000, False, 429
        self.fake.down_until = 0.0
        self.tmp = tempfile.mkdtemp()

    def out(self, name):
        return os.path.join(self.tmp, name)

    def load(self, name):
        with open(self.out(name)) as f:
            return json.load(f)

    def test_the_key_comes_from_the_environment_and_is_never_written(self):
        r = tool("needles.py", "--url", self.url, "--sizes", "2000", "--seeds", "1", "--out", self.out("n.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.out("n.json")) as f:
            text = f.read()
        self.assertNotIn(KEY, text)
        self.assertTrue(json.loads(text)["endpoint"]["api_key"])

    def test_without_the_key_the_gateway_refuses(self):
        r = tool("needles.py", "--url", self.url, "--sizes", "2000", "--out", self.out("n.json"), env_key=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("401", r.stderr)

    def test_needles_fail_on_a_wrong_answer(self):
        self.fake.wrong_needles = True
        r = tool("needles.py", "--url", self.url, "--sizes", "2000,4000", "--seeds", "1", "--out", self.out("n.json"))
        self.assertEqual(r.returncode, 1, r.stdout)
        res = self.load("n.json")
        self.assertEqual((res["misses"], res["cases"]), (4, 4))

    def test_overload_passes_when_excess_load_gets_429(self):
        self.fake.capacity = 4
        r = tool("overload.py", "--url", self.url, "--levels", "16", "--step-s", "2", "--burst", "40",
                 "--recover-s", "10", "--out", self.out("o.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.load("o.json")
        self.assertGreater(sum(s["classes"]["429"] for s in res["steps"]), 0)
        self.assertEqual(sum(s["classes"]["5xx"] for s in res["steps"]), 0)
        self.assertIsNotNone(res["recovered_after_s"])
        self.assertTrue(res["shed"])

    def test_overload_can_require_that_the_endpoint_was_saturated(self):
        r = tool("overload.py", "--url", self.url, "--levels", "4", "--step-s", "1", "--burst", "0",
                 "--recover-s", "10", "--require-shed", "--out", self.out("o.json"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertFalse(self.load("o.json")["shed"])

    def test_a_client_out_of_file_descriptors_is_not_blamed_on_the_server(self):
        r = tool("overload.py", "--url", self.url, "--levels", "200", "--step-s", "2", "--burst", "0",
                 "--recover-s", "10", "--out", self.out("o.json"), nofile=100)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("raise `ulimit -n`", r.stdout)
        res = self.load("o.json")
        self.assertGreater(res["steps"][0]["classes"]["client"], 0)
        self.assertEqual(res["steps"][0]["classes"]["conn"], 0)
        # About 110 clients without a socket, backing off 0.5 s for 2 s: hundreds of attempts, not a spin.
        self.assertLess(res["steps"][0]["classes"]["client"], 2000)

    def test_overload_fails_on_5xx(self):
        self.fake.capacity, self.fake.over_capacity_status = 4, 500
        r = tool("overload.py", "--url", self.url, "--levels", "16", "--step-s", "2", "--burst", "0",
                 "--recover-s", "10", "--out", self.out("o.json"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("5xx", self.load("o.json")["gate_failures"][0])

    def test_soak_reports_per_minute_and_passes_clean_traffic(self):
        r = tool("soak.py", "--url", self.url, "--minutes", "0.05", "--concurrency", "4",
                 "--max-prompt-tokens", "6000", "--unsupported-params", "min_p", "--out", self.out("s.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.load("s.json")
        self.assertGreater(res["total"]["requests"], 10)
        self.assertEqual(res["total"]["failures"], 0)
        self.assertTrue(res["by_minute"])

    def test_soak_counts_errors_inside_a_200_stream(self):
        sys.path.insert(0, HERE)
        import traffic

        os.environ["STRESS_API_KEY"] = KEY
        from types import SimpleNamespace

        import endpoint

        async def go():
            args = SimpleNamespace(url=self.url, api_key_env="STRESS_API_KEY", ca_cert="")
            async with endpoint.client_session(args) as session:
                body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "min_p": 0.05}
                return await traffic.send(session, self.url + "/v1/chat/completions", "fake-model", "stream_full",
                                          True, body, __import__("random").Random(0), time.time(), 30)

        row = asyncio.run(go())
        self.assertEqual(row["cls"], "4xx", row)
        self.assertIn("in-stream 400", row["err"])

    def test_soak_leaves_out_parameters_the_deployment_rejects(self):
        r = tool("soak.py", "--url", self.url, "--minutes", "0.05", "--concurrency", "4",
                 "--max-prompt-tokens", "6000", "--out", self.out("s.json"))
        self.assertEqual(r.returncode, 1, r.stdout)  # the default mix sends min_p
        self.assertGreater(self.load("s.json")["total"]["classes"]["4xx"], 0)
        r = tool("soak.py", "--url", self.url, "--minutes", "0.05", "--concurrency", "4", "--max-prompt-tokens",
                 "6000", "--unsupported-params", "min_p", "--out", self.out("s.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_replay_counts_an_in_stream_error_as_a_failed_step(self):
        sys.path.insert(0, HERE)
        import oracle_replay
        from types import SimpleNamespace

        os.environ["STRESS_API_KEY"] = KEY
        import endpoint

        args = SimpleNamespace(url=self.url, api_key_env="STRESS_API_KEY", ca_cert="", model="fake-model")
        stats = oracle_replay.Stats()
        stats.active_sessions, stats.skipped_context = 0, 0

        async def go():
            async with endpoint.client_session(args) as session:
                with open(self.out("r.jsonl"), "w") as out:
                    hist = [{"role": "user", "content": "hi"}]
                    # min_p rides along in the body through a wrapper session.post
                    orig = session.post

                    def post(url, data=None, **kw):
                        body = json.loads(data)
                        body["min_p"] = 0.05
                        return orig(url, data=json.dumps(body), **kw)

                    session.post = post
                    return await oracle_replay.run_step(session, args, 0, 0, hist, 8, stats, out)

        self.assertIsNone(asyncio.run(go()))
        self.assertEqual(list(stats.errors), ["400: min_p is not supported with DSPARK"])
        self.assertEqual(stats.rows, [])

    def test_fault_measures_the_outage_and_the_recovery(self):
        cmd = f"{sys.executable} -c \"import urllib.request; urllib.request.urlopen('{self.url}/_down?s=3')\""
        r = tool("fault.py", "--url", self.url, "--fault-cmd", cmd, "--concurrency", "4", "--baseline-s", "2",
                 "--window-s", "2", "--settle-s", "4", "--recover-timeout-s", "40", "--after-s", "1",
                 "--max-prompt-tokens", "6000", "--unsupported-params", "min_p", "--check-cmd", "true",
                 "--out", self.out("f.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.load("f.json")
        self.assertGreater(res["outage"]["failed_requests"], 0)
        self.assertEqual(res["outage"]["classes"]["5xx"], res["outage"]["failed_requests"])
        self.assertGreaterEqual(res["recovered_after_s"], 4)
        self.assertEqual(res["fault_command"]["exit_code"], 0)

    def test_fault_fails_when_the_lost_requests_exceed_the_cap(self):
        cmd = f"{sys.executable} -c \"import urllib.request; urllib.request.urlopen('{self.url}/_down?s=2')\""
        r = tool("fault.py", "--url", self.url, "--fault-cmd", cmd, "--concurrency", "4", "--baseline-s", "1",
                 "--window-s", "1", "--settle-s", "3", "--recover-timeout-s", "30", "--after-s", "0",
                 "--max-prompt-tokens", "6000", "--unsupported-params", "min_p", "--max-failed-requests", "0",
                 "--out", self.out("f.json"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("requests failed (max 0)", " ".join(self.load("f.json")["gate_failures"]))

    def test_prometheus_snapshot_sums_ranks_like_a_direct_scrape(self):
        sys.path.insert(0, HERE)
        import prom_metrics

        self.fake.counters = {"prompt": 1000.0, "cached": 400.0, "generated": 50.0}
        s = prom_metrics.PromScraper(self.url, prefill='role="prefill"', decode='role="decode"', lag_s=0)
        snap = s.snapshot()
        self.assertEqual(snap.prefill.values["sglang:prompt_tokens_total"], 1000.0)
        self.assertEqual(snap.prefill.values["sglang:cached_tokens_total"], 400.0)
        self.assertEqual(snap.decode.values["sglang:generation_tokens_total"], 50.0)

    def test_exposition_escapes_label_values(self):
        sys.path.insert(0, HERE)
        import live_metrics
        import prom_metrics

        text = prom_metrics.exposition(
            [{"metric": {"__name__": "smg_http_responses_total", "status_code": '5"0\\0'}, "value": [0, "3"]}]
        )
        self.assertEqual(live_metrics.parse(text).values["smg_http_responses_total"], 3.0)

    def test_suite_writes_one_report_and_passes(self):
        out = self.out("suite")
        r = tool("cluster_stress.py", "--url", self.url, "--phases", "needles,overload,soak,fault",
                 "--out-dir", out, "--soak-minutes", "0.05", "--sessions", "4", "--unsupported-params", "min_p",
                 "--needles-args", "--sizes 2000 --seeds 1", "--overload-args", "--levels 8 --step-s 1 --burst 10",
                 timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        self.assertEqual([x["status"] for x in rep["results"]], ["pass", "pass", "pass", "skipped"])
        with open(os.path.join(out, "report.md")) as f:
            md = f.read()
        self.assertIn("| needles | PASS |", md)
        for name in os.listdir(out):
            with open(os.path.join(out, name)) as f:
                self.assertNotIn(KEY, f.read(), name)


if __name__ == "__main__":
    unittest.main()
