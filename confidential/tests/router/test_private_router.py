"""Check the private router log writer with real child processes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / "confidential/router/private_router.py"
spec = importlib.util.spec_from_file_location("private_router", PROGRAM)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CANARY = "PRIVATE_ROUTER_TEST_VALUE_094378"
# The node log agent enforces this contract on the router container's stdout.
CONTRACT_KEYS = {"timestamp", "source_revision", "event", "role", "status", "code"}
CONTRACT_EVENTS = module.EVENTS | {"router_config_rejected", "router_exit", "router_failed", "router_started"}
REVISION = "0123456789abcdef0123456789abcdef01234567"


def check_contract(test, rows):
    for row in rows:
        test.assertLessEqual(set(row), CONTRACT_KEYS)
        test.assertIn(row["event"], CONTRACT_EVENTS)
        test.assertIn(row["role"], {"router", "prefill", "decode"})
        test.assertRegex(row["source_revision"], r"\A(?:[0-9a-f]{40}|unknown)\Z")
        # ISO-8601 UTC, as datetime.isoformat() writes it.
        test.assertRegex(row["timestamp"], r"\A\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00\Z")
        if "status" in row:
            test.assertIs(int, type(row["status"]))
            test.assertTrue(100 <= row["status"] <= 599)
        if "code" in row:
            test.assertIs(int, type(row["code"]))


class PrivateRouterTests(unittest.TestCase):
    def fake(self, directory, source, json_log=True):
        package = Path(directory) / "sglang_router"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "launch_router.py").write_text(source)
        # The launcher reads this file to decide whether to pass --json-log.
        flag = 'f"--{prefix}json-log"' if json_log else 'f"--{prefix}log-level"'
        (package / "router_args.py").write_text(f"FLAGS = [{flag}]\n")
        environment = {**os.environ, "PYTHONPATH": directory}
        environment.pop("CI_DIAGNOSTIC_SOURCE_REVISION", None)
        environment.pop("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", None)
        return environment

    def test_only_fixed_fields_reach_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self.fake(directory, f"""
import json,sys
canary={CANARY!r}
print(canary,flush=True)
print(json.dumps({{'ci_event':'worker_http_error','role':'prefill','status':400,
                  'message':canary,'span':{{'request_id':canary,'uri':canary}}}}),flush=True)
print(json.dumps({{'ci_event':canary,'role':'decode','status':400}}),flush=True)
print(json.dumps({{'ci_event':'worker_http_error','role':canary,'status':400}}),flush=True)
print(json.dumps({{'ci_event':'worker_http_error','role':'decode','status':canary}}),flush=True)
print(json.dumps({{'ci_event':'worker_http_error','role':{{'private':canary}},'status':400}}),flush=True)
print(canary*1000,flush=True)
print(json.dumps({{'ci_event':'worker_request_failed','role':'decode'}}),flush=True)
sys.exit(7)
""")
            result = subprocess.run([sys.executable, str(PROGRAM)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(7, result.returncode)
            self.assertEqual("", result.stderr)
            self.assertNotIn(CANARY, result.stdout)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(["router_started", "worker_http_error", "worker_request_failed", "router_exit"],
                             [row["event"] for row in rows])
            check_contract(self, rows)
            self.assertEqual({"event": "router_started", "role": "router"},
                             {k: v for k, v in rows[0].items() if k not in {"timestamp", "source_revision"}})
            self.assertEqual(7, rows[-1]["code"])

    def test_canary_in_message_url_or_request_id_never_reaches_stdout(self):
        # Each line is a reviewed event with the canary in one private field.
        # Only the fixed fields may pass; no line may carry the canary.
        private = [
            {"message": CANARY},
            {"msg": f"request failed {CANARY}"},
            {"url": f"http://{CANARY}.inference-workers:30000/generate"},
            {"worker_url": f"http://10.0.0.1:30000/{CANARY}"},
            {"uri": f"/v1/chat/completions?id={CANARY}"},
            {"request_id": CANARY},
            {"x-request-id": CANARY},
            {"span": {"request_id": CANARY, "uri": CANARY}},
            {"fields": {"message": CANARY, "request_id": CANARY}},
            {"target": CANARY, "code": 9, "timestamp": CANARY, "source_revision": CANARY},
        ]
        with tempfile.TemporaryDirectory() as directory:
            environment = self.fake(directory, f"""
import json,sys
for extra in {private!r}:
    print(json.dumps({{'ci_event':'worker_http_error','role':'decode','status':502,**extra}}),flush=True)
print('ERROR request {CANARY} failed url=http://{CANARY}/', flush=True)
print('{{"ci_event":"worker_http_error","message":"' + {CANARY!r} + '"', flush=True)
sys.stderr.write({CANARY!r} + '\\n')
sys.exit(0)
""")
            environment["CI_DIAGNOSTIC_SOURCE_REVISION"] = REVISION
            result = subprocess.run([sys.executable, str(PROGRAM)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode)
            self.assertEqual("", result.stderr)
            self.assertNotIn(CANARY, result.stdout)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            check_contract(self, rows)
            self.assertEqual(["router_started"] + ["worker_http_error"] * len(private),
                             [row["event"] for row in rows])
            for row in rows[1:]:
                self.assertEqual({"timestamp", "source_revision", "event", "role", "status"}, set(row))
                self.assertEqual(("decode", 502), (row["role"], row["status"]))
            self.assertTrue(all(row["source_revision"] == REVISION for row in rows))

    def test_json_log_is_passed_only_when_the_router_accepts_it(self):
        for json_log in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                argv = Path(directory) / "argv.json"
                environment = self.fake(directory, f"""
import json,sys
open({str(argv)!r},'w').write(json.dumps(sys.argv[1:]))
""", json_log=json_log)
                result = subprocess.run([sys.executable, str(PROGRAM), "--port=1"], env=environment,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(0, result.returncode)
                expected = ["--port=1", *(["--json-log"] if json_log else []), "--log-level", "warn"]
                given = json.loads(argv.read_text())
                # The launcher serves the metrics address; the router's exporter moves to loopback.
                self.assertEqual(expected + ["--prometheus-host", "127.0.0.1", "--prometheus-port"], given[:-1])
                self.assertTrue(1024 <= int(given[-1]) <= 65535)
                self.assertEqual(["router_started"],
                                 [json.loads(line)["event"] for line in result.stdout.splitlines()])

    def test_launcher_opens_only_its_metrics_listener(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self.fake(directory, "print('private child output', flush=True)\n")
            audit = Path(directory) / "network-events"
            audit.write_text("")
            (Path(directory) / "sitecustomize.py").write_text(
                "import sys\n"
                f"path = {str(audit)!r}\n"
                "def audit(event, args):\n"
                "    if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):\n"
                "        with open(path, 'a') as log: log.write(event + '\\n')\n"
                "sys.addaudithook(audit)\n")
            environment["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = "http://127.0.0.1:4318/v1/logs"
            result = subprocess.run([sys.executable, str(PROGRAM)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("", result.stderr)
            self.assertEqual(["router_started"],
                             [json.loads(line)["event"] for line in result.stdout.splitlines()])
            # It binds its metrics listener (and a loopback port probe); it connects nowhere
            # and looks up no name until a scrape asks for the router's metrics.
            self.assertEqual({"socket.bind"}, set(audit.read_text().split()))

    def test_emit_refuses_fields_outside_the_contract(self):
        for event, fields in (("router_exit", {"role": "router", "code": "1"}),
                              ("router_exit", {"role": "router", "message": "x"}),
                              ("router_started", {"role": "decode"}),
                              ("worker_http_error", {"role": "decode", "status": 99}),
                              ("worker_http_error", {"role": "decode", "url": "x"}),
                              ("diagnostic_delivery_failed", {"role": "router", "dropped": 1})):
            with self.assertRaises(ValueError, msg=(event, fields)):
                module.emit(event, **fields)

    def test_trace_and_file_flags_are_rejected_including_abbreviations(self):
        for flag in ("--log-dir", "--log-d", "--enable-trace", "--enable-tra",
                     "--otlp-traces-endpoint", "--otlp-traces-e"):
            result = subprocess.run([sys.executable, str(PROGRAM), flag, CANARY],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(2, result.returncode, flag)
            self.assertNotIn(CANARY, result.stdout + result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(["router_config_rejected"], [row["event"] for row in rows])
            check_contract(self, rows)

    def test_sigterm_reaches_child(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self.fake(directory, """
import json,signal,time
signal.signal(signal.SIGTERM,lambda *_:exit(0))
print(json.dumps({'ci_event':'worker_request_failed','role':'decode'}),flush=True)
while True:time.sleep(1)
""")
            process = subprocess.Popen([sys.executable, str(PROGRAM)], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual("router_started", json.loads(process.stdout.readline())["event"])
                self.assertEqual("worker_request_failed", json.loads(process.stdout.readline())["event"])
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(0, process.returncode)
                self.assertEqual("", stderr)
                self.assertEqual("", stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()



class MemoryMetricsTests(unittest.TestCase):
    """The launcher serves the router's metrics endpoint with two cgroup memory gauges added."""

    def test_the_gauges_read_the_cgroup_files(self):
        with tempfile.TemporaryDirectory() as directory:
            cgroup = Path(directory)
            (cgroup / "memory.current").write_text("123456\n")
            (cgroup / "memory.peak").write_text("987654321\n")
            text = module.memory_metrics(cgroup)
            self.assertIn("# TYPE ci_router_memory_current_bytes gauge\nci_router_memory_current_bytes 123456\n", text)
            self.assertIn("# TYPE ci_router_memory_peak_bytes gauge\nci_router_memory_peak_bytes 987654321\n", text)
            # A missing file or a value that is not a number leaves its gauge out.
            (cgroup / "memory.peak").unlink()
            (cgroup / "memory.current").write_text("max\n")
            self.assertEqual("", module.memory_metrics(cgroup))

    def test_the_metrics_address_is_the_routers(self):
        self.assertEqual(("0.0.0.0", 29000), module.metrics_address([]))
        self.assertEqual(("0.0.0.0", 29001), module.metrics_address(["--prometheus-host=0.0.0.0", "--prometheus-port=29001"]))
        self.assertEqual(("::", 7), module.metrics_address(["--prometheus-host", "::", "--prometheus-port", "8",
                                                            "--prometheus-port=7"]))
        self.assertIsNone(module.metrics_address(["--prometheus-port=x"]))

    def free_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def router(self, directory):
        """A fake router that serves its metrics on the --prometheus-port it gets last."""
        return PrivateRouterTests.fake(self, directory, """
import http.server,signal,sys,threading
args=sys.argv[1:]
port=[int(args[i+1]) for i,a in enumerate(args) if a=='--prometheus-port'][-1]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body=b'# TYPE smg_requests_total counter\\nsmg_requests_total 3'
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*a):pass
s=http.server.HTTPServer(('127.0.0.1',port),H)
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
s.serve_forever()
""")

    def test_the_endpoint_serves_the_routers_metrics_and_the_gauges(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self.router(directory)
            port = self.free_port()
            process = subprocess.Popen([sys.executable, str(PROGRAM), "--prometheus-host=127.0.0.1",
                                        f"--prometheus-port={port}"], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual("router_started", json.loads(process.stdout.readline())["event"])
                body = ""
                for _ in range(50):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                            body = response.read().decode()
                        if "smg_requests_total" in body:
                            break
                    except OSError:
                        pass
                    time.sleep(0.1)
                self.assertIn("smg_requests_total 3\n", body)
                gauges = [line for line in body.splitlines() if line.startswith("ci_router_memory")]
                # The gauges of this process's cgroup, when it has the files (in a container).
                expected = module.memory_metrics()
                names = lambda lines: [line.split()[0] for line in lines]  # noqa: E731
                self.assertEqual(names(line for line in expected.splitlines() if not line.startswith("#")),
                                 names(gauges))
                for line in gauges:
                    self.assertRegex(line, r"\Aci_router_memory_(current|peak)_bytes [0-9]+\Z")
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/other", timeout=2)
                process.send_signal(signal.SIGTERM)
                _, stderr = process.communicate(timeout=10)
                self.assertEqual("", stderr)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_a_taken_address_leaves_the_router_its_own_endpoint(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            argv = Path(directory) / "argv.json"
            environment = PrivateRouterTests.fake(self, directory, f"""
import json,sys
open({str(argv)!r},'w').write(json.dumps(sys.argv[1:]))
""")
            result = subprocess.run([sys.executable, str(PROGRAM), "--prometheus-host=127.0.0.1",
                                     f"--prometheus-port={port}"], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(["--prometheus-host=127.0.0.1", f"--prometheus-port={port}", "--json-log",
                              "--log-level", "warn"], json.loads(argv.read_text()))

if __name__ == "__main__":
    unittest.main()
