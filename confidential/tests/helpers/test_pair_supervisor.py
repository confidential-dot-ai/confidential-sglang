"""Run with the GPU acceptance checks on the isolated test CVM.

These tests need no model. They exercise the same supervisor with fake device
identities and real child processes. GPU correctness is a separate gate.
"""

import contextlib
import importlib.util
import http.client
import io
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "confidential/helpers"))
spec = importlib.util.spec_from_file_location("pair_supervisor", ROOT / "confidential/helpers/pair_supervisor.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def unused_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PrivacyTests(unittest.TestCase):
    def test_status_handler_exception_cannot_export_client_data(self):
        canary = "PRIVATE_PROMPT_RESPONSE_API_CREDENTIAL"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                raise RuntimeError(self.headers["Authorization"] + self.path)

            def log_message(self, *_):
                pass

        def run(server_type):
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                server = server_type(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                try:
                    connection.request("GET", "/" + canary, headers={"Authorization": canary})
                    with self.assertRaises(http.client.RemoteDisconnected):
                        connection.getresponse()
                finally:
                    connection.close()
                    server.shutdown()
                    server.server_close()
                    thread.join(2)
            return output.getvalue(), errors.getvalue()

        # A positive control proves this fixture reaches the unsafe base path.
        self.assertIn(canary, run(ThreadingHTTPServer)[1])
        output, errors = run(pair.PrivateStatusServer)
        self.assertNotIn(canary, output + errors)
        self.assertEqual("", errors)
        record = json.loads(output)
        self.assertEqual({"timestamp_ns", "run_id", "source_revision", "event", "role"}, set(record))
        self.assertRegex(record["run_id"], r"^[0-9a-f]{32}$")
        self.assertEqual("status_request_failed", record["event"])

    def test_rejects_sensitive_or_unreviewed_fields_whole(self):
        fake = "FAKE-API-KEY-and-FAKE-CUSTOMER-PROMPT"
        for field in ("message", "prompt", "output", "tensor", "api_key", "customer", "locals", "rid", "run_id", "headers", "path", "exception"):
            self.assertIsNone(pair.reviewed_event({"event": "rank_failed", "role": "decode", field: fake}))
        for value in (fake, True, None, [], {}, float("nan"), float("inf")):
            self.assertIsNone(pair.reviewed_event({"event": "cuda_error", "role": "prefill", "code": value}))

    def test_only_fixed_codes_and_numeric_metadata_are_emitted(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pair.emit("nccl_error", "decode", rank=1, code=7, count=2)
        record = json.loads(output.getvalue())
        self.assertEqual({"timestamp_ns", "run_id", "source_revision", "event", "role", "rank", "code", "count"}, set(record))
        self.assertEqual("nccl_error", record["event"])

    def test_refuses_raw_artifact_outputs(self):
        for flag in ("--log-requests", "--crash-dump-folder", "--debug-tensor-dump-output-folder", "--enable-trace",
                     "--log-req", "--crash-dump-f", "--debug-tensor-dump-o", "--enable-tr", "--enable_trace"):
            cfg = config()
            cfg["workers"][0]["command"].append(flag)
            with self.assertRaises(ValueError):
                pair.validate_config(cfg)
        for name in ("TORCH_NCCL_DUMP_ON_TIMEOUT", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
                     "TORCH_NCCL_DEBUG_INFO_PIPE_FILE", "NCCL_DEBUG_FILE", "TORCH_COMPILE_DEBUG",
                     "CUDA_ENABLE_COREDUMP", "SGLANG_LOGGING_CONFIG_PATH", "TORCH_LOGS",
                     "TORCH_LOGS_OUT", "TORCHINDUCTOR_SAVE_ARGS", "TORCHDYNAMO_REPRO_AFTER"):
            cfg = config()
            cfg["workers"][0]["env"][name] = "PRIVATE_PATH_OR_VALUE"
            with self.assertRaises(ValueError):
                pair.validate_config(cfg)


def config():
    return {"modelGate": [sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(120)", "/run/dsv41/engram.ready"], "workers": [{"role": role, "gpus": 1, "ranks": 1, "port": unused_port(),
                         "command": [sys.executable, "-c", "pass"], "env": {}}
                        for role in ("prefill", "decode")],
            "statusPort": unused_port(), "progressTimeoutSeconds": 1,
            "startupTimeoutSeconds": 5, "shutdownTimeoutSeconds": 1}


class RankHealthTests(unittest.TestCase):
    def test_failure_details_export_only_reason_and_ages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decode-0.json"
            now = 1000.0  # A new CI VM may have less than 120 seconds of uptime.
            path.write_text(json.dumps({"pid": os.getpid(), "state": "busy", "updated": now,
                                        "completed": 0, "pending_since": now - 120}))
            self.assertEqual({"code": 3, "age_ms": 0, "gpu_pending_ms": 120000},
                             pair.rank_failure_details(path, now, 90))
            path.write_text(json.dumps({"prompt": "PRIVATE_CUSTOMER_TEXT"}))
            self.assertEqual({"code": 1}, pair.rank_failure_details(path, now, 90))

    def test_restart_records_stay_bounded_without_following_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "pair"
            outside = Path(directory) / "outside"
            outside.mkdir()
            marker = outside / "keep"
            marker.write_text("unrelated")
            for index in range(20):
                current = pair.new_generation(base)
                (current / "decode-0.json").write_text(json.dumps({"pid": index}))
                self.assertLessEqual(len([p for p in base.iterdir() if p.is_dir()]), 2)
            (base / "generation-link").symlink_to(outside, target_is_directory=True)
            pair.new_generation(base)
            self.assertEqual("unrelated", marker.read_text())
            self.assertTrue((base / "generation-link").is_symlink())

    def test_dead_and_stale_rank_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decode-0.json"
            now = time.monotonic()
            record = {"pid": os.getpid(), "state": "idle", "updated": now, "completed": now, "pending_since": 0}
            path.write_text(json.dumps(record))
            self.assertEqual("ready", pair.rank_status(path, now, 1, None)[0])
            self.assertEqual("failed", pair.rank_status(path, now + 2, 1, os.getpid())[0])
            record.update(state="busy", pending_since=now - 2, updated=now)
            path.write_text(json.dumps(record))
            self.assertEqual("failed", pair.rank_status(path, now, 1, None)[0])

    def test_idle_is_healthy_without_completed_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefill-0.json"
            now = time.monotonic()
            path.write_text(json.dumps({"pid": os.getpid(), "state": "idle", "updated": now,
                                       "completed": 0, "pending_since": 0}))
            self.assertEqual("ready", pair.rank_status(path, now, 1, None)[0])

    def test_rank_startup_uses_startup_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefill-0.json"
            now = 1000.0
            path.write_text(json.dumps({"pid": os.getpid(), "state": "starting", "updated": now - 100,
                                       "completed": 0, "pending_since": 0}))
            self.assertEqual("starting", pair.rank_status(path, now, 1, None)[0])

    def test_changed_pid_and_unknown_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decode-0.json"
            now = time.monotonic()
            record = {"pid": os.getpid(), "state": "idle", "updated": now, "completed": now, "pending_since": 0}
            path.write_text(json.dumps(record))
            self.assertEqual("failed", pair.rank_status(path, now, 1, os.getpid() + 1)[0])
            record["prompt"] = "FAKE-SECRET"
            path.write_text(json.dumps(record))
            self.assertEqual("failed", pair.rank_status(path, now, 1, None)[0])


class DeviceAssignmentTests(unittest.TestCase):
    def test_requires_exact_distinct_devices(self):
        for devices in ([], ["GPU-a"], ["GPU-a", "GPU-a"], ["GPU-a", "GPU-b", "GPU-c"]):
            with self.assertRaises(ValueError):
                pair.Supervisor(config(), Path("/unused"), devices)


class ProcessAcceptanceTests(unittest.TestCase):
    def test_failure_in_one_pair_does_not_stop_the_other_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisors, controls, monitors = [], [], []
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    for index in range(2):
                        cfg = config()
                        control = Path(directory) / f"decode-{index}"
                        controls.append(control)
                        for w in cfg["workers"]:
                            marker = control if w["role"] == "decode" else Path(directory) / f"prefill-{index}"
                            w["command"] = [sys.executable, str(Path(__file__).with_name("pair_fake_worker.py")),
                                            "--port", str(w["port"])]
                            w["env"] = {"PAIR_TEST_CONTROL": str(marker)}
                        supervisor = pair.Supervisor(cfg, Path(directory) / f"run-{index}",
                                                     [f"GPU-{index}-a", f"GPU-{index}-b"])
                        supervisors.append(supervisor)
                        supervisor.start()
                        monitor = threading.Thread(target=supervisor.monitor, daemon=True)
                        monitors.append(monitor)
                        monitor.start()
                    controls[1].write_text("stall")
                    monitors[1].join(timeout=5)
                    self.assertFalse(monitors[1].is_alive())
                    supervisors[1].shutdown()
                    self.assertEqual("ready", supervisors[0].snapshot[0])
                    for _ in range(10):
                        for w in supervisors[0].config["workers"]:
                            conn = http.client.HTTPConnection("127.0.0.1", w["port"], timeout=1)
                            conn.request("GET", "/v1/models")
                            self.assertEqual(200, conn.getresponse().status)
                            conn.close()
                finally:
                    for supervisor in supervisors:
                        supervisor.stop.set()
                    for monitor in monitors:
                        monitor.join(timeout=3)
                    for supervisor in supervisors:
                        supervisor.shutdown()

    def exercise_failure(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            controls = {}
            for worker in cfg["workers"]:
                role = worker["role"]
                controls[role] = Path(directory) / role
                worker["command"] = [sys.executable, str(Path(__file__).with_name("pair_fake_worker.py")),
                                     "--port", str(worker["port"])]
                worker["env"] = {"PAIR_TEST_CONTROL": str(controls[role])}
            supervisor = pair.Supervisor(cfg, Path(directory) / "run", ["GPU-fake-a", "GPU-fake-b"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    supervisor.start()
                    result = []
                    monitor = threading.Thread(target=lambda: result.append(supervisor.monitor()), daemon=True)
                    monitor.start()
                    deadline = time.monotonic() + 4
                    while supervisor.snapshot[0] != "ready" and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual("ready", supervisor.snapshot[0])
                    # The status handler stays fast and never contacts a worker.
                    for _ in range(10):
                        started = time.monotonic()
                        connection = http.client.HTTPConnection("127.0.0.1", cfg["statusPort"], timeout=1)
                        connection.request("GET", "/livez")
                        self.assertEqual(200, connection.getresponse().status)
                        connection.close()
                        self.assertLess(time.monotonic() - started, 0.25)
                    controls["decode"].write_text(failure)
                    monitor.join(timeout=5)
                    self.assertFalse(monitor.is_alive())
                    self.assertEqual([1], result)
                    self.assertEqual("failed", supervisor.snapshot[0])
                finally:
                    supervisor.shutdown()
            raw = output.getvalue()
            for canary in ("FAKE-KEY-DO-NOT-LOG", "FAKE-PROMPT-DO-NOT-LOG"):
                self.assertNotIn(canary, raw)
            records = [json.loads(line) for line in raw.splitlines()]
            self.assertTrue(any(record["event"] == "cuda_error" for record in records))
            for record in records:
                    self.assertRegex(record["run_id"], r"^[0-9a-f]{32}$")
                    self.assertIsNotNone(pair.reviewed_event({k: v for k, v in record.items() if k not in {"timestamp_ns", "run_id", "source_revision"}}))
            self.assertTrue(all(child.poll() is not None for child in supervisor.children.values()))

    def test_parallel_start_is_an_optional_boolean(self):
        cfg = config()
        pair.validate_config(cfg)
        cfg["parallelStart"] = True
        pair.validate_config(cfg)
        for bad in ("yes", 1, None):
            cfg["parallelStart"] = bad
            with self.assertRaises(ValueError):
                pair.validate_config(cfg)
        cfg = config()
        cfg["parallelStartup"] = True
        with self.assertRaises(ValueError):
            pair.validate_config(cfg)

    def test_parallel_start_runs_decode_while_prefill_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            cfg["parallelStart"] = True
            controls = {}
            for worker in cfg["workers"]:
                role = worker["role"]
                controls[role] = Path(directory) / role
                worker["command"] = [sys.executable, str(Path(__file__).with_name("pair_fake_worker.py")),
                                     "--port", str(worker["port"])]
                worker["env"] = {"PAIR_TEST_CONTROL": str(controls[role])}
            # Prefill writes no progress record yet: it is still loading.
            controls["prefill"].write_text("stall")
            supervisor = pair.Supervisor(cfg, Path(directory) / "run", ["GPU-fake-a", "GPU-fake-b"])
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    supervisor.start()
                    self.assertEqual({"model", "prefill", "decode"}, set(supervisor.children))
                    result = []
                    monitor = threading.Thread(target=lambda: result.append(supervisor.monitor()), daemon=True)
                    monitor.start()
                    deadline = time.monotonic() + 1.5
                    while time.monotonic() < deadline:
                        self.assertEqual("starting", supervisor.snapshot[0])
                        time.sleep(0.05)
                    controls["prefill"].write_text("")
                    deadline = time.monotonic() + 4
                    while supervisor.snapshot[0] != "ready" and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual("ready", supervisor.snapshot[0])
                    self.assertTrue(all(child.poll() is None for child in supervisor.children.values()))
                finally:
                    supervisor.shutdown()

    def test_http_alive_gpu_progress_stalled_fails_and_stops_pair(self):
        self.exercise_failure("stall")

    def test_clean_frontend_cannot_hide_exited_worker(self):
        self.exercise_failure("exit")



GATE_SCRIPT = r"""
import json, pathlib, sys, time
events = [
    {"event": "model_fetch_started", "host": "models.example.invalid", "files": 83, "size_bytes": 510299999999},
    {"event": "model_fetch_retry", "host": "models.example.invalid", "file": "model-00003-of-00046.safetensors",
     "attempt": 1, "error_class": "http_error", "http_status": 503},
    {"event": "model_fetch_complete", "host": "models.example.invalid", "downloaded": 83, "kept": 0,
     "size_bytes": 510299999999, "duration_ms": 900000},
    {"event": "model_verified", "files": 83, "size_bytes": 510299999999, "duration_ms": 120000},
    # Refused whole: a field outside the contract, a URL as a file, a role, an event the gate does not have.
    {"event": "model_fetch_failed", "error_class": "http_error", "url": "http://FAKE-PROMPT-DO-NOT-LOG"},
    {"event": "model_check_failed", "error_class": "missing", "file": "../FAKE-KEY-DO-NOT-LOG"},
    {"event": "model_verified", "role": "decode", "files": 1, "size_bytes": 1, "duration_ms": 1},
    {"event": "worker_ready", "load_seconds": 1},
]
for event in events:
    print(json.dumps(event), flush=True)
print("Traceback FAKE-PROMPT-DO-NOT-LOG", flush=True)
pathlib.Path(sys.argv[1]).write_text("ready")
time.sleep(120)
"""


class GateAndReadyEventTests(unittest.TestCase):
    def test_gate_events_pass_only_their_reviewed_fields(self):
        failed = {"event": "model_check_failed", "error_class": "digest_mismatch",
                  "file": "model-00001-of-00046.safetensors", "expected_size": 7, "got_size": 7,
                  "expected_sha256": "a" * 64, "got_sha256": "b" * 64}
        self.assertEqual({**failed, "role": "pair"}, pair.gate_event(failed))
        # Null fields are left out.
        self.assertEqual({"event": "model_check_failed", "role": "pair", "error_class": "timeout"},
                         pair.gate_event({"event": "model_check_failed", "error_class": "timeout", "file": None}))
        for bad in ({**failed, "file": "/etc/passwd"}, {**failed, "file": "a/../b"}, {**failed, "got_sha256": "x"},
                    {**failed, "error_class": "Traceback"}, {**failed, "message": "x"}, {**failed, "role": "decode"},
                    {"event": "rank_failed"}, {"event": "worker_ready", "load_seconds": 1},
                    {"event": "model_fetch_started", "host": "http://models.example", "files": 1, "size_bytes": 1}):
            with self.subTest(bad=bad):
                self.assertIsNone(pair.gate_event(bad))

    def test_a_malformed_gate_record_does_not_stop_forwarding(self):
        lines = [b'{"event":[]}\n', b'{"event":{"a":1}}\n', b'[1,2]\n', b'"x"\n',
                 b'{"event":"model_verified","files":1,"size_bytes":2,"duration_ms":3}\n']
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pair.forward_gate_events(io.BytesIO(b"".join(lines)))
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(["model_verified"], [record["event"] for record in records])

    def test_jit_compile_carries_only_a_kernel_name_and_a_time(self):
        record = {"event": "jit_compile", "role": "decode", "rank": 1, "kernel": "_fwd_kernel", "duration_ms": 9}
        self.assertEqual(record, pair.reviewed_event(record))
        for kernel in ("/tmp/x.cubin", "a b", "x" * 65, 3):
            self.assertIsNone(pair.reviewed_event({**record, "kernel": kernel}))

    def test_a_pair_reports_its_gate_its_readiness_and_late_compiles(self):
        self.run_pair(parallel=False)

    def test_a_parallel_start_reports_readiness_and_late_compiles_too(self):
        self.run_pair(parallel=True)

    def run_pair(self, parallel):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            if parallel:
                cfg["parallelStart"] = True
            cfg["modelGate"] = [sys.executable, "-c", GATE_SCRIPT, "/run/dsv41/engram.ready"]
            controls = {}
            for worker in cfg["workers"]:
                role = worker["role"]
                controls[role] = Path(directory) / role
                worker["command"] = [sys.executable, str(Path(__file__).with_name("pair_fake_worker.py")),
                                     "--port", str(worker["port"])]
                worker["env"] = {"PAIR_TEST_CONTROL": str(controls[role])}
            supervisor = pair.Supervisor(cfg, Path(directory) / "run", ["GPU-fake-a", "GPU-fake-b"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    supervisor.start()
                    monitor = threading.Thread(target=supervisor.monitor, daemon=True)
                    monitor.start()
                    deadline = time.monotonic() + 4
                    while supervisor.ready_roles != {"prefill", "decode"} and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual({"prefill", "decode"}, supervisor.ready_roles)
                    controls["decode"].write_text("compile")
                    deadline = time.monotonic() + 4
                    while "_late_kernel" not in output.getvalue() and time.monotonic() < deadline:
                        time.sleep(0.05)
                finally:
                    supervisor.shutdown()
            raw = output.getvalue()
            for canary in ("FAKE-KEY-DO-NOT-LOG", "FAKE-PROMPT-DO-NOT-LOG", "Traceback", "_startup_kernel"):
                self.assertNotIn(canary, raw)
            records = [json.loads(line) for line in raw.splitlines()]
            strip = lambda record: {k: v for k, v in record.items()  # noqa: E731
                                    if k not in {"timestamp_ns", "run_id", "source_revision"}}
            gate = [strip(record) for record in records if record["event"].startswith("model_")]
            self.assertEqual([
                {"event": "model_fetch_started", "role": "pair", "host": "models.example.invalid", "files": 83,
                 "size_bytes": 510299999999},
                {"event": "model_fetch_retry", "role": "pair", "host": "models.example.invalid",
                 "file": "model-00003-of-00046.safetensors", "attempt": 1, "error_class": "http_error",
                 "http_status": 503},
                {"event": "model_fetch_complete", "role": "pair", "host": "models.example.invalid", "downloaded": 83,
                 "kept": 0, "size_bytes": 510299999999, "duration_ms": 900000},
                {"event": "model_verified", "role": "pair", "files": 83, "size_bytes": 510299999999,
                 "duration_ms": 120000},
            ], gate)
            ready = [strip(record) for record in records if record["event"] == "worker_ready"]
            self.assertEqual(["decode", "prefill"] if parallel else ["prefill", "decode"],
                             sorted(record["role"] for record in ready) if parallel else
                             [record["role"] for record in ready])
            for record in ready:
                self.assertEqual({"event", "role", "gpus", "load_seconds"}, set(record))
                self.assertEqual(1, record["gpus"])
            self.assertEqual([{"event": "jit_compile", "role": "decode", "rank": 0, "kernel": "_late_kernel",
                               "duration_ms": 1234}],
                             [strip(record) for record in records if record["event"] == "jit_compile"])
            for record in records:
                self.assertIsNotNone(pair.reviewed_event(strip(record)), record)


if __name__ == "__main__":
    unittest.main()
