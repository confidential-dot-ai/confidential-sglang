"""Test progress rules without importing Torch or loading a model."""

from array import array
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("pair_progress", ROOT / "python/sglang/srt/utils/pair_progress.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


class Event:
    done = False
    queries = 0

    def query(self):
        self.queries += 1
        return self.done


class ProgressTests(unittest.TestCase):
    def test_native_token_arrays_export_lengths_without_values_or_text(self):
        request=SimpleNamespace(origin_input_ids=array('q',[987654321,987654322]),
                                output_ids=array('q',[987654323]),
                                origin_input_text='PRIVATE_PROMPT',output_text='PRIVATE_RESPONSE')
        scheduler=SimpleNamespace(cur_batch=SimpleNamespace(reqs=[request]))
        reporter=SimpleNamespace(scheduler=scheduler,stages=[4],last_step_ms=12,depth=0)
        with patch.object(progress,'_reporter',reporter),patch.object(progress,'_send_diagnostic') as send:
            progress._scheduler_snapshot('/unused','decode',0)
        record=send.call_args.args[1]
        self.assertEqual(2,record['input_tokens'])
        self.assertEqual(1,record['output_tokens'])
        raw=json.dumps(record)
        for private in ('987654321','987654322','987654323','PRIVATE_PROMPT','PRIVATE_RESPONSE'):
            self.assertNotIn(private,raw)

    def test_continuous_batches_bound_file_writes_and_preserve_busy_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            original = Path.write_text
            with patch.object(Path, "write_text", autospec=True, side_effect=original) as writes:
                p = progress.Progress(directory, "decode", 0, event_factory=Event, clock=lambda: now[0])
                p.ready()
                for index in range(1000):
                    with p.stage("run_batch"):
                        if index == 0:
                            self.assertEqual("busy", json.loads(p.path.read_text())["state"])
                        now[0] += 0.001
                self.assertLessEqual(writes.call_count, 6)
            p.publish(force=True)
            record = json.loads(p.path.read_text())
            self.assertEqual("busy", record["state"])
            self.assertEqual(100.0, record["pending_since"])

    def test_last_batch_stays_busy_until_idle_poll_observes_its_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            event = Event()
            calls = []
            def factory():
                calls.append(now[0])
                return event
            p = progress.Progress(directory, "decode", 0, event_factory=factory, clock=lambda: now[0])
            p.ready()
            # A fast final batch ends before the marker cadence. It must still
            # be visible as pending GPU work while the CPU runs idle stages.
            with p.stage("run_batch"):
                now[0] += 0.001
            p.publish(force=True)
            self.assertEqual([], calls)
            self.assertEqual(100.0, json.loads(p.path.read_text())["pending_since"])
            for _ in range(100):
                now[0] += 0.001
                with p.stage("recv_requests"):
                    pass
            self.assertEqual(1, len(calls))
            p.publish(force=True)
            self.assertEqual(100.0, json.loads(p.path.read_text())["pending_since"])
            event.done = True
            now[0] += 0.2
            with p.stage("recv_requests"):
                pass
            p.publish(force=True)
            self.assertEqual("idle", json.loads(p.path.read_text())["state"])

    def test_marker_creation_is_bounded_and_keeps_oldest_unfinished_work(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            calls = []
            def factory():
                calls.append(now[0])
                return Event()
            p = progress.Progress(directory, "decode", 0, event_factory=factory, clock=lambda: now[0])
            p.ready()
            for _ in range(1000):
                with p.stage("run_batch"):
                    now[0] += 0.001
            self.assertLessEqual(len(calls), 10)
            p.publish(force=True)
            self.assertEqual(100.0, json.loads(p.path.read_text())["pending_since"])

    def test_scheduler_snapshot_uses_lengths_without_request_content(self):
        private = "PRIVATE_PROMPT_RESPONSE_CREDENTIAL_CLIENT_ID"
        request = SimpleNamespace(origin_input_ids=[private] * 3,
                                  output_ids=[private] * 2,
                                  rid=private, text=private, api_key=private)
        reporter = SimpleNamespace(
            scheduler=SimpleNamespace(cur_batch=SimpleNamespace(reqs=[request])),
            stages=[4], last_step_ms=12, depth=1, stage_started=100,
            clock=lambda: 101)
        with patch.object(progress, "_reporter", reporter), patch.object(progress, "_send_diagnostic") as send:
            progress._scheduler_snapshot("unused", "decode", 1)
        record = send.call_args.args[1]
        self.assertEqual({"event": "scheduler_snapshot", "role": "decode", "rank": 1,
                          "stage_code": 4, "last_step_ms": 12, "step_age_ms": 1000,
                          "batch_size": 1, "input_tokens": 3, "output_tokens": 2}, record)
        self.assertNotIn(private, json.dumps(record))

    def test_gpu_query_failure_marks_failed_without_exception_text(self):
        class FailedEvent:
            def query(self):
                raise RuntimeError("PRIVATE_NATIVE_ERROR")
        with tempfile.TemporaryDirectory() as directory:
            p = progress.Progress(directory, "decode", 0, clock=lambda: 100)
            p.ready()
            p.pending.append((99, FailedEvent()))
            with patch.object(progress, "error_event") as emit:
                with self.assertRaises(RuntimeError):
                    p.poll()
            emit.assert_called_once_with("gpu_event_query_failed", rank=0)
            raw = p.path.read_text()
            self.assertNotIn("PRIVATE_NATIVE_ERROR", raw)
            self.assertEqual("failed", json.loads(raw)["state"])

    def test_busy_gpu_stays_busy_until_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            event = Event()
            p = progress.Progress(directory, "decode", 1, event_factory=lambda: event, clock=lambda: now[0])
            p.ready()
            with p.stage("run_batch"):
                now[0] += 1
            p.publish(force=True)
            record = json.loads(p.path.read_text())
            self.assertEqual("busy", record["state"])
            self.assertEqual(100.0, record["pending_since"])
            self.assertEqual(0, record["completed"])
            # A CPU heartbeat is not proof the GPU completed its work.
            now[0] += 10
            with p.stage("recv_requests"):
                pass
            p.publish(force=True)
            self.assertEqual(100.0, json.loads(p.path.read_text())["pending_since"])
            event.done = True
            now[0] += 0.2
            p.poll()
            p.publish(force=True)
            record = json.loads(p.path.read_text())
            self.assertEqual("idle", record["state"])
            self.assertEqual(now[0], record["completed"])

    def test_query_rate_is_bounded_without_hiding_stalled_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            event = Event()
            p = progress.Progress(directory, "decode", 0, event_factory=lambda: event, clock=lambda: now[0])
            p.ready()
            with p.stage("run_batch"):
                now[0] += 1
            initial = event.queries
            for _ in range(1000):
                now[0] += 0.0001
                with p.stage("recv_requests"):
                    pass
            self.assertLessEqual(event.queries - initial, 2)
            p.publish(force=True)
            record = json.loads(p.path.read_text())
            self.assertEqual("busy", record["state"])
            self.assertEqual(100.0, record["pending_since"])
            self.assertEqual(0, record["completed"])

    def test_cpu_stall_is_visible_before_gpu_event_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            p = progress.Progress(directory, "prefill", 0, event_factory=lambda: Event(), clock=lambda: 100.0)
            p.ready()
            with p.stage("run_batch"):
                record = json.loads(p.path.read_text())
                self.assertEqual("busy", record["state"])
                self.assertEqual(100.0, record["pending_since"])

    def test_errors_publish_state_without_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            p = progress.Progress(directory, "decode", 0, clock=lambda: 100.0)
            p.ready()
            with self.assertRaises(RuntimeError):
                with p.stage("process_batch_result"):
                    raise RuntimeError("FAKE-CUSTOMER-PROMPT")
            raw = p.path.read_text()
            self.assertNotIn("FAKE-CUSTOMER-PROMPT", raw)
            self.assertEqual("failed", json.loads(raw)["state"])



class CompileEventTests(unittest.TestCase):
    """A Triton compile reaches the pair supervisor as a kernel name and a time, nothing else."""

    def receive(self, call):
        import os
        import socket

        with tempfile.TemporaryDirectory() as directory:
            receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            receiver.bind(str(Path(directory) / "events.sock"))
            receiver.settimeout(0.5)
            environment = {"SGLANG_PAIR_ROLE": "decode", "SGLANG_PAIR_RUN_DIR": directory}
            try:
                with patch.dict(os.environ, environment), patch.object(progress, "_reporter",
                                                                        SimpleNamespace(rank=1)):
                    call()
                try:
                    return json.loads(receiver.recv(4096))
                except socket.timeout:
                    return None
            finally:
                receiver.close()

    def test_a_compile_sends_its_kernel_and_duration(self):
        self.assertEqual({"event": "jit_compile", "role": "decode", "rank": 1, "kernel": "_fwd_kernel",
                          "duration_ms": 1234},
                         self.receive(lambda: progress.compile_event("_fwd_kernel", 1234)))

    def test_a_name_that_is_not_an_identifier_is_not_sent(self):
        for kernel in ("/tmp/PRIVATE.cubin", "a b", "x" * 65, None):
            self.assertIsNone(self.receive(lambda: progress.compile_event(kernel, 1)))
        self.assertIsNone(self.receive(lambda: progress.compile_event("_k", -1)))

    def test_outside_a_pair_nothing_is_sent(self):
        import os

        with patch.dict(os.environ, {}, clear=True), patch.object(progress.socket, "socket",
                                                                   side_effect=AssertionError("no socket")):
            progress.compile_event("_k", 1)

    def test_the_triton_listener_reports_only_real_compiles(self):
        import sys
        import types

        calls = []
        fake_progress = types.SimpleNamespace(compile_event=lambda kernel, ms: calls.append((kernel, ms)))
        stubs = {"torch": types.ModuleType("torch"), "sglang": types.ModuleType("sglang"),
                 "sglang.srt": types.ModuleType("sglang.srt"),
                 "sglang.srt.environ": types.SimpleNamespace(envs=None),
                 "sglang.srt.utils": types.SimpleNamespace(pair_progress=fake_progress),
                 "sglang.srt.utils.device": types.SimpleNamespace(get_available_gpu_memory=None),
                 "sglang.srt.utils.pair_progress": fake_progress}
        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "triton_load_watch", ROOT / "python/sglang/srt/utils/triton_load_watch.py")
            watch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(watch)
            times = SimpleNamespace(total=2_500_000)
            src = SimpleNamespace(name="_from_src")
            watch._on_compilation(src=src, metadata={"name": "_kernel"}, metadata_group={}, times=times,
                                  cache_hit=False)
            watch._on_compilation(src=src, metadata={}, metadata_group={}, times=times, cache_hit=False)
            watch._on_compilation(src=src, metadata={"name": "_cached"}, metadata_group={}, times=times,
                                  cache_hit=True)
        self.assertEqual([("_kernel", 2500), ("_from_src", 2500)], calls)


if __name__ == "__main__":
    unittest.main()
