"""Test progress rules without importing Torch or loading a model."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("pair_progress", ROOT / "python/sglang/srt/utils/pair_progress.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


class Event:
    done = False

    def query(self):
        return self.done


class ProgressTests(unittest.TestCase):
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
            p.poll()
            p.publish(force=True)
            record = json.loads(p.path.read_text())
            self.assertEqual("idle", record["state"])
            self.assertEqual(now[0], record["completed"])

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


if __name__ == "__main__":
    unittest.main()
