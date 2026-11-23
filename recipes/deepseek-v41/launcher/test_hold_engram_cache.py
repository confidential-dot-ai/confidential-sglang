"""hold_engram_cache.py: locked mode, and the populated fallback when mlock is refused."""

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HOLDER = Path(__file__).with_name("hold_engram_cache.py")


def checkpoint(root: Path) -> None:
    (root / "a.safetensors").write_bytes(os.urandom(3 << 20))
    (root / "b.safetensors").write_bytes(os.urandom((2 << 20) + 123))
    (root / "c.safetensors").write_bytes(b"x" * 4096)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.engram.embed.weight": "a.safetensors",
        "model.layers.1.engram.embed.weight": "b.safetensors",
        "model.layers.0.mlp.weight": "c.safetensors",
    }}))


def run(model: Path, *flags, memlock: int | None = None):
    ready = model / "engram.ready"
    limit = None if memlock is None else (lambda: resource.setrlimit(resource.RLIMIT_MEMLOCK, (memlock, memlock)))
    proc = subprocess.Popen([sys.executable, str(HOLDER), "--model", str(model), "--ready-file", str(ready), "--reserve-gib", "0", *flags],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=limit)
    deadline = time.monotonic() + 60
    while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    status = Path(f"/proc/{proc.pid}/status").read_text() if proc.poll() is None else ""
    proc.send_signal(signal.SIGTERM) if proc.poll() is None else None
    out = proc.communicate(timeout=30)[0]
    return ready.read_text() if ready.exists() else None, out, status, proc


def locked_kib(status: str) -> int:
    return next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmLck:"))


class TestHoldEngramCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        checkpoint(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lock_off_populates_unlocked_and_writes_the_pid(self):
        ready, out, status, proc = run(self.model, "--lock", "off")
        self.assertEqual(ready.strip(), str(proc.pid))
        self.assertIn("READY: mode=populated", out)
        self.assertEqual(locked_kib(status), 0)
        self.assertNotIn("c.safetensors", out)  # only Engram shards

    @unittest.skipIf(os.geteuid() == 0, "root has CAP_IPC_LOCK: mlock is never refused")
    def test_auto_falls_back_when_mlock_is_refused(self):
        # 64 KiB: the first 3 MiB shard cannot be locked, as under C8s.
        ready, out, status, proc = run(self.model, memlock=64 << 10)
        self.assertEqual(ready.strip(), str(proc.pid))
        self.assertIn("mode=populated: mlock refused", out)
        self.assertIn("READY: mode=populated", out)
        self.assertEqual(locked_kib(status), 0)  # nothing stays part-locked

    @unittest.skipIf(os.geteuid() == 0, "root has CAP_IPC_LOCK: mlock is never refused")
    def test_required_fails_when_mlock_is_refused(self):
        ready, out, _, proc = run(self.model, "--lock", "required", memlock=64 << 10)
        self.assertIsNone(ready)
        self.assertNotEqual(proc.returncode, 0)

    @unittest.skipIf(0 <= resource.getrlimit(resource.RLIMIT_MEMLOCK)[1] < 8 << 20 and os.geteuid() != 0,
                     "the hard RLIMIT_MEMLOCK is below the test shards")
    def test_auto_locks_with_enough_memlock(self):
        hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
        ready, out, status, proc = run(self.model, memlock=hard)
        self.assertEqual(ready.strip(), str(proc.pid))
        self.assertIn("READY: mode=locked", out)
        self.assertGreaterEqual(locked_kib(status) * 1024, (3 << 20) + (2 << 20))


if __name__ == "__main__":
    unittest.main()
