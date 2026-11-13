"""Exercise fatal recovery in real processes, without Torch or a model."""

from pathlib import Path
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[4]
CHILD = r'''
import importlib.util
import os
import signal
import sys
import time
import types

psutil = types.ModuleType("psutil")
psutil.Process = lambda: types.SimpleNamespace(parent=lambda: None)
sys.modules["psutil"] = psutil
for name, function in (
    ("cudacore_pyspy_dump_utils", "pyspy_dump_schedulers"),
    ("pair_progress", "error_event"),
):
    dependency = types.ModuleType("sglang.srt.utils." + name)
    setattr(dependency, function, lambda *args, **kwargs: None)
    sys.modules[dependency.__name__] = dependency
spec = importlib.util.spec_from_file_location("watchdog", sys.argv[1])
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)
arm = watchdog._arm_fatal_exit
watchdog._arm_fatal_exit = lambda: arm(0.25)
os.environ["SGLANG_PAIR_RUN_DIR"] = "/unused-test-directory"
signal.signal(signal.SIGQUIT, signal.SIG_IGN)
if sys.argv[2] == "blocked_abort":
    watchdog._abort_pynccl_communicators = lambda: time.sleep(60)
    watcher = watchdog.WatchdogRaw("scheduler", lambda: 0, lambda: True, 0.02)
else:
    failed = types.SimpleNamespace(is_alive=lambda: False, exitcode=7, pid=123)
    watcher = watchdog.SubprocessWatchdog([failed])
    watcher._check_processes()
time.sleep(60)
'''


class WatchdogRecoveryTests(unittest.TestCase):
    def check_fatal_exit(self, mode):
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-c", CHILD,
             str(ROOT / "python/sglang/srt/utils/watchdog.py"), mode],
            capture_output=True, timeout=3,
        )
        self.assertEqual(1, result.returncode, result.stderr.decode())
        self.assertLess(time.monotonic() - started, 3)

    def test_blocked_nccl_abort_cannot_prevent_exit(self):
        self.check_fatal_exit("blocked_abort")

    def test_ignored_sigquit_cannot_prevent_exit(self):
        self.check_fatal_exit("ignored_sigquit")


if __name__ == "__main__":
    unittest.main()
