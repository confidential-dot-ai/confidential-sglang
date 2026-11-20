"""Static checks of the router image recipe (confidential/router)."""
import importlib.util
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "confidential" / "router"
GATEWAY = ROOT / "sgl-model-gateway"
DOCKERFILE = (RECIPE / "Dockerfile").read_text()

spec = importlib.util.spec_from_file_location("private_router", RECIPE / "private_router.py")
private_router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(private_router)

# Logged by the router below the launcher's --log-level warn; the launcher
# drops it, which is intended (a client going away is not a router event).
NOT_FORWARDED = {"client_disconnected"}


class RouterRecipeTests(unittest.TestCase):
    def test_every_base_image_is_pinned_by_digest(self):
        froms = re.findall(r"^FROM\s+(\S+)", DOCKERFILE, re.M)
        self.assertEqual(len(froms), 2)
        for image in froms:
            self.assertRegex(image, r"@sha256:[0-9a-f]{64}$")

    def test_the_wheel_stage_is_named_for_the_engine_image(self):
        self.assertRegex(DOCKERFILE, r"(?m)^FROM \S+ AS router-wheel$")
        self.assertIn("--out /out", DOCKERFILE)

    def test_apt_reads_one_snapshot(self):
        self.assertRegex(DOCKERFILE, r"(?m)^ARG DEBIAN_SNAPSHOT=\d{8}T\d{6}Z$")
        self.assertIn("rm -f /etc/apt/sources.list.d/*", DOCKERFILE)
        self.assertNotIn("deb.debian.org", DOCKERFILE)

    def test_inputs_are_locked(self):
        self.assertIn("maturin build --locked", DOCKERFILE)
        self.assertIn("--require-hashes -r /inputs/requirements.lock", DOCKERFILE)
        self.assertRegex(DOCKERFILE, r"maturin==[\d.]+ --hash=sha256:%s")
        self.assertRegex(DOCKERFILE, r"\b[0-9a-f]{64}\b > /tmp/maturin.lock")
        lock = (RECIPE / "requirements.lock").read_text()
        for line in lock.splitlines():
            if re.match(r"^[A-Za-z0-9]", line):
                self.assertRegex(line, r"^[A-Za-z0-9_.-]+==\S+ \\$")

    def test_cargo_lock_is_committed(self):
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "sgl-model-gateway/Cargo.lock"],
            capture_output=True,
        )
        self.assertEqual(tracked.returncode, 0, "sgl-model-gateway/Cargo.lock must be tracked")

    def test_runs_as_1001_with_the_private_launcher(self):
        self.assertIn("USER 1001:1001", DOCKERFILE)
        self.assertIn("/usr/local/bin/ci-sglang-router", DOCKERFILE)
        self.assertIn("--no-compile", DOCKERFILE)

    def test_every_router_event_is_reviewed_by_the_launcher(self):
        events = set()
        for path in (GATEWAY / "src").rglob("*.rs"):
            events |= set(re.findall(r'ci_event = "([a-z_]+)"', path.read_text()))
        self.assertTrue(events)
        self.assertEqual(events - NOT_FORWARDED, set(private_router.EVENTS))


if __name__ == "__main__":
    unittest.main()
