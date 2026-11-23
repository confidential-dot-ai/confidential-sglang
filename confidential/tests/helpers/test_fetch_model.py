from __future__ import annotations

import hashlib
import http.server
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FETCH = ROOT / "confidential/helpers/fetch_model.py"
GATE = ROOT / "confidential/helpers/wait_for_model.py"
REPOSITORY = "example/tiny"
REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
METADATA = ".model-lock-local-manifest.json"
FILES = {
    "config.json": b'{"model_type":"llama"}\n',
    "tokenizer_config.json": b'{"tokenizer_class":"LlamaTokenizer"}\n',
    "model.safetensors": b"weights" * 1000,
    "sub/extra.txt": b"nested\n",
}
LFS = {"model.safetensors"}


class FakeHub(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    requests: list[str] = []
    # The HTTP statuses the next downloads of a file get, before it is served.
    failures: dict[str, list[int]] = {}

    def do_GET(self) -> None:  # noqa: N802
        self.requests.append(self.path)
        name = self.path.rsplit("/resolve/" + REVISION + "/", 1)[-1]
        if self.failures.get(name):
            self.send_error(self.failures[name].pop(0))
            return
        tree = f"/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true"
        prefix = f"/{REPOSITORY}/resolve/{REVISION}/"
        if self.path == tree:
            entries = [{"type": "directory", "path": "sub"}]
            for name, body in self.files.items():
                entry = {"type": "file", "path": name, "size": len(body), "oid": "0" * 40}
                if name in LFS:
                    entry["lfs"] = {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}
                entries.append(entry)
            body = json.dumps(entries).encode()
        elif self.path.startswith(prefix) and self.path[len(prefix):] in self.files:
            body = self.files[self.path[len(prefix):]]
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


class FetchModelTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeHub.files = dict(FILES)
        FakeHub.requests = []
        FakeHub.failures = {}
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.cache = self.base / "cache"
        self.cache.mkdir()
        self.manifest = self.base / "manifest.json"
        result = self.fetch_model("manifest", "--repository", REPOSITORY, "--revision", REVISION)
        self.assertEqual(0, result.returncode, result.stderr)
        self.manifest.write_bytes(result.stdout)
        self.digest = hashlib.sha256(result.stdout).hexdigest()
        FakeHub.requests = []

    def fetch_model(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["python3", str(FETCH), "--endpoint", self.endpoint, *args],
                              capture_output=True, timeout=30)

    def fetch(self, digest: str | None = None) -> subprocess.CompletedProcess[bytes]:
        return self.fetch_model("fetch", "--path", str(self.cache), "--manifest", str(self.manifest),
                                "--manifest-sha256", digest or self.digest, "--revision-metadata", METADATA)

    def downloads(self) -> list[str]:
        return [path for path in FakeHub.requests if "/resolve/" in path]

    def test_manifest_lists_every_file_with_its_sha256(self) -> None:
        manifest = json.loads(self.manifest.read_bytes())
        self.assertEqual(f"{REPOSITORY}@{REVISION}", manifest["immutable_ref"])
        self.assertEqual(
            {name: (hashlib.sha256(body).hexdigest(), len(body)) for name, body in FILES.items()},
            {item["path"]: (item["content_sha256"], item["size"]) for item in manifest["files"]},
        )

    def test_fetch_fills_the_cache_and_the_gate_accepts_it(self) -> None:
        result = self.fetch()
        self.assertEqual(0, result.returncode, result.stderr)
        for name, body in FILES.items():
            self.assertEqual(body, (self.cache / name).read_bytes())
        self.assertEqual(self.manifest.read_bytes(), (self.cache / METADATA).read_bytes())
        for path in self.cache.rglob("*"):
            if path.is_dir():
                path.chmod(0o555)
        self.cache.chmod(0o555)
        self.addCleanup(lambda: [p.chmod(0o755) for p in [self.cache, *self.cache.rglob("*")] if p.is_dir()])
        mountinfo = self.base / "mountinfo"
        mountinfo.write_text(f"44 33 252:16 / {self.cache.resolve()} ro,nosuid,nodev,noexec - ext4 /dev/vdc ro\n")
        digests = {METADATA: self.digest, **{name: hashlib.sha256(body).hexdigest() for name, body in FILES.items()
                                             if name != "sub/extra.txt"}}
        gate = ["python3", str(GATE), "--path", str(self.cache), "--timeout-seconds", "1", "--poll-seconds", "0.05",
                "--expected-repository", REPOSITORY, "--expected-revision", REVISION,
                "--revision-metadata", METADATA, "--mountinfo", str(mountinfo),
                *[f"--expected-file={name}={digest}" for name, digest in sorted(digests.items())]]
        command = ["--", "python3", "-c", "print('server-started')"]
        refused = subprocess.run(gate + command, text=True, capture_output=True, timeout=5)
        self.assertNotEqual(0, refused.returncode)
        self.assertIn("dm-verity mapping is not attached", refused.stderr)
        accepted = subprocess.run(gate + ["--plain-cache"] + command, text=True, capture_output=True, timeout=5)
        self.assertEqual(0, accepted.returncode, accepted.stderr)
        # Server output is private: only the fixed check and exit events reach stdout.
        verified, exit_line = accepted.stdout.splitlines()
        self.assertEqual(("model_verified", len(FILES), sum(map(len, FILES.values()))),
                         tuple(json.loads(verified)[key] for key in ("event", "files", "size_bytes")))
        self.assertEqual('{"event":"worker_exit","exit_code":0,"signal":null}', exit_line)

    def test_second_fetch_keeps_good_files_and_repairs_the_rest(self) -> None:
        self.assertEqual(0, self.fetch().returncode)
        (self.cache / "config.json").chmod(0o644)
        (self.cache / "config.json").write_bytes(b'{"model_type":"evil"}\n')
        (self.cache / "planted.py").write_bytes(b"print('x')\n")
        (self.cache / "junk").mkdir()
        FakeHub.requests = []
        result = self.fetch()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([f"/{REPOSITORY}/resolve/{REVISION}/config.json"], self.downloads())
        self.assertEqual(FILES["config.json"], (self.cache / "config.json").read_bytes())
        self.assertFalse((self.cache / "planted.py").exists())
        self.assertFalse((self.cache / "junk").exists())

    def test_unpinned_manifest_is_refused(self) -> None:
        result = self.fetch("0" * 64)
        self.assertNotEqual(0, result.returncode)
        self.assertIn(b"pinned SHA-256", result.stderr)
        self.assertEqual([], self.downloads())

    def test_plain_http_model_server_cannot_change_a_byte(self) -> None:
        # The chart allows a plain HTTP model server (inference.model.download.scheme: http):
        # the pinned byte manifest, not TLS, is what pins the model bytes. A server or a
        # network attacker that changes bytes, even at the same size, is refused, and
        # no changed file is left for the server to read.
        self.assertTrue(self.endpoint.startswith("http://"))
        body = FILES["model.safetensors"]
        FakeHub.files["model.safetensors"] = b"W" + body[1:]
        result = self.fetch()
        self.assertNotEqual(0, result.returncode)
        self.assertIn(b"model.safetensors: the download does not match the manifest", result.stderr)
        self.assertFalse((self.cache / "model.safetensors").exists())
        self.assertFalse((self.cache / "model.safetensors.partial").exists())
        self.assertFalse((self.cache / METADATA).exists())
        FakeHub.files["model.safetensors"] = body
        self.assertEqual(0, self.fetch().returncode)
        self.assertEqual(body, (self.cache / "model.safetensors").read_bytes())


    def events(self, result: subprocess.CompletedProcess[bytes]) -> list[dict]:
        """fetch's stdout: one fixed event per line, with no URL, path or response text in it."""
        lines = result.stdout.decode().splitlines()
        for line in lines:
            self.assertNotIn("://", line)
            self.assertNotIn(self.endpoint, line)
            self.assertNotIn("/resolve/", line)
            self.assertNotIn(str(self.base), line)
        return [json.loads(line) for line in lines]

    def test_fetch_reports_its_start_and_its_end(self) -> None:
        result = self.fetch()
        self.assertEqual(0, result.returncode, result.stderr)
        started, complete = self.events(result)
        self.assertEqual({"event": "model_fetch_started", "host": "127.0.0.1", "files": len(FILES),
                          "size_bytes": sum(map(len, FILES.values()))}, started)
        self.assertEqual(("model_fetch_complete", "127.0.0.1", len(FILES), 0, sum(map(len, FILES.values()))),
                         tuple(complete[key] for key in ("event", "host", "downloaded", "kept", "size_bytes")))
        self.assertIsInstance(complete["duration_ms"], int)
        # A second fetch keeps every file.
        again = self.events(self.fetch())[-1]
        self.assertEqual((0, len(FILES), 0), (again["downloaded"], again["kept"], again["size_bytes"]))

    def test_a_failed_download_is_retried_and_reported(self) -> None:
        FakeHub.failures = {"config.json": [503]}
        result = self.fetch()
        self.assertEqual(0, result.returncode, result.stderr)
        events = self.events(result)
        self.assertEqual({"event": "model_fetch_retry", "host": "127.0.0.1", "file": "config.json", "attempt": 1,
                          "error_class": "http_error", "http_status": 503},
                         next(event for event in events if event["event"] == "model_fetch_retry"))
        self.assertEqual("model_fetch_complete", events[-1]["event"])

    def test_a_changed_byte_is_reported_with_both_sizes_and_digests(self) -> None:
        body = FILES["model.safetensors"]
        FakeHub.files["model.safetensors"] = b"W" + body[1:] + b"!"
        result = self.fetch()
        self.assertNotEqual(0, result.returncode)
        events = self.events(result)
        self.assertEqual(["model_fetch_started", "model_fetch_retry", "model_fetch_retry", "model_fetch_failed"],
                         [event["event"] for event in events])
        self.assertEqual({"event": "model_fetch_failed", "host": "127.0.0.1", "file": "model.safetensors",
                          "attempt": 3, "error_class": "size_mismatch", "http_status": None,
                          "expected_size": len(body), "got_size": len(body) + 1,
                          "expected_sha256": hashlib.sha256(body).hexdigest(),
                          "got_sha256": hashlib.sha256(b"W" + body[1:] + b"!").hexdigest()}, events[-1])

    def test_a_missing_file_is_not_retried(self) -> None:
        FakeHub.failures = {"config.json": [404]}
        result = self.fetch()
        self.assertNotEqual(0, result.returncode)
        failed = self.events(result)[-1]
        self.assertEqual(("model_fetch_failed", "config.json", 1, "http_error", 404),
                         tuple(failed[key] for key in ("event", "file", "attempt", "error_class", "http_status")))
        self.assertEqual(1, sum(path.endswith("/config.json") for path in self.downloads()))

    def test_an_unpinned_manifest_is_reported(self) -> None:
        failed, = self.events(self.fetch("0" * 64))
        self.assertEqual(("model_fetch_failed", "manifest_digest", None),
                         (failed["event"], failed["error_class"], failed["file"]))


if __name__ == "__main__":
    unittest.main()
