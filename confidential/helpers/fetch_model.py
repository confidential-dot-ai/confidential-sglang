#!/usr/bin/env python3
"""Fill a plain model cache from Hugging Face and check it against a pinned byte manifest.

fetch-model manifest: print the byte manifest of a pinned revision (release
authors; it reads the Hugging Face file list, where every LFS file carries its
SHA-256, and hashes the small non-LFS files).

fetch-model fetch: runs in the TEE before the worker. The manifest file comes
from the release values and its SHA-256 is pinned in this argv. Every file
whose bytes do not match is downloaded again (up to ATTEMPTS times), files
the manifest does not list are deleted, and the manifest is written next to
the files, where wait-for-model checks every file once more before the server
starts.

fetch writes fixed JSON events to stdout, one per line, and nothing else
there (EVENT_KEYS). They name only manifest files, the model server's host
name, sizes, SHA-256 digests of model bytes, HTTP status codes and fixed error
classes; never a URL, a header or a response body. sglang-pair forwards them
(model_fetch_started, model_fetch_retry, model_fetch_complete,
model_fetch_failed). The free-text error still goes to stderr.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BLOCK = 8 * 1024 * 1024
WORKERS = 8
SCHEMA = "confidential-benchmark/local-model-byte-manifest/v1"
# Each file is downloaded at most this often; a retry waits RETRY_SECONDS x attempt.
ATTEMPTS = 3
RETRY_SECONDS = 2.0
# The events of `fetch` and their exact keys (a value may be null).
EVENT_KEYS = {
    "model_fetch_started": frozenset({"event", "host", "files", "size_bytes"}),
    "model_fetch_retry": frozenset({"event", "host", "file", "attempt", "error_class", "http_status"}),
    "model_fetch_complete": frozenset({"event", "host", "downloaded", "kept", "size_bytes", "duration_ms"}),
    "model_fetch_failed": frozenset({"event", "host", "file", "attempt", "error_class", "http_status",
                                     "expected_size", "got_size", "expected_sha256", "got_sha256"}),
}
ERROR_CLASSES = frozenset({"http_error", "connection", "timeout", "size_mismatch", "digest_mismatch", "io",
                           "manifest_digest", "manifest_invalid", "unsafe_path"})
HOST = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?")
_EVENT_LOCK = threading.Lock()


class FetchError(ValueError):
    """`error_class` and the file fields describe the failure in model_fetch_failed."""

    def __init__(self, message: str, error_class: str = "manifest_invalid", **details: object) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.details = details


def emit(event: dict) -> None:
    """Write one event of EVENT_KEYS to stdout. A value outside the contract is a bug: refuse it."""
    if set(event) != EVENT_KEYS.get(event.get("event"), frozenset()):
        raise ValueError("invalid fetch-model event")
    if event.get("error_class") not in (None, *ERROR_CLASSES):
        raise ValueError("invalid fetch-model error class")
    line = json.dumps(event, separators=(",", ":"), sort_keys=True)
    with _EVENT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def endpoint_host(endpoint: str) -> str | None:
    host = (urllib.parse.urlsplit(endpoint).hostname or "").lower()
    return host if HOST.fullmatch(host) else None


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: object) -> str:
    if not isinstance(name, str) or not name or name.startswith("/") or ".." in Path(name).parts:
        raise FetchError(f"unsafe path in the manifest: {name!r}")
    return name


def file_url(endpoint: str, repository: str, revision: str, name: str) -> str:
    return f"{endpoint}/{repository}/resolve/{revision}/{urllib.parse.quote(name)}"


def tree(endpoint: str, repository: str, revision: str) -> list[dict]:
    url = f"{endpoint}/api/models/{repository}/tree/{revision}?recursive=true"
    entries = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            entries += json.loads(response.read())
            match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
        url = match.group(1) if match else ""
    return sorted((item for item in entries if item.get("type") == "file"), key=lambda item: item["path"])


def build_manifest(endpoint: str, repository: str, revision: str) -> bytes:
    entries = tree(endpoint, repository, revision)
    files = []
    for item in entries:
        name = safe_name(item["path"])
        lfs = item.get("lfs")
        if lfs:
            digest, size = lfs["oid"], lfs["size"]
        else:
            with urllib.request.urlopen(file_url(endpoint, repository, revision, name), timeout=60) as response:
                body = response.read()
            digest, size = hashlib.sha256(body).hexdigest(), len(body)
        files.append({"content_sha256": digest, "path": name, "size": size})
    if not files:
        raise FetchError("the revision has no files")
    payload = {"files": files, "immutable_revision": revision, "repository": repository}
    return canonical({
        "canonical_local_manifest_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "classification": "verified_local_snapshot_bytes",
        "files": files,
        "immutable_ref": f"{repository}@{revision}",
        "immutable_revision": revision,
        "repository": repository,
        "schema": SCHEMA,
        "source_remote_inventory_sha256": hashlib.sha256(canonical(entries)).hexdigest(),
    }) + b"\n"


def download(url: str, target: Path, digest: str, size: int) -> None:
    partial = target.with_name(target.name + ".partial")
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            for block in iter(lambda: response.read(BLOCK), b""):
                hasher.update(block)
                stream.write(block)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{target.name}: HTTP {error.code}", "http_error", http_status=error.code) from None
    except TimeoutError:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{target.name}: the download timed out", "timeout") from None
    except urllib.error.URLError as error:
        partial.unlink(missing_ok=True)
        kind = "timeout" if isinstance(error.reason, TimeoutError) else "connection"
        raise FetchError(f"{target.name}: the model server is unreachable", kind) from None
    except (ConnectionError, OSError) as error:
        partial.unlink(missing_ok=True)
        kind = "connection" if isinstance(error, ConnectionError) else "io"
        raise FetchError(f"{target.name}: {kind} error during the download", kind) from None
    got_size, got_digest = partial.stat().st_size, hasher.hexdigest()
    if got_size != size or got_digest != digest:
        partial.unlink()
        raise FetchError(f"{target.name}: the download does not match the manifest",
                         "size_mismatch" if got_size != size else "digest_mismatch",
                         expected_size=size, got_size=got_size, expected_sha256=digest, got_sha256=got_digest)
    partial.chmod(0o444)
    partial.replace(target)


def download_with_retries(url: str, target: Path, digest: str, size: int, *, host: str | None, name: str,
                          attempts: int = ATTEMPTS, wait: float = RETRY_SECONDS) -> None:
    """download(), up to `attempts` times. Each failed attempt but the last is one model_fetch_retry."""
    for attempt in range(1, attempts + 1):
        try:
            download(url, target, digest, size)
            return
        except FetchError as error:
            # A 4xx other than 408 and 429 does not change on a retry.
            status = error.details.get("http_status")
            final = attempt == attempts or (isinstance(status, int) and 400 <= status < 500 and status not in (408, 429))
            if final:
                error.details.update(file=name, attempt=attempt)
                raise
            emit({"event": "model_fetch_retry", "host": host, "file": name, "attempt": attempt,
                  "error_class": error.error_class, "http_status": status})
            time.sleep(wait * attempt)


def fetch(args: argparse.Namespace) -> None:
    started = time.monotonic()
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise FetchError("the manifest file does not match its pinned SHA-256", "manifest_digest")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        raise FetchError("the manifest has another schema")
    repository, revision = manifest["repository"], manifest["immutable_revision"]
    try:
        files = {safe_name(item["path"]): (item["content_sha256"], item["size"]) for item in manifest["files"]}
    except FetchError as error:
        error.error_class = "unsafe_path"
        raise
    if args.revision_metadata in files:
        raise FetchError("the manifest lists itself")
    root = args.path
    host = endpoint_host(args.endpoint)
    emit({"event": "model_fetch_started", "host": host, "files": len(files),
          "size_bytes": sum(size for _digest, size in files.values())})

    def ensure(name: str) -> str:
        digest, size = files[name]
        target = root / name
        if target.is_file() and not target.is_symlink() and target.stat().st_size == size and sha256_file(target) == digest:
            return "kept"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        download_with_retries(file_url(args.endpoint, repository, revision, name), target, digest, size,
                              host=host, name=name)
        return "downloaded"

    order = sorted(files, key=lambda name: files[name][1], reverse=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(ensure, order))
    keep = set(files) | {args.revision_metadata}
    for directory, directories, names in os.walk(root, topdown=False):
        base = Path(directory)
        for name in names:
            if str((base / name).relative_to(root)) not in keep:
                (base / name).unlink()
        for name in directories:
            path = base / name
            if path.is_symlink():
                path.unlink()
            elif not any(path.iterdir()):
                path.rmdir()
    metadata = root / args.revision_metadata
    partial = metadata.with_name(metadata.name + ".partial")
    partial.write_bytes(raw)
    partial.chmod(0o444)
    partial.replace(metadata)
    downloaded = [name for name, result in zip(order, results) if result == "downloaded"]
    emit({"event": "model_fetch_complete", "host": host, "downloaded": len(downloaded),
          "kept": len(results) - len(downloaded), "size_bytes": sum(files[name][1] for name in downloaded),
          "duration_ms": int((time.monotonic() - started) * 1000)})
    print(f"fetch-model: {results.count('downloaded')} downloaded, {results.count('kept')} kept, "
          f"{repository}@{revision}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    steps = parser.add_subparsers(dest="step", required=True)
    manifest = steps.add_parser("manifest")
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--revision", required=True)
    run = steps.add_parser("fetch")
    run.add_argument("--path", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--revision-metadata", required=True)
    args = parser.parse_args()
    try:
        if args.step == "manifest":
            sys.stdout.buffer.write(build_manifest(args.endpoint, args.repository, args.revision))
        else:
            fetch(args)
    except (FetchError, OSError, KeyError, TypeError, ValueError) as error:
        if args.step == "fetch":
            details = error.details if isinstance(error, FetchError) else {}
            kind = error.error_class if isinstance(error, FetchError) else (
                "io" if isinstance(error, OSError) else "manifest_invalid")
            emit({"event": "model_fetch_failed", "host": endpoint_host(args.endpoint), "error_class": kind,
                  **{key: details.get(key) for key in ("file", "attempt", "http_status", "expected_size",
                                                         "got_size", "expected_sha256", "got_sha256")}})
        print(f"fetch-model: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
