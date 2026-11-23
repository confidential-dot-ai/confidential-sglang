#!/usr/bin/env python3
"""Fill a plain model cache from Hugging Face and check it against a pinned byte manifest.

fetch-model manifest: print the byte manifest of a pinned revision (release
authors; it reads the Hugging Face file list, where every LFS file carries its
SHA-256, and hashes the small non-LFS files).

fetch-model fetch: runs in the TEE before the worker. The manifest file comes
from the release values and its SHA-256 is pinned in this argv. Every file
whose bytes do not match is downloaded again, files the manifest does not
list are deleted, and the manifest is written next to the files, where
wait-for-model checks every file once more before the server starts.

fetch uses CONNECTIONS parallel HTTP/1.1 connections in total. A file larger
than CHUNK is split into byte ranges, so one very large file uses many
connections. A range is tried up to ATTEMPTS times; with more than one
--endpoint (for example an in-site mirror first, then the main model server),
each retry goes to a server not yet tried for it, and a server that does not
answer moves to the back for every later range. The SHA-256 of the whole file,
not the transport, is the check (confidential-inference #363).

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
import http.client
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
# Measured to the us-central-1 GPU workers on 2026-11-23: 8 whole-file
# connections gave about 16 MB/s from the Azure model server, 64 ranged
# connections about 84 MB/s, and more than 64 was slower
# (confidential-inference-model-server docs/download-throughput.md;
# confidential-inference #363).
CONNECTIONS = 64
CHUNK = 64 * 1024 * 1024
HASHERS = 8
SCHEMA = "confidential-benchmark/local-model-byte-manifest/v1"
# Each byte range is tried at most this often; retry n waits RETRY_SECONDS x 2^(n-1), at most 8 s.
ATTEMPTS = 4
RETRY_SECONDS = 1.0
# The events of `fetch` and their exact keys (a value may be null).
EVENT_KEYS = {
    "model_fetch_started": frozenset({"event", "host", "files", "size_bytes"}),
    "model_fetch_retry": frozenset({"event", "host", "file", "attempt", "error_class", "http_status"}),
    "model_fetch_complete": frozenset({"event", "host", "downloaded", "kept", "size_bytes", "duration_ms"}),
    "model_fetch_failed": frozenset({"event", "host", "file", "attempt", "error_class", "http_status",
                                     "expected_size", "got_size", "expected_sha256", "got_sha256"}),
}
ERROR_CLASSES = frozenset({"http_error", "connection", "timeout", "size_mismatch", "digest_mismatch", "io",
                           "manifest_digest", "manifest_invalid", "unsafe_path", "range_unsupported"})
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


class Endpoints:
    """The model servers in order of preference. One that does not answer moves to the back for every range."""

    def __init__(self, endpoints: list[str]) -> None:
        self.order = list(endpoints)
        self.lock = threading.Lock()

    def pick(self, tried: set[str]) -> str:
        """The preferred server not yet tried for this range, else the preferred one."""
        with self.lock:
            return next((endpoint for endpoint in self.order if endpoint not in tried), self.order[0])

    def failed(self, endpoint: str) -> None:
        with self.lock:
            if len(self.order) > 1 and self.order[0] == endpoint:
                self.order.append(self.order.pop(0))
                print(f"fetch-model: {endpoint} failed; using {self.order[0]} first", file=sys.stderr, flush=True)


def classify(error: BaseException) -> tuple[str, int | None]:
    """The fixed error class and HTTP status of a failed range."""
    if isinstance(error, FetchError):
        return error.error_class, error.details.get("http_status")
    if isinstance(error, urllib.error.HTTPError):
        return "http_error", error.code
    if isinstance(error, TimeoutError) or (isinstance(error, urllib.error.URLError)
                                           and isinstance(error.reason, TimeoutError)):
        return "timeout", None
    if isinstance(error, (urllib.error.URLError, ConnectionError, http.client.HTTPException)):
        return "connection", None
    return "io", None


def fetch_range(endpoints: Endpoints, name: str, path: str, fd: int, start: int, end: int, whole: bool,
                attempts: int | None = None, wait: float | None = None) -> None:
    """Write bytes start..end of file `name` at the same offset of fd, from the first server that gives them.

    A failed attempt is one model_fetch_retry, and the next attempt goes to a
    server not yet tried for this range. A 4xx other than 408 and 429 ends the
    range only when every server gave one.
    """
    attempts = ATTEMPTS if attempts is None else attempts
    wait = RETRY_SECONDS if wait is None else wait
    tried: set[str] = set()
    for attempt in range(1, attempts + 1):
        endpoint = endpoints.pick(tried)
        tried.add(endpoint)
        request = urllib.request.Request(endpoint + path, headers={} if whole else {"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if not whole and response.status != 206:
                    raise FetchError(f"{name}: the server ignored the byte range", "range_unsupported",
                                     http_status=response.status)
                buffer = memoryview(bytearray(BLOCK))
                offset = start
                while offset <= end:
                    count = response.readinto(buffer[:min(BLOCK, end + 1 - offset)])
                    if not count:
                        raise ConnectionError("the response ended early")
                    os.pwrite(fd, buffer[:count], offset)
                    offset += count
            return
        except (OSError, FetchError, http.client.HTTPException) as error:
            kind, status = classify(error)
            if kind in ("connection", "timeout"):
                endpoints.failed(endpoint)  # no answer: prefer the next server for every range
            permanent = isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)
            final = attempt == attempts or (permanent and tried >= set(endpoints.order))
            if final:
                raise FetchError(f"{name}: bytes {start}-{end}: {kind}", kind, http_status=status, file=name,
                                 attempt=attempt, host=endpoint_host(endpoint)) from None
            emit({"event": "model_fetch_retry", "host": endpoint_host(endpoint), "file": name, "attempt": attempt,
                  "error_class": kind, "http_status": status})
            if not permanent:
                time.sleep(min(8.0, wait * 2 ** (attempt - 1)))


def download_all(endpoints: Endpoints, jobs: list[tuple[str, str, Path, str, int]]) -> None:
    """Download every (name, url path, target, digest, size) over CONNECTIONS ranged connections in total.

    Each file is written to <target>.partial, checked against its SHA-256 once
    all its ranges are written, and only then renamed into place. The first
    failure (by file name) is raised with its details.
    """
    lock = threading.Lock()
    left: dict[str, int] = {}
    fds: dict[str, int] = {}
    failed: dict[str, FetchError] = {}
    chunks = []
    for name, path, target, digest, size in jobs:
        partial = target.with_name(target.name + ".partial")
        fds[name] = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fds[name], size)
        starts = list(range(0, size, CHUNK)) or [0]
        left[name] = len(starts)
        chunks += [(name, path, start, min(size, start + CHUNK) - 1, len(starts) == 1) for start in starts]
    info = {name: (target, digest, size) for name, _, target, digest, size in jobs}

    def verify(name: str) -> None:
        target, digest, size = info[name]
        partial = target.with_name(target.name + ".partial")
        if name in failed:
            partial.unlink(missing_ok=True)
            return
        got_size, got = partial.stat().st_size, sha256_file(partial)
        if got_size != size or got != digest:
            partial.unlink()
            failed.setdefault(name, FetchError(
                f"{name}: the download does not match the manifest",
                "size_mismatch" if got_size != size else "digest_mismatch", file=name, expected_size=size,
                got_size=got_size, expected_sha256=digest, got_sha256=got))
            return
        partial.chmod(0o444)
        partial.replace(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=HASHERS) as hashers:
        checks = []

        def run(chunk: tuple[str, str, int, int, bool]) -> None:
            name, path, start, end, whole = chunk
            try:
                if name not in failed and end >= start:
                    fetch_range(endpoints, name, path, fds[name], start, end, whole)
            except FetchError as error:
                failed.setdefault(name, error)
            except OSError as error:
                failed.setdefault(name, FetchError(f"{name}: {error.strerror or 'io error'}", "io", file=name))
            with lock:
                left[name] -= 1
                finished = left[name] == 0
            if finished:
                os.close(fds[name])
                checks.append(hashers.submit(verify, name))

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as pool:
            list(pool.map(run, chunks))
        for check in checks:
            check.result()
    if failed:
        first = failed[min(failed)]
        raise FetchError("; ".join(f"{error}" for _name, error in sorted(failed.items())), first.error_class,
                         **first.details)


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
    host = endpoint_host(args.endpoint[0])
    emit({"event": "model_fetch_started", "host": host, "files": len(files),
          "size_bytes": sum(size for _digest, size in files.values())})

    def kept(name: str) -> bool:
        digest, size = files[name]
        target = root / name
        return (target.is_file() and not target.is_symlink() and target.stat().st_size == size
                and sha256_file(target) == digest)

    order = sorted(files, key=lambda name: files[name][1], reverse=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=HASHERS) as pool:
        downloaded = [name for name, good in zip(order, pool.map(kept, order)) if not good]
    jobs = []
    for name in downloaded:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        jobs.append((name, file_url("", repository, revision, name), target, *files[name]))
    download_all(Endpoints(args.endpoint), jobs)
    results = ["downloaded"] * len(downloaded) + ["kept"] * (len(files) - len(downloaded))
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
    emit({"event": "model_fetch_complete", "host": host, "downloaded": len(downloaded),
          "kept": len(results) - len(downloaded), "size_bytes": sum(files[name][1] for name in downloaded),
          "duration_ms": int((time.monotonic() - started) * 1000)})
    print(f"fetch-model: {results.count('downloaded')} downloaded, {results.count('kept')} kept, "
          f"{repository}@{revision}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", action="append",
                        help="a model server; repeat it for fallbacks in order, for example an in-site mirror "
                             "first (default: https://huggingface.co)")
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
    args.endpoint = [endpoint.rstrip("/") for endpoint in args.endpoint or ["https://huggingface.co"]]
    try:
        if args.step == "manifest":
            sys.stdout.buffer.write(build_manifest(args.endpoint[0], args.repository, args.revision))
        else:
            fetch(args)
    except (FetchError, OSError, KeyError, TypeError, ValueError) as error:
        if args.step == "fetch":
            details = error.details if isinstance(error, FetchError) else {}
            kind = error.error_class if isinstance(error, FetchError) else (
                "io" if isinstance(error, OSError) else "manifest_invalid")
            emit({"event": "model_fetch_failed", "host": details.get("host") or endpoint_host(args.endpoint[0]),
                  "error_class": kind,
                  **{key: details.get(key) for key in ("file", "attempt", "http_status", "expected_size",
                                                         "got_size", "expected_sha256", "got_sha256")}})
        print(f"fetch-model: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
