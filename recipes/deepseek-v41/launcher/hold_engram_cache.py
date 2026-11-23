"""Keep immutable Engram checkpoint pages resident for all serving processes.

Two modes, logged as `mode=locked` or `mode=populated`:

- locked: every shard is mlocked (needs RLIMIT_MEMLOCK above the shards, ~198
  GiB for DeepSeek-V4.1; e.g. systemd LimitMEMLOCK=infinity). The pages cannot
  be reclaimed while this process lives.
- populated: mlock is refused (no RLIMIT_MEMLOCK headroom and no CAP_IPC_LOCK,
  as under C8s, whose containerd sets no LimitMEMLOCK). Every shard is mapped
  read-only and faulted into the page cache (MADV_POPULATE_READ), and every
  --refresh-seconds the mappings are populated again, so pages the kernel
  reclaimed come back before a worker needs them. The pages can be reclaimed
  under memory pressure in between.

--lock auto (default) tries locked and falls back to populated; --lock required
fails if mlock is refused; --lock off skips mlock. Read-only mappings avoid
copy-on-write allocation when pages fault. Stopping this process releases the
locks and mappings; checkpoint files are never modified.
"""

import argparse
import ctypes
import errno
import json
import mmap
import os
import signal
import socket
import time
from pathlib import Path

import numpy as np

MADV_WILLNEED = 3
MADV_POPULATE_READ = 22  # Linux 5.14+
PAGE = os.sysconf("SC_PAGE_SIZE")
# mlock fails with these when the limit or the capability is missing.
LOCK_REFUSED = {errno.EPERM, errno.ENOMEM, errno.EAGAIN}

libc = ctypes.CDLL(None, use_errno=True)
for name in ("mlock", "munlock", "madvise"):
    getattr(libc, name).argtypes = (ctypes.c_void_p, ctypes.c_size_t) + ((ctypes.c_int,) if name == "madvise" else ())
    getattr(libc, name).restype = ctypes.c_int
libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
libc.mincore.restype = ctypes.c_int


def resident_bytes(view):
    """Bytes of the mapping that are in memory now (mincore)."""
    pages = -(-view.nbytes // PAGE)
    vec = np.zeros(pages, dtype=np.uint8)
    if libc.mincore(view.ctypes.data, view.nbytes, vec.ctypes.data):
        return -1
    return int(np.count_nonzero(vec & 1)) * PAGE


def populate(view):
    """Fault every page of a read-only mapping into the page cache."""
    if libc.madvise(view.ctypes.data, view.nbytes, MADV_POPULATE_READ) == 0:
        return
    # Kernels before 5.14 (EINVAL): ask for read-ahead, then read one byte per page.
    libc.madvise(view.ctypes.data, view.nbytes, MADV_WILLNEED)
    int(view[::PAGE].sum(dtype=np.uint64))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Write this process's PID here once every shard is resident "
        "(the container healthcheck and launcher read it)",
    )
    parser.add_argument("--lock", choices=("auto", "required", "off"), default="auto",
                        help="auto: mlock, else populate unlocked (default); required: fail "
                        "if mlock is refused; off: populate unlocked")
    parser.add_argument("--reserve-gib", type=float, default=32,
                        help="RAM that must stay available besides the shards")
    parser.add_argument("--refresh-seconds", type=float, default=300,
                        help="populated mode: populate the mappings again this often (0: never)")
    args = parser.parse_args()
    if args.ready_file:
        args.ready_file.unlink(missing_ok=True)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    files = sorted({v for k, v in index["weight_map"].items() if ".engram.embed." in k})
    if not files:
        raise ValueError("Checkpoint has no Engram tensors")
    paths = [args.model / name for name in files]
    total = sum(p.stat().st_size for p in paths)
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if total + args.reserve_gib * 1024**3 > available:
        raise MemoryError(
            f"Insufficient available RAM to hold Engram with a {args.reserve_gib:g} GiB reserve"
        )
    started = time.monotonic()
    mappings = []
    for path in paths:
        with path.open("rb") as f:
            mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mappings.append((path, mapping, np.frombuffer(mapping, dtype=np.uint8)))

    mode, reason = "populated", "--lock off"
    if args.lock != "off":
        locked = []
        for path, _, view in mappings:
            print(f"Locking {path.name}: {view.nbytes} bytes", flush=True)
            if libc.mlock(view.ctypes.data, view.nbytes):
                error = ctypes.get_errno()
                if args.lock == "required" or error not in LOCK_REFUSED:
                    raise OSError(error, os.strerror(error))
                # All or nothing: a part-locked set would hide the refusal.
                for view_locked in locked:
                    libc.munlock(view_locked.ctypes.data, view_locked.nbytes)
                reason = f"mlock refused ({errno.errorcode.get(error, error)}: {os.strerror(error)})"
                print(f"mode=populated: {reason}; holding the shards unlocked", flush=True)
                break
            locked.append(view)
            print(f"Resident: {path.name}", flush=True)
        else:
            mode, reason = "locked", "mlock"
    if mode == "populated":
        for path, _, view in mappings:
            populate(view)
            print(f"Populated {path.name}: {resident_bytes(view)} of {view.nbytes} bytes resident", flush=True)
    resident = sum(resident_bytes(view) for _, _, view in mappings)
    print(
        f"READY: mode={mode} ({reason}): {total} checkpoint bytes read-only in the shared "
        f"CPU page cache, {resident} resident, {time.monotonic() - started:.1f} s",
        flush=True,
    )
    if args.ready_file:
        staging = args.ready_file.with_suffix(".tmp")
        staging.write_text(f"{os.getpid()}\n")
        staging.replace(args.ready_file)
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"READY=1", address)
    if mode == "locked" or args.refresh_seconds <= 0:
        while True:
            signal.pause()
    while True:
        time.sleep(args.refresh_seconds)
        before = sum(resident_bytes(view) for _, _, view in mappings)
        if before < total:
            for _, _, view in mappings:
                populate(view)
            print(f"mode=populated: refreshed, {total - before} bytes had been reclaimed", flush=True)


if __name__ == "__main__":
    main()
