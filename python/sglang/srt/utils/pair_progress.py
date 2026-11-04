"""Publish numeric scheduler progress for a local pair supervisor.

No prompts, request IDs, tensors, exception text or local variables leave this
module. A heartbeat comes from the scheduler thread, never an independent
timer that could keep a stalled scheduler looking healthy.
"""

from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time


_reporter = None


def error_event(event, *, rank=None, code=None):
    role = os.environ.get("SGLANG_PAIR_ROLE")
    directory = os.environ.get("SGLANG_PAIR_RUN_DIR")
    allowed = {"rank_failed", "watchdog_timeout", "communicator_abort_timeout", "cuda_error", "nccl_error", "diagnostic_failed", "kv_transfer_failed"}
    if role not in {"prefill", "decode"} or not directory or event not in allowed:
        return
    if rank is None and _reporter is not None:
        rank = _reporter.rank
    value = {"event": event, "role": role}
    for key, number in (("rank", rank), ("code", code)):
        if number is not None:
            if type(number) is not int or not -(2**63) < number < 2**63:
                return
            value[key] = number
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.setblocking(False)
            sender.sendto(json.dumps(value).encode(), str(Path(directory) / "events.sock"))
    except OSError:
        pass  # Diagnostics must not block or keep a failed process alive.


class Progress:
    def __init__(self, directory, role, rank, *, event_factory=None, clock=time.monotonic):
        self.path = Path(directory) / f"{role}-{rank}.json"
        self.rank = rank
        self.clock = clock
        self.event_factory = event_factory
        self.pending = deque()
        self.depth = 0
        self.stage_started = 0.0
        self.completed = 0.0
        self.last_write = 0.0
        self.starting = True
        self.failed = False
        self.publish(force=True)

    def publish(self, force=False):
        now = self.clock()
        if not force and now - self.last_write < 0.5:
            return
        pending_since = min([stamp for stamp, _ in self.pending] + ([self.stage_started] if self.depth else []) or [0.0])
        state = "failed" if self.failed else "starting" if self.starting else "busy" if pending_since else "idle"
        record = {"pid": os.getpid(), "state": state, "updated": now,
                  "completed": self.completed, "pending_since": pending_since}
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(record))
            temporary.replace(self.path)
            self.last_write = now
        except OSError:
            # The external supervisor treats absent/stale progress as failure.
            error_event("diagnostic_failed", rank=self.rank)

    def poll(self):
        while self.pending and self.pending[0][1].query():
            self.pending.popleft()
            self.completed = self.clock()

    @contextmanager
    def stage(self, name):
        self.poll()
        if not self.depth:
            self.stage_started = self.clock()
        self.depth += 1
        self.publish(force=name == "run_batch")
        try:
            yield
            if name == "run_batch":
                # Enqueue a completion event on the worker's forward stream.
                # Merely incrementing forward_ct does not prove GPU progress.
                event = self.event_factory()
                self.pending.append((self.stage_started, event))
        except BaseException:
            self.failed = True
            error_event("rank_failed", rank=self.rank)
            raise
        finally:
            self.depth -= 1
            self.poll()
            self.publish(force=self.failed)

    def ready(self):
        self.starting = False
        self.publish(force=True)


def initialize(rank):
    global _reporter
    directory = os.environ.get("SGLANG_PAIR_RUN_DIR")
    role = os.environ.get("SGLANG_PAIR_ROLE")
    if directory and role in {"prefill", "decode"}:
        _reporter = Progress(directory, role, rank)


def scheduler_ready(scheduler):
    if _reporter is None:
        return
    import torch

    def completion_event():
        event = torch.cuda.Event()
        event.record(scheduler.forward_stream)
        return event

    _reporter.event_factory = completion_event
    _reporter.ready()


@contextmanager
def stage(name):
    if _reporter is None or _reporter.starting:
        yield
    else:
        with _reporter.stage(name):
            yield
