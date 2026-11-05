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
_stack_stream = None
STAGES = {"recv_requests": 1, "process_input_requests": 2,
          "get_next_batch_to_run": 3, "run_batch": 4,
          "process_batch_result": 5, "sanity_check_cache": 6,
          "idle": 7, "process_queue": 8}


def _scheduler_snapshot(directory, role, rank):
    reporter = _reporter
    scheduler = getattr(reporter, "scheduler", None)
    if reporter is None or scheduler is None:
        _send_diagnostic(directory, {"event": "scheduler_snapshot_unavailable", "role": role,
                                     "rank": rank, "code": 1})
        return
    record = {"event": "scheduler_snapshot", "role": role, "rank": rank,
              "stage_code": reporter.stages[-1] if reporter.stages else 0,
              "last_step_ms": reporter.last_step_ms}
    if reporter.depth:
        record["step_age_ms"] = max(0, int((reporter.clock() - reporter.stage_started) * 1000))
    batch = getattr(scheduler, "cur_batch", None)
    if batch is None:
        batch = getattr(scheduler, "running_batch", None)
    requests = getattr(batch, "reqs", None)
    if type(requests) is list and len(requests) <= 65536:
        requests = tuple(requests)
        record["batch_size"] = len(requests)
        for attribute, field in (("origin_input_ids", "input_tokens"), ("output_ids", "output_tokens")):
            values = [getattr(request, attribute, None) for request in requests]
            if all(type(value) is list for value in values):
                record[field] = sum(len(value) for value in values)
    _send_diagnostic(directory, record)


def _diagnostic_requests(stream, directory, role, rank):
    import select

    while True:
        try:
            readable, _, _ = select.select([stream], [], [], 1)
            if not readable:
                continue
            command = stream.recv(1)
            if not command:
                return
        except (OSError, ValueError):
            return
        if command != b"N":
            continue
        try:
            _scheduler_snapshot(directory, role, rank)
        except Exception:
            _send_diagnostic(directory, {"event": "scheduler_snapshot_unavailable", "role": role,
                                         "rank": rank, "code": 2})
        try:
            import torch
            from sglang.srt.utils.pair_nccl_diagnostics import project_trace

            raw = torch._C._distributed_c10d._dump_nccl_trace_json(True, False)
            records, rejected = project_trace(raw)
            del raw
            sent = 0
            for record in reversed(records):
                # Preserve the newest operation first if a damaged receiver
                # cannot accept the complete bounded snapshot.
                sent += bool(_send_diagnostic(directory, {"event": "nccl_collective", "role": role,
                                                          "rank": rank, **record}))
            _send_diagnostic(directory, {"event": "nccl_snapshot", "role": role,
                                         "rank": rank, "count": sent,
                                         "rejected": rejected + len(records) - sent})
        except Exception:
            _send_diagnostic(directory, {"event": "nccl_snapshot_unavailable", "role": role,
                                         "rank": rank, "code": 2})


def _send_diagnostic(directory, record):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            # Snapshot traffic comes from the diagnostic thread only. A short
            # connected-socket wait handles the receiver's small datagram queue.
            # At most 34 records bound this wait to 680ms per rank. Error events
            # in the scheduler/watchdog remain nonblocking below.
            sender.settimeout(0.02)
            sender.connect(str(Path(directory) / "events.sock"))
            sender.send(json.dumps(record).encode())
        return True
    except OSError:
        return False


def error_event(event, *, rank=None, code=None, count=None, bytes_count=None):
    role = os.environ.get("SGLANG_PAIR_ROLE")
    directory = os.environ.get("SGLANG_PAIR_RUN_DIR")
    allowed = {"rank_failed", "watchdog_timeout", "communicator_abort_timeout", "cuda_error", "nccl_error", "diagnostic_failed", "kv_transfer_failed", "communicator_abort_started", "communicator_abort_finished", "communicator_abort_failed", "fatal_exit_deadline", "gpu_event_query_failed"}
    if role not in {"prefill", "decode"} or not directory or event not in allowed:
        return
    if rank is None and _reporter is not None:
        rank = _reporter.rank
    value = {"event": event, "role": role}
    for key, number in (("rank", rank), ("code", code), ("count", count), ("bytes", bytes_count)):
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
        self.unobserved_since = None
        self.depth = 0
        self.stage_started = 0.0
        self.completed = 0.0
        self.last_write = 0.0
        self.last_published_state = None
        self.last_poll = float("-inf")
        self.starting = True
        self.failed = False
        self.scheduler = None
        self.stages = []
        self.last_step_ms = 0
        self.publish(force=True)

    def publish(self, force=False):
        now = self.clock()
        if not force and now - self.last_write < 0.5:
            return
        pending_since = min([stamp for stamp, _ in self.pending]
                            + ([self.unobserved_since] if self.unobserved_since is not None else [])
                            + ([self.stage_started] if self.depth else []) or [0.0])
        state = "failed" if self.failed else "starting" if self.starting else "busy" if pending_since else "idle"
        record = {"pid": os.getpid(), "state": state, "updated": now,
                  "completed": self.completed, "pending_since": pending_since}
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(record))
            temporary.replace(self.path)
            self.last_write = now
            self.last_published_state = state
        except OSError:
            # The external supervisor treats absent/stale progress as failure.
            error_event("diagnostic_failed", rank=self.rank)

    def poll(self):
        now = self.clock()
        if now - self.last_poll < 0.1:
            return
        self.last_poll = now
        try:
            if self.unobserved_since is not None:
                # One marker covers all work submitted since the last marker.
                # Idle polling also records a marker, so the final batch cannot
                # disappear from progress tracking when no later batch arrives.
                self.pending.append((self.unobserved_since, self.event_factory()))
                self.unobserved_since = None
            while self.pending and self.pending[0][1].query():
                self.pending.popleft()
                self.completed = self.clock()
        except Exception:
            self.failed = True
            error_event("gpu_event_query_failed", rank=self.rank)
            self.publish(force=True)
            raise

    @contextmanager
    def stage(self, name):
        self.poll()
        if not self.depth:
            self.stage_started = self.clock()
        self.depth += 1
        self.stages.append(STAGES.get(name, 0))
        # Publish entry into busy work before it can block. Once busy, the
        # regular heartbeat preserves the oldest pending timestamp. Rewriting
        # the file for every batch adds I/O and GIL handoffs to the hot path.
        self.publish(force=name == "run_batch" and self.last_published_state != "busy")
        try:
            yield
            if name == "run_batch":
                # Track every batch immediately. CUDA marker creation and
                # queries run at most every 100ms, including during idle loops.
                if self.unobserved_since is None:
                    self.unobserved_since = self.stage_started
        except BaseException:
            self.failed = True
            error_event("rank_failed", rank=self.rank)
            raise
        finally:
            if name == "run_batch":
                self.last_step_ms = max(0, int((self.clock() - self.stage_started) * 1000))
            self.depth -= 1
            self.stages.pop()
            self.poll()
            self.publish(force=self.failed)

    def ready(self):
        self.starting = False
        self.publish(force=True)


def initialize(rank):
    global _reporter, _stack_stream
    directory = os.environ.get("SGLANG_PAIR_RUN_DIR")
    role = os.environ.get("SGLANG_PAIR_ROLE")
    if directory and role in {"prefill", "decode"}:
        _reporter = Progress(directory, role, rank)
        # Raw stack bytes go only to the local supervisor's memory. Its parser
        # exports approved code locations. No traceback file is created.
        stream = None
        registered = False
        try:
            import faulthandler
            import signal

            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stream.settimeout(0.2)
            stream.connect(str(Path(directory) / "stacks.sock"))
            stream.sendall(json.dumps({"role": role, "rank": rank}).encode() + b"\n")
            faulthandler.register(signal.SIGUSR2, file=stream.fileno(), all_threads=True)
            registered = True
            stream.sendall(b"ready\n")
            stream.setblocking(False)
            _stack_stream = stream
            import threading
            threading.Thread(target=_diagnostic_requests,
                             args=(stream, directory, role, rank), daemon=True).start()
        except (OSError, RuntimeError, ValueError):
            if registered:
                faulthandler.unregister(signal.SIGUSR2)
            if stream is not None:
                stream.close()
            error_event("diagnostic_failed", rank=rank)


def scheduler_ready(scheduler):
    if _reporter is None:
        return
    import torch

    def completion_event():
        event = torch.cuda.Event()
        event.record(scheduler.forward_stream)
        return event

    _reporter.event_factory = completion_event
    _reporter.scheduler = scheduler
    _reporter.ready()


@contextmanager
def stage(name):
    if _reporter is None or _reporter.starting:
        yield
    else:
        with _reporter.stage(name):
            yield
