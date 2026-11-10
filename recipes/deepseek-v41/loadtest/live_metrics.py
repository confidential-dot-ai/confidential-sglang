"""Live production-readiness metrics for a DeepSeek-V4.1 deployment.

Scrapes the Prometheus endpoints of the prefill and decode workers (started
with --enable-metrics) and of the router (--prometheus-port), and every
--interval seconds prints one line and appends one JSON record with rates over
that window:

  traffic      requests/s, input tok/s (total, cached, uncached), output tok/s,
               served TPM in total and per GPU, output share
  cache        prefix-cache hit, KV pool use (cache pressure), evictions/s,
               HiCache host-tier use, write-back (GPU->host) and load-back
               (host->GPU) tok/s
  preemption   retracted requests/s and tokens/s, paused and aborted requests
  queues       running and waiting requests, PD bootstrap/prealloc/transfer
               queues, router admission in flight and rejections/s
  latency      TTFT, inter-token latency, queue time and router request
               duration (p50/p90/p99 of the window, from histogram deltas)
  health       non-2xx router responses/s, KV-transfer and bootstrap failures,
               prefill retries, speculative accept length

It only reads metrics, so it can run beside any load: a load test, a soak, or
production traffic. With --duration it stops on its own and prints a summary
over the whole run; otherwise Ctrl-C prints the summary.

With --serve PORT it also publishes the latest window as Prometheus gauges at
http://HOST:PORT/metrics (prefix inference_, labels from --label k=v), so a
Prometheus can scrape the derived numbers (TPM per GPU, hit rate, TTFT/ITL
quantiles) without PromQL over the raw counters. inference_up is 0 when the last
scrape of the workers or router failed; the other gauges then keep their last
values.

In a PD deployment the input-side counters come from prefill and the output
side from decode: both servers count every request's prompt, so summing them
would double the input rate. A colocated (non-PD) worker is passed as --worker
and supplies both sides.

Standard library only, so it runs from the host python as well as the image.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass
class Scrape:
    """One endpoint's samples, summed over every label except `le`."""

    values: dict[str, float] = field(default_factory=dict)
    # histogram name -> {le: cumulative count}
    buckets: dict[str, dict[float, float]] = field(default_factory=dict)
    # labelled values kept for metrics whose labels matter (router status codes)
    labelled: dict[str, dict[tuple, float]] = field(default_factory=dict)


KEEP_LABELS = {"smg_http_responses_total": ("status",)}


def parse(text: str) -> Scrape:
    out = Scrape()
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, raw_labels, raw_value = m.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isnan(value):
            continue
        labels = dict(_LABEL.findall(raw_labels or ""))
        if name.endswith("_bucket") and "le" in labels:
            le = math.inf if labels["le"] in ("+Inf", "inf") else float(labels["le"])
            hist = out.buckets.setdefault(name[: -len("_bucket")], {})
            hist[le] = hist.get(le, 0.0) + value
            continue
        if name.endswith("_created"):
            continue
        out.values[name] = out.values.get(name, 0.0) + value
        keep = KEEP_LABELS.get(name)
        if keep:
            key = tuple(labels.get(k, "") for k in keep)
            series = out.labelled.setdefault(name, {})
            series[key] = series.get(key, 0.0) + value
    return out


def merge(scrapes: list[Scrape]) -> Scrape:
    """Sum several endpoints of one role (e.g. two prefill workers)."""
    out = Scrape()
    for s in scrapes:
        for k, v in s.values.items():
            out.values[k] = out.values.get(k, 0.0) + v
        for name, hist in s.buckets.items():
            dst = out.buckets.setdefault(name, {})
            for le, c in hist.items():
                dst[le] = dst.get(le, 0.0) + c
        for name, series in s.labelled.items():
            dst = out.labelled.setdefault(name, {})
            for key, v in series.items():
                dst[key] = dst.get(key, 0.0) + v
    return out


def counter_delta(new: float | None, old: float | None) -> float | None:
    """A counter's increase; a decrease means the process restarted."""
    if new is None:
        return None
    if old is None:
        return None
    return new if new < old else new - old


def histogram_quantiles(
    new: dict[float, float] | None,
    old: dict[float, float] | None,
    qs=(50, 90, 99),
) -> dict[str, float | None]:
    """Quantiles of the observations made between two cumulative snapshots,
    interpolated linearly inside a bucket (as Prometheus histogram_quantile)."""
    empty = {f"p{q}": None for q in qs}
    if not new:
        return empty
    old = old or {}
    les = sorted(new)
    counts = []
    for le in les:
        d = new[le] - old.get(le, 0.0)
        counts.append(max(d, 0.0))
    # Cumulative counts must be monotonic; a restart resets them.
    if any(new[le] < old.get(le, 0.0) for le in les):
        counts = [new[le] for le in les]
    total = counts[-1] if counts else 0.0
    if total <= 0:
        return empty
    out = {}
    for q in qs:
        rank = q / 100 * total
        prev_le, prev_c = 0.0, 0.0
        val = None
        for le, c in zip(les, counts):
            if c >= rank:
                if math.isinf(le):
                    val = prev_le  # beyond the last finite bucket
                elif c == prev_c:
                    val = le
                else:
                    val = prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
                break
            prev_le, prev_c = le, c
        out[f"p{q}"] = val
    return out


def histogram_count(new, old) -> float:
    if not new:
        return 0.0
    top = new.get(math.inf, max(new.values()))
    before = (old or {}).get(math.inf, 0.0)
    return top - before if top >= before else top


@dataclass
class Snapshot:
    t: float
    prefill: Scrape
    decode: Scrape
    router: Scrape | None


def _rate(new: Scrape, old: Scrape, name: str, dt: float) -> float | None:
    d = counter_delta(new.values.get(name), old.values.get(name))
    return None if d is None else d / dt


def _gauge(s: Scrape, name: str) -> float | None:
    return s.values.get(name)


def window(new: Snapshot, old: Snapshot, gpus: int | None) -> dict:
    """Every marquee metric over the window between two snapshots."""
    dt = new.t - old.t
    p, p0, d, d0 = new.prefill, old.prefill, new.decode, old.decode
    r, r0 = new.router, old.router

    def rp(name):
        return _rate(p, p0, name, dt)

    def rd(name):
        return _rate(d, d0, name, dt)

    both = merge([p, d]) if p is not d else p
    both0 = merge([p0, d0]) if p0 is not d0 else p0

    def rb(name):
        return _rate(both, both0, name, dt)

    prompt = rp("sglang:prompt_tokens_total")
    cached = rp("sglang:cached_tokens_total")
    output = rd("sglang:generation_tokens_total")
    served = None if prompt is None or output is None else prompt + output
    rec = {
        "t": round(new.t, 3),
        "window_s": round(dt, 3),
        "traffic": {
            "requests_per_s": rd("sglang:num_requests_total"),
            "router_requests_per_s": None
            if r is None
            else _rate(r, r0, "smg_http_requests_total", dt),
            "input_tok_s": prompt,
            "cached_input_tok_s": cached,
            "uncached_input_tok_s": None
            if prompt is None or cached is None
            else prompt - cached,
            "output_tok_s": output,
            "served_tpm": None if served is None else served * 60,
            "served_tpm_per_gpu": None
            if served is None or not gpus
            else served * 60 / gpus,
            "output_share": None if not served else output / served,
        },
        "cache": {
            "prefix_hit": None if not prompt else cached / prompt,
            "prefill_kv_use": _gauge(p, "sglang:token_usage"),
            "decode_kv_use": _gauge(d, "sglang:token_usage"),
            "prefill_kv_used_tokens": _gauge(p, "sglang:kv_used_tokens"),
            "prefill_kv_evictable_tokens": _gauge(p, "sglang:kv_evictable_tokens"),
            "decode_kv_used_tokens": _gauge(d, "sglang:kv_used_tokens"),
            "evicted_tok_s": rb("sglang:evicted_tokens_total"),
            "hicache_host_used_tokens": _gauge(p, "sglang:hicache_host_used_tokens"),
            "hicache_host_total_tokens": _gauge(p, "sglang:hicache_host_total_tokens"),
            "hicache_writeback_tok_s": rb("sglang:hicache_backup_tokens_total"),
            "hicache_loadback_tok_s": rb("sglang:load_back_tokens_total"),
            "hicache_dropped_tok_s": rb("sglang:hicache_dropped_tokens_total"),
        },
        "preemption": {
            "retracted_req_s": rb("sglang:num_retracted_requests_total"),
            "retracted_input_tok_s": rb("sglang:num_retracted_input_tokens_total"),
            "retracted_output_tok_s": rb("sglang:num_retracted_output_tokens_total"),
            "paused_reqs": _gauge(both, "sglang:num_paused_reqs"),
            "aborted_req_s": rb("sglang:num_aborted_requests_total"),
        },
        "queues": {
            "prefill_running": _gauge(p, "sglang:num_running_reqs"),
            "prefill_waiting": _gauge(p, "sglang:num_queue_reqs"),
            "decode_running": _gauge(d, "sglang:num_running_reqs"),
            "decode_waiting": _gauge(d, "sglang:num_queue_reqs"),
            "prefill_bootstrap": _gauge(p, "sglang:num_prefill_bootstrap_queue_reqs"),
            "prefill_inflight": _gauge(p, "sglang:num_prefill_inflight_queue_reqs"),
            "decode_prealloc": _gauge(d, "sglang:num_decode_prealloc_queue_reqs"),
            "decode_transfer": _gauge(d, "sglang:num_decode_transfer_queue_reqs"),
            "router_inflight": None if r is None else _gauge(r, "smg_admission_inflight"),
            "router_rejected_s": None
            if r is None
            else _rate(r, r0, "smg_admission_rejected_total", dt),
        },
        "latency_s": {
            "ttft": histogram_quantiles(
                d.buckets.get("sglang:time_to_first_token_seconds"),
                d0.buckets.get("sglang:time_to_first_token_seconds"),
            ),
            "itl": histogram_quantiles(
                d.buckets.get("sglang:inter_token_latency_seconds"),
                d0.buckets.get("sglang:inter_token_latency_seconds"),
            ),
            "prefill_queue": histogram_quantiles(
                p.buckets.get("sglang:queue_time_seconds"),
                p0.buckets.get("sglang:queue_time_seconds"),
            ),
            "router_request": histogram_quantiles(
                None if r is None else r.buckets.get("smg_http_request_duration_seconds"),
                None if r0 is None else r0.buckets.get("smg_http_request_duration_seconds"),
            ),
        },
        "health": {
            "router_non2xx_s": None if r is None else _non2xx_rate(r, r0, dt),
            "kv_transfer_failed_s": rb("sglang:num_transfer_failed_reqs_total"),
            "bootstrap_failed_s": rb("sglang:num_bootstrap_failed_reqs_total"),
            "prefill_retries_s": rp("sglang:num_prefill_retries_total"),
            "spec_accept_length": _gauge(d, "sglang:spec_accept_length"),
        },
    }
    return rec


def _non2xx_rate(r: Scrape, r0: Scrape, dt: float) -> float | None:
    new = r.labelled.get("smg_http_responses_total")
    if new is None:
        return None
    old = r0.labelled.get("smg_http_responses_total", {})
    total = 0.0
    for key, v in new.items():
        if key[0].startswith("2"):
            continue
        d = counter_delta(v, old.get(key, 0.0))
        total += d or 0.0
    return total / dt


def fmt(v, kind="n"):
    if v is None:
        return "-"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "s":
        return f"{v:.2f}s" if v < 10 else f"{v:.0f}s"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.0f}K"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.2f}"


def router_line(rec: dict) -> str:
    t, q, lat, h = rec["traffic"], rec["queues"], rec["latency_s"], rec["health"]
    rr = lat["router_request"]
    return (
        f"router req/s {fmt(t['router_requests_per_s'])}  non-2xx {fmt(h['router_non2xx_s'])}/s  "
        f"in flight {fmt(q['router_inflight'])}  rejected {fmt(q['router_rejected_s'])}/s  "
        f"duration p50/p99 {fmt(rr['p50'], 's')}/{fmt(rr['p99'], 's')}"
    )


def line(rec: dict) -> str:
    t, c, pr, q, lat = (
        rec["traffic"],
        rec["cache"],
        rec["preemption"],
        rec["queues"],
        rec["latency_s"],
    )
    return (
        f"req/s {fmt(t['requests_per_s'])}  in {fmt(t['input_tok_s'])} tok/s "
        f"(hit {fmt(c['prefix_hit'], 'pct')})  out {fmt(t['output_tok_s'])} tok/s  "
        f"TPM/GPU {fmt(t['served_tpm_per_gpu'])}  "
        f"KV P/D {fmt(c['prefill_kv_use'], 'pct')}/{fmt(c['decode_kv_use'], 'pct')}  "
        f"evict {fmt(c['evicted_tok_s'])}/s  host wb/lb {fmt(c['hicache_writeback_tok_s'])}/"
        f"{fmt(c['hicache_loadback_tok_s'])}  retract {fmt(pr['retracted_req_s'])}/s  "
        f"run/wait P {fmt(q['prefill_running'])}/{fmt(q['prefill_waiting'])} "
        f"D {fmt(q['decode_running'])}/{fmt(q['decode_waiting'])}  "
        f"TTFT p50/p99 {fmt(lat['ttft']['p50'], 's')}/{fmt(lat['ttft']['p99'], 's')}  "
        f"ITL p50 {fmt(None if lat['itl']['p50'] is None else lat['itl']['p50'] * 1000)}ms"
    )


PROM_PREFIX = "inference_"
_PROM_SECTIONS = ("traffic", "cache", "preemption", "queues", "health")
_PROM_LABEL_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _prom_labels(labels: dict) -> str:
    if not labels:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items())) + "}"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def exposition(rec: dict | None, labels: dict | None = None, up: bool = True) -> str:
    """The window record as Prometheus text: one gauge per numeric field
    (inference_<section>_<field>), latency quantiles as
    inference_latency_seconds{stage,quantile}. Missing values are omitted."""
    labels = dict(labels or {})
    out = [
        f"# TYPE {PROM_PREFIX}up gauge",
        f"{PROM_PREFIX}up{_prom_labels(labels)} {1 if up else 0}",
    ]
    if rec is None:
        return "\n".join(out) + "\n"
    out += [
        f"# TYPE {PROM_PREFIX}window_seconds gauge",
        f"{PROM_PREFIX}window_seconds{_prom_labels(labels)} {rec['window_s']}",
        f"# TYPE {PROM_PREFIX}window_end_timestamp_seconds gauge",
        f"{PROM_PREFIX}window_end_timestamp_seconds{_prom_labels(labels)} {rec['t']}",
    ]
    for section in _PROM_SECTIONS:
        for key, v in rec.get(section, {}).items():
            if _num(v):
                name = f"{PROM_PREFIX}{section}_{key}"
                out += [f"# TYPE {name} gauge", f"{name}{_prom_labels(labels)} {v}"]
    lat_name = f"{PROM_PREFIX}latency_seconds"
    lat = [f"# TYPE {lat_name} gauge"]
    for stage, qs in rec.get("latency_s", {}).items():
        for q, v in qs.items():
            if _num(v):
                l = dict(labels, stage=stage, quantile=str(int(q[1:]) / 100))
                lat.append(f"{lat_name}{_prom_labels(l)} {v}")
    if len(lat) > 1:
        out += lat
    return "\n".join(out) + "\n"


class Exporter:
    """Serves the latest exposition at /metrics from a daemon thread."""

    def __init__(self, host: str, port: int, labels: dict):
        self.labels = labels
        self._text = exposition(None, labels, up=False)
        self._lock = threading.Lock()
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] != "/metrics":
                    self.send_error(404)
                    return
                with exporter._lock:
                    body = exporter._text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, rec: dict | None, up: bool) -> None:
        with self._lock:
            if rec is None:
                # Keep the last values, flip inference_up.
                lines = self._text.splitlines()
                key = f"{PROM_PREFIX}up{_prom_labels(self.labels)} "
                self._text = "\n".join(
                    (key + ("1" if up else "0")) if ln.startswith(key) else ln for ln in lines
                ) + "\n"
            else:
                self._text = exposition(rec, self.labels, up)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def parse_labels(items: list[str]) -> dict:
    labels = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not _PROM_LABEL_OK.match(k) or k in ("stage", "quantile"):
            raise ValueError(f"bad --label {item!r}: want name=value (not stage/quantile)")
        labels[k] = v
    return labels


class MetricsUnavailable(Exception):
    """An endpoint answered but serves no /metrics (metrics flag off)."""


def fetch(url: str, timeout: float) -> Scrape:
    full = url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(full, timeout=timeout) as resp:
            return parse(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise MetricsUnavailable(f"{full} returned 404") from None
        raise


class Scraper:
    def __init__(self, prefill, decode, workers, router, timeout, router_only=False):
        if router_only:
            if not router:
                raise ValueError("--router-only needs --router")
            prefill = decode = workers = []
        elif workers and (prefill or decode):
            raise ValueError("--worker (colocated) cannot be combined with --prefill/--decode")
        elif not workers and not (prefill and decode):
            raise ValueError("pass --prefill and --decode (PD) or --worker (colocated)")
        self.prefill, self.decode = prefill or workers, decode or workers
        self.colocated = bool(workers) or router_only
        self.router_only = router_only
        self.router, self.timeout = router, timeout

    def snapshot(self) -> Snapshot:
        t = time.time()
        if self.router_only:
            return Snapshot(t, Scrape(), Scrape(), fetch(self.router, self.timeout))
        p = merge([fetch(u, self.timeout) for u in self.prefill])
        d = p if self.colocated else merge([fetch(u, self.timeout) for u in self.decode])
        r = fetch(self.router, self.timeout) if self.router else None
        return Snapshot(t, p, d, r)


def check_enabled(snap: Snapshot) -> list[str]:
    """Names of required series missing from a scrape (metrics flag off, or a
    different SGLang version)."""
    missing = []
    for role, s, names in (
        ("prefill", snap.prefill, ["sglang:prompt_tokens_total", "sglang:token_usage"]),
        ("decode", snap.decode, ["sglang:generation_tokens_total"]),
    ):
        missing += [f"{role}:{n}" for n in names if n not in s.values]
    return missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prefill", action="append", default=[], help="prefill worker URL (repeatable)")
    p.add_argument("--decode", action="append", default=[], help="decode worker URL (repeatable)")
    p.add_argument("--worker", action="append", default=[], help="colocated worker URL (repeatable)")
    p.add_argument("--router", default="", help="router Prometheus URL, e.g. http://127.0.0.1:29001")
    p.add_argument("--gpus", type=int, default=0, help="physical GPUs serving, for per-GPU TPM")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=0, help="stop after S seconds (0 = until Ctrl-C)")
    p.add_argument("--out", default="", help="append one JSON record per interval here")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument(
        "--router-only",
        action="store_true",
        help="scrape only the router (workers without --enable-metrics): request rate, "
        "status codes, admission and request duration; no token, cache or TTFT metrics",
    )
    p.add_argument("--serve", type=int, default=0, metavar="PORT",
                   help="also serve the latest window as Prometheus gauges on PORT (/metrics)")
    p.add_argument("--serve-host", default="127.0.0.1", help="bind address for --serve")
    p.add_argument("--label", action="append", default=[], metavar="K=V",
                   help="constant label on served gauges, e.g. topology=4xB200-1P1D (repeatable)")
    args = p.parse_args(argv)
    if not args.prefill and not args.decode and not args.worker:
        args.prefill, args.decode = ["http://127.0.0.1:8041"], ["http://127.0.0.1:8042"]
        args.router = args.router or "http://127.0.0.1:29001"

    try:
        scraper = Scraper(
            args.prefill, args.decode, args.worker, args.router, args.timeout, args.router_only
        )
        first = prev = scraper.snapshot()
    except ValueError as e:
        p.error(str(e))
    except MetricsUnavailable as e:
        print(
            f"No worker metrics: {e}. Start the workers with --enable-metrics (the "
            "launcher's --worker-metrics, on by default), or pass --router-only for the "
            "router's view alone.",
            file=sys.stderr,
        )
        return 2
    except OSError as e:
        print(f"Cannot reach a metrics endpoint: {e}", file=sys.stderr)
        return 2
    missing = [] if args.router_only else check_enabled(first)
    if missing:
        print(
            "Missing worker metrics (start the workers with --enable-metrics): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    exporter = None
    if args.serve:
        try:
            exporter = Exporter(args.serve_host, args.serve, parse_labels(args.label))
        except ValueError as e:
            p.error(str(e))
        print(f"serving Prometheus gauges on http://{args.serve_host}:{exporter.port}/metrics",
              file=sys.stderr, flush=True)
    out = open(args.out, "a") if args.out else None
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    end = first.t + args.duration if args.duration else math.inf
    next_at = first.t + args.interval
    while not stop["now"] and time.time() < end:
        deadline = min(next_at, end)
        while not stop["now"] and time.time() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.time())))
        try:
            snap = scraper.snapshot()
        except OSError as e:
            print(f"scrape failed: {e}", file=sys.stderr, flush=True)
            if exporter:
                exporter.update(None, up=False)
            # Retry after a full interval; prev stays the last good scrape.
            next_at = time.time() + args.interval
            continue
        next_at = snap.t + args.interval
        if snap.t - prev.t <= 0:
            continue
        rec = window(snap, prev, args.gpus)
        if exporter:
            exporter.update(rec, up=True)
        text = router_line(rec) if scraper.router_only else line(rec)
        print(f"[{snap.t - first.t:6.0f}s] " + text, flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        prev = snap
    if prev is not first:
        summary = window(prev, first, args.gpus)
        print("SUMMARY " + json.dumps(summary), flush=True)
        if out:
            out.write(json.dumps({"summary": summary}) + "\n")
    if out:
        out.close()
    if exporter:
        exporter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
