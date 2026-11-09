"""Oracle scorecard: a fixed workload, a fixed measurement window, pass or fail.

Replays an Oracle dataset (oracle_replay.py, Oracle decoding on by default) and
measures one window after a warmup:

    |-- warmup --|------------- measure -------------|-- drain --|
                 A                                   B

Server counters (live_metrics.py) are read at A and B, so token rates, cache
hit, KV pressure, evictions, HiCache traffic and retractions cover exactly the
window. Client latencies (TTFT, step latency, TPOT) come from the steps that
finished inside it. No new steps start after B; in-flight steps get --drain-s
to finish and are then cancelled.

The result fails unless, inside the window:
  - no request failed (non-200 other than 429) and no KV transfer failed;
  - the dataset was still running at B (a window cut short is invalid);
  - the prefix-cache hit and output share are inside the dataset's bands;
  - TTFT p50 is at most --ttft-p50-max (default 3 s);
  - no request was retracted (each retraction re-runs a prefill).
--allow-failure keeps the exit code 0 for calibration runs; the JSON still
lists every failed gate.

The physical GPU count, GPU model and topology are required: served TPM per GPU
is only comparable when its denominator is explicit.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request

import live_metrics as lm
import oracle_replay as replay


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def band(text):
    lo, hi = (float(x) for x in text.split(","))
    if not 0 <= lo <= hi <= 1:
        raise argparse.ArgumentTypeError(f"band must be lo,hi in [0, 1], got {text}")
    return lo, hi


def server_info(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/server_info", timeout=10) as r:
            info = json.loads(r.read())
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    keep = (
        "model_path", "tp_size", "ep_size", "dp_size", "context_length", "chunked_prefill_size",
        "max_total_tokens", "max_running_requests", "disaggregation_mode", "speculative_algorithm",
        "enable_hierarchical_cache", "hicache_ratio", "hicache_size", "page_size", "version",
    )
    args = info.get("server_args", info) if isinstance(info, dict) else {}
    return {k: args.get(k, info.get(k)) for k in keep if k in args or k in info}


def gates(summary, args):
    failures = []
    w, c = summary["window"], summary["client"]
    if not summary["valid_window"]:
        failures.append("traffic ended before the window closed (window invalid)")
    if c["failed"]:
        failures.append(f"failed requests={c['failed']} (required 0)")
    transfer = w["health"]["kv_transfer_failed_s"]
    if transfer:
        failures.append(f"KV transfer failures {transfer * w['window_s']:.0f} (required 0)")
    hit = w["cache"]["prefix_hit"]
    if args.cache_hit_band and (hit is None or not args.cache_hit_band[0] <= hit <= args.cache_hit_band[1]):
        failures.append(f"prefix_hit={hit} outside {list(args.cache_hit_band)}")
    share = w["traffic"]["output_share"]
    if args.output_share_band and (
        share is None or not args.output_share_band[0] <= share <= args.output_share_band[1]
    ):
        failures.append(f"output_share={share} outside {list(args.output_share_band)}")
    ttft = c["ttft_s"][50]
    if ttft is None or ttft != ttft or ttft > args.ttft_p50_max:
        failures.append(f"client TTFT p50={ttft} s (max {args.ttft_p50_max})")
    retracted = w["preemption"]["retracted_req_s"]
    if retracted:
        failures.append(f"retractions {retracted * w['window_s']:.0f} (required 0)")
    return failures


async def run(args):
    if args.oracle_off:
        args.oracle = False
    convs = replay.load(args)
    scraper = lm.Scraper(args.prefill, args.decode, args.worker, args.router, 10.0)
    first = scraper.snapshot()
    missing = lm.check_enabled(first)
    if missing:
        raise SystemExit("worker metrics missing (start with --enable-metrics): " + ", ".join(missing))

    stats = replay.Stats()
    args.time_limit = args.warmup_s + args.measure_s
    task = asyncio.create_task(replay.replay(args, convs, stats))
    loop = asyncio.get_running_loop()

    async def snapshot_at(offset):
        await asyncio.sleep(max(0.0, stats.t0 + offset - time.time()))
        if task.done():
            return None, 0
        snap = await loop.run_in_executor(None, scraper.snapshot)
        return snap, stats.active_sessions

    snap_a, _ = await snapshot_at(args.warmup_s)
    snap_b, active_at_b = await snapshot_at(args.warmup_s + args.measure_s)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=args.drain_s)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    if task.done() and not task.cancelled() and task.exception():
        raise task.exception()
    if snap_a is None or snap_b is None:
        raise SystemExit("the replay finished before the measurement window closed; use more sessions or a shorter window")

    a, b = snap_a.t - stats.t0, snap_b.t - stats.t0
    rows = [r for r in stats.rows if a <= r["end"] <= b]
    client = replay.summarize(rows, b - a, convs, stats)
    client["failed"] = sum(1 for t in stats.error_times if a <= t <= b)
    client.pop("errors")
    client["errors_whole_run"] = stats.errors
    window = lm.window(snap_b, snap_a, args.gpus)
    summary = {
        "window": window,
        "client": client,
        "valid_window": active_at_b > 0,
        "active_sessions_at_close": active_at_b,
    }
    summary["gate_failures"] = gates(summary, args)
    summary["passed"] = not summary["gate_failures"]
    return summary


def main(argv=None):
    p = replay.parser()
    p.description = __doc__.split("\n\n")[0]
    p.add_argument("--oracle-off", action="store_true", help="replay without forced tokens (not scorecard-comparable)")
    p.add_argument("--prefill", action="append", default=[])
    p.add_argument("--decode", action="append", default=[])
    p.add_argument("--worker", action="append", default=[])
    p.add_argument("--router", default="")
    p.add_argument("--gpus", type=int, required=True, help="physical GPUs serving the endpoint")
    p.add_argument("--gpu-model", required=True)
    p.add_argument("--topology", required=True, help='e.g. "1 node; 2P TP2 + 2D TP2; GPUs 0-7"')
    p.add_argument("--image", default="", help="image digest the workers run")
    p.add_argument("--warmup-s", type=float, default=120)
    p.add_argument("--measure-s", type=float, default=600)
    p.add_argument("--drain-s", type=float, default=300)
    p.add_argument("--cache-hit-band", type=band, default=None, help="lo,hi; required unless --allow-failure")
    p.add_argument("--output-share-band", type=band, default=None, help="lo,hi; required unless --allow-failure")
    p.add_argument("--ttft-p50-max", type=float, default=3.0)
    p.add_argument("--allow-failure", action="store_true")
    p.add_argument("--out", required=True, help="result JSON")
    p.set_defaults(oracle=True)
    args = p.parse_args(argv)
    if not args.prefill and not args.decode and not args.worker:
        args.prefill, args.decode = ["http://127.0.0.1:8041"], ["http://127.0.0.1:8042"]
        args.router = args.router or "http://127.0.0.1:29001"
    if not args.allow_failure and not (args.cache_hit_band and args.output_share_band):
        p.error("--cache-hit-band and --output-share-band are required (or --allow-failure to calibrate)")

    started = time.time()
    summary = asyncio.run(run(args))
    with open(args.dataset) as f:
        head = f.read(1 << 16)
    workers = {u: server_info(u) for u in (args.prefill + args.decode + args.worker)}
    result = {
        "started": started,
        "args": {k: v for k, v in vars(args).items() if k != "out"},
        "provenance": {
            "dataset_sha256": sha256(args.dataset),
            "dataset_metadata_head": head[:2000],
            "physical_gpus": args.gpus,
            "gpu_model": args.gpu_model,
            "topology": args.topology,
            "image": args.image,
            "workers": workers,
            "host": platform.node(),
            "git": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=sys.path[0] or "."
            ).stdout.strip(),
        },
        **summary,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    w, c = summary["window"], summary["client"]
    print(
        f"{'PASS' if summary['passed'] else 'FAIL'}  served TPM/GPU {lm.fmt(w['traffic']['served_tpm_per_gpu'])}  "
        f"hit {lm.fmt(w['cache']['prefix_hit'], 'pct')}  out share {lm.fmt(w['traffic']['output_share'], 'pct')}  "
        f"TTFT p50/p99 {c['ttft_s'][50]}/{c['ttft_s'][99]} s  steps/s {c['steps_per_s']}  -> {args.out}"
    )
    for f in summary["gate_failures"]:
        print("  gate: " + f)
    return 0 if summary["passed"] or args.allow_failure else 1


if __name__ == "__main__":
    sys.exit(main())
