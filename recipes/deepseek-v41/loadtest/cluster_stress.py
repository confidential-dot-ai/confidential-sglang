"""Stress-test a candidate deployment from outside, through its gateway, and write one report.

Phases (--phases, in this order):

  func      functional battery: tools, API features, long context, limits  (func_tests.py)
  needles   long-prompt recall, fixed cases; any miss fails                  (needles.py)
  load      realmix agent traces at a fixed session count, scorecard gates  (oracle_scorecard.py)
  overload  past capacity: 429s only, no 5xx, recovers                       (overload.py)
  soak      mixed traffic for a long time; no failures, no drift            (soak.py)
  fault     a control-plane fault mid-load; time to recover                 (fault.py, needs --fault-cmd)

Every phase writes its own JSON into --out-dir; this script adds report.json and
report.md (a table to paste into a PR or a release note) and exits 1 if any
phase failed. The API key comes from the environment (--api-key-env, default
STRESS_API_KEY) and is never written anywhere.

  export STRESS_API_KEY=...            # a test key with raised rate limits
  python cluster_stress.py --url https://GATEWAY --context-length 131072 \\
      --dataset oracle-realmix-...json --gpus 8 --gpu-model B200 --topology "2 pairs, 1P1D TP2" \\
      --prom-url https://PROMETHEUS --prom-worker 'environment="candidate",job="inference-workers"' \\
      --fault-cmd 'kubectl -n inference delete pod inference-pair-0 --wait=false'

Extra options for one phase go in --func-args, --needles-args, --load-args,
--overload-args, --soak-args and --fault-args (one shell-quoted string each).

--profile sets the defaults for one kind of deployment (PROFILES); any option
given on the command line still wins:

  qwen-short   1-GPU Qwen3.5-0.8B staging worker, ~15 minutes: func (needles up
               to 262144 - 8192 tokens), needles to 240K, overload at 16-128
               clients, a 5-minute soak. No load or fault phase.
  dsv41-full   DeepSeek-V4.1 PD: every phase, 131072 context, n=2 and min_p
               known to fail on PD with DSpark, a 4-hour soak.

Without --profile the defaults are the tools' own (the 2026-11-16 4x B200 run).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import endpoint
import prom_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ("func", "needles", "load", "overload", "soak", "fault")

# Per-deployment defaults (argparse dests). Explicit options override them.
PROFILES = {
    # Qwen3.5-0.8B on one GPU (recipes/qwen35-0.8b): aggregated, no speculative
    # decoding, so logprobs are served (t_logprobs_rejected expects a 4xx only
    # under DSpark) and n=2 works. Needles stay inside the 262144-token context
    # with room for the answer and the chat template: func's largest needle is
    # context - 8192, the needles phase stops at 240000 and calibrates on the
    # served tokenizer. The realmix load phase is a DeepSeek-V4.1 dataset, so it
    # is left out.
    "qwen-short": {
        "phases": "func,needles,overload,soak",
        "context_length": 262144,
        "sessions": 8,
        "known_failures": "t_logprobs_rejected",
        "soak_minutes": 5,
        "func_args": "--trials 4 --concurrency 8",
        "needles_args": "--sizes 16000,64000,128000,192000,240000 --tokens-per-sentence 0",
        "overload_args": "--levels 16,32,64,128 --step-s 30 --burst 500 --recover-s 120",
        "soak_args": "--max-prompt-tokens 48000",
    },
    # DeepSeek-V4.1 PD with DSpark (CLUSTER-STRESS.md "Choices to make").
    "dsv41-full": {
        "phases": ",".join(PHASES),
        "context_length": 131072,
        "sessions": 32,
        "known_failures": "t_n2",
        "unsupported_params": "min_p",
        "soak_minutes": 240,
    },
}


def endpoint_argv(args):
    out = ["--url", args.url, "--model", args.model, "--api-key-env", args.api_key_env,
           "--thinking-kwarg", args.thinking_kwarg, "--thinking-default", args.thinking_default]
    return out + (["--ca-cert", args.ca_cert] if args.ca_cert else [])


def prom_argv(args):
    if not args.prom_url:
        return []
    out = ["--prom-url", args.prom_url, "--prom-lag-s", str(args.prom_lag_s)]
    for flag in ("prefill", "decode", "worker", "router", "token_env"):
        v = getattr(args, f"prom_{flag}")
        if v:
            out += [f"--prom-{flag.replace('_', '-')}", v]
    return out


def direct_metrics_argv(args):
    """Worker and router /metrics URLs, for a deployment whose ports are reachable."""
    out = []
    for flag in ("prefill", "decode", "worker"):
        for u in getattr(args, f"metrics_{flag}"):
            out += [f"--{flag}", u]
    return out + (["--router", args.metrics_router] if args.metrics_router else [])


def phase_argv(name, args, out):
    """The command line of one phase, or a reason it is skipped."""
    ep = endpoint_argv(args)
    extra = shlex.split(getattr(args, f"{name}_args"))
    if name == "func":
        known = ["--known-failures", args.known_failures] if args.known_failures else []
        return ["func_tests.py", *ep, "--context-length", str(args.context_length), *known,
                "--out", out, "--out-jsonl", out.replace(".json", ".jsonl"), *extra]
    if name == "needles":
        return ["needles.py", *ep, "--out", out, *extra]
    if name == "load":
        if not args.dataset:
            return "no --dataset"
        bands = []
        if args.cache_hit_band and args.output_share_band:
            bands = ["--cache-hit-band", args.cache_hit_band, "--output-share-band", args.output_share_band]
        else:
            bands = ["--allow-failure"]
        oracle = ["--source-tokenizer", args.source_tokenizer] if args.oracle else ["--oracle-off"]
        metrics = prom_argv(args) or direct_metrics_argv(args)
        return ["oracle_scorecard.py", args.dataset, *ep, "--concurrency", str(args.sessions),
                "--gpus", str(args.gpus), "--gpu-model", args.gpu_model, "--topology", args.topology,
                "--image", args.image, *bands, *oracle, *metrics,
                "--output", out.replace(".json", ".jsonl"), "--out", out, *extra]
    if name == "overload":
        return ["overload.py", *ep, "--out", out, *extra]
    unsupported = ["--unsupported-params", args.unsupported_params] if args.unsupported_params else []
    if name == "soak":
        return ["soak.py", *ep, *unsupported, "--minutes", str(args.soak_minutes), "--concurrency", str(args.sessions),
                "--max-prompt-tokens", str(min(80000, args.context_length - 8192)), "--out", out, *extra]
    if name == "fault":
        if not args.fault_cmd:
            return "no --fault-cmd"
        check = ["--check-cmd", args.check_cmd] if args.check_cmd else []
        return ["fault.py", *ep, *unsupported, "--fault-cmd", args.fault_cmd, *check, "--out", out, *extra]
    raise ValueError(name)


def run_phase(name, args):
    out = os.path.join(args.out_dir, f"{name}.json")
    argv = phase_argv(name, args, out)
    if isinstance(argv, str):
        print(f"\n######## {name}: skipped ({argv})", flush=True)
        return {"phase": name, "status": "skipped", "reason": argv}
    print(f"\n######## {name}  {time.strftime('%H:%M:%S')}", flush=True)
    t = time.time()
    log = os.path.join(args.out_dir, f"{name}.log")
    with open(log, "w") as f:
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, argv[0]), *argv[1:]],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
    result = {"phase": name, "exit_code": code, "seconds": round(time.time() - t), "log": log}
    try:
        with open(out) as f:
            result["result"] = json.load(f)
        result["status"] = "pass" if result["result"].get("passed") else "fail"
    except (OSError, ValueError):
        result["status"] = "error"
    return result


def headline(r):
    """One line of the phase's key numbers for report.md."""
    x = r.get("result") or {}
    name = r["phase"]
    if r["status"] in ("skipped", "error"):
        return r.get("reason") or f"no result (exit {r.get('exit_code')}, see {os.path.basename(r.get('log', ''))})"
    if name == "func":
        return f"{x['passed_cases']}/{x['cases']} cases; known failures {x['known_failures'] or 'none'}"
    if name == "needles":
        return f"{x['misses']} missed, {x['errors']} errors of {x['cases']} (up to {x['max_prompt_tokens']} tokens)"
    if name == "load":
        w, c = x["window"], x["client"]
        return (f"{w['traffic']['served_tpm_per_gpu'] or 0:,.0f} TPM/GPU, hit {w['cache']['prefix_hit']}, "
                f"TTFT p50/p99 {c['ttft_s']['50']}/{c['ttft_s']['99']} s, {c['steps_per_s']} steps/s")
    if name == "overload":
        top = x["steps"][-1] if x["steps"] else {}
        shed = "shed with 429" if x.get("shed") else "NOT saturated: no 429, raise --levels/--burst"
        return (f"{shed}; recovered after {x['recovered_after_s']} s; last step {top.get('step')}: "
                f"{top.get('ok_per_s')} ok/s, p50 {top.get('latency_s', {}).get('50')} s")
    if name == "soak":
        t = x["total"]
        return (f"{t['requests']} requests, {t['failures']} failed, {t['classes']['429']} x 429, "
                f"p50/p99 {t['latency_s']['50']}/{t['latency_s']['99']} s")
    if name == "fault":
        return f"recovered after {x['recovered_after_s']} s, {x['outage']['failed_requests']} requests failed"
    return ""


def report_md(args, results):
    lines = [
        f"# Cluster stress: {args.url}",
        "",
        f"Model `{args.model}`, context {args.context_length}, {args.sessions} sessions, "
        f"profile {getattr(args, 'profile', None) or 'none'}; "
        f"started {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(args.started))}.",
        "",
        "| Phase | Result | Key numbers |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['phase']} | {r['status'].upper()} | {headline(r)} |")
    gates = [(r["phase"], g) for r in results for g in (r.get("result") or {}).get("gate_failures", [])]
    if gates:
        lines += ["", "Failed gates:", ""] + [f"- {p}: {g}" for p, g in gates]
    return "\n".join(lines) + "\n"


def parser(profile=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), default=None,
                   help="defaults for one kind of deployment (see the module docstring)")
    endpoint.add_endpoint_args(p)
    p.add_argument("--phases", default=",".join(PHASES))
    p.add_argument("--out-dir", default="")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--context-length", type=int, default=131072)
    p.add_argument("--sessions", type=int, default=32, help="load and soak concurrency (the operating point)")
    p.add_argument("--known-failures", default="", help="func cases allowed to fail, e.g. t_n2 on PD")
    p.add_argument("--unsupported-params", default="", help="sampling parameters the deployment rejects (min_p with DSpark)")
    p.add_argument("--dataset", default=os.environ.get("STRESS_DATASET", ""),
                   help="Oracle dataset for the load phase (default $STRESS_DATASET)")
    p.add_argument("--oracle", action="store_true", help="force completions (workers must run --oracle-decode)")
    p.add_argument("--source-tokenizer", default=os.environ.get("STRESS_SOURCE_TOKENIZER", ""),
                   help="the dataset's tokenizer, with --oracle (default $STRESS_SOURCE_TOKENIZER)")
    p.add_argument("--cache-hit-band", default="")
    p.add_argument("--output-share-band", default="")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--gpu-model", default="")
    p.add_argument("--topology", default="")
    p.add_argument("--image", default="")
    p.add_argument("--soak-minutes", type=float, default=60)
    p.add_argument("--fault-cmd", default="")
    p.add_argument("--check-cmd", default="")
    prom_metrics.add_args(p)
    for flag in ("prefill", "decode", "worker"):
        p.add_argument(f"--metrics-{flag}", action="append", default=[], help=f"{flag} /metrics base URL (direct)")
    p.add_argument("--metrics-router", default="", help="router metrics base URL (direct)")
    for name in PHASES:
        p.add_argument(f"--{name}-args", default="", help=f"extra options for the {name} phase")
    if profile:
        p.set_defaults(**PROFILES[profile])
    return p


def parse_args(argv=None):
    """Parse twice: --profile first, then everything with that profile's defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES), default=None)
    known, _ = pre.parse_known_args(argv)
    return parser(known.profile).parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.started = time.time()
    phases = [x for x in args.phases.split(",") if x]
    bad = [x for x in phases if x not in PHASES]
    if bad:
        raise SystemExit(f"unknown phases {bad}; choose from {PHASES}")
    if "load" in phases and args.dataset and not (args.gpus and args.gpu_model and args.topology):
        raise SystemExit("the load phase needs --gpus, --gpu-model and --topology")
    if "load" in phases and args.dataset and not (args.prom_url or direct_metrics_argv(args)):
        raise SystemExit("the load phase needs server counters: --prom-url on a cluster, "
                         "or --metrics-worker / --metrics-prefill + --metrics-decode URLs")
    endpoint.resolve_model(args)
    args.out_dir = args.out_dir or time.strftime("stress-%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"cluster stress: {json.dumps(endpoint.describe(args))} phases {phases} -> {args.out_dir}", flush=True)
    results = []
    for name in phases:
        results.append(run_phase(name, args))
        if args.stop_on_fail and results[-1]["status"] in ("fail", "error"):
            break
    passed = all(r["status"] in ("pass", "skipped") for r in results)
    meta = {"endpoint": endpoint.describe(args), "profile": args.profile, "started": args.started,
            "phases": phases, "passed": passed}
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({**meta, "results": results}, f, indent=1)
    md = report_md(args, results)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
