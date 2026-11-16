"""Fault injection under load: run a command mid-load, measure the outage and the recovery.

    |-- baseline --| fault-cmd |------ outage ------|--- recovered ---|
                   F           first failure   last failure   R

The command is whatever the operator uses on the cluster, for example
`kubectl delete pod inference-pair-0 --wait=false` (exec into pods is off, so
faults go through the control plane). It runs in a shell with this process's
environment; the suite records its exit code and the last lines of its output,
not the command's secrets.

Recovered (R) = the first check (every 5 s, and not before --settle-s after the
fault) at which the last --window-s seconds had no failed request and at least
--recovered-ratio of the baseline's completions per second. The settle time keeps
a graceful drain right after the fault from counting as a recovery.
A worker reload takes minutes (15-35 for DeepSeek-V4.1), so --recover-timeout-s
defaults to an hour.

Passes when the baseline had no failures and the endpoint recovered in time.
--max-failed-requests caps the requests lost during the outage (default: no
cap; set 0 where a second replica should carry the traffic). --check-cmd runs
after recovery (for example a check that the node's GPUs report no NVRM errors)
and must exit 0.

  python fault.py --url https://GATEWAY --concurrency 16 \\
      --fault-cmd 'kubectl -n inference delete pod inference-pair-0 --wait=false'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import endpoint
import traffic


async def shell(cmd, timeout):
    t = time.time()
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        code = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        out, code = b"", None
    tail = out.decode("utf-8", "replace").strip().splitlines()[-5:]
    return {"exit_code": code, "seconds": round(time.time() - t, 1), "output_tail": tail}


def rate(rows, start, end):
    """ok/s and failures among requests that ended in [start, end)."""
    sel = [r for r in rows if start <= r["end"] < end]
    ok = sum(1 for r in sel if r["cls"] == "ok")
    bad = sum(1 for r in sel if r["cls"] in traffic.FAILURES)
    return ok / max(end - start, 1e-9), bad


def recovered(rows, baseline_ok_s, now, window, ratio):
    """Whether the window ending now was clean: no failures, near-baseline throughput."""
    ok_s, bad = rate(rows, now - window, now)
    return bad == 0 and ok_s >= ratio * baseline_ok_s


def outage(rows, fault_at, recovered_at):
    lost = [r for r in rows if r["cls"] in traffic.FAILURES and fault_at <= r["end"] < (recovered_at or 1e18)]
    return {
        "failed_requests": len(lost),
        "first_failure_after_fault_s": round(lost[0]["end"] - fault_at, 1) if lost else None,
        "last_failure_after_fault_s": round(max(r["end"] for r in lost) - fault_at, 1) if lost else None,
        "classes": traffic.tally(lost)["classes"] if lost else {},
        "first_errors": [r["err"] for r in lost][:5],
    }


async def fault(args):
    t0 = time.time()
    url = args.url.rstrip("/") + "/v1/chat/completions"
    async with endpoint.client_session(args, limit=0) as session:
        load = traffic.Load(session, url, args.model, args.concurrency,
                            make=lambda rnd: traffic.mixed(rnd, args.max_prompt_tokens, traffic.samplers_without(args.unsupported_params)),
                            seed=args.seed, timeout=args.request_timeout, t0=t0)
        task = asyncio.create_task(load.run())
        await asyncio.sleep(args.baseline_s)
        fault_at = time.time() - t0
        base_ok_s, base_bad = rate(load.rows, 0, fault_at)
        print(f"baseline {base_ok_s:.2f} ok/s, {base_bad} failures; running the fault command", flush=True)
        cmd = await shell(args.fault_cmd, args.cmd_timeout_s)
        print(f"fault command exit {cmd['exit_code']} after {cmd['seconds']} s", flush=True)
        recovered_at = None
        while time.time() - t0 < fault_at + args.recover_timeout_s:
            await asyncio.sleep(5)
            now = time.time() - t0
            if now - fault_at >= max(args.settle_s, args.window_s) and recovered(
                load.rows, base_ok_s, now, args.window_s, args.recovered_ratio
            ):
                recovered_at = now
                break
            ok_s, bad = rate(load.rows, now - args.window_s, now)
            print(f"[{now - fault_at:6.0f} s after fault] {ok_s:.2f} ok/s, {bad} failures in the last "
                  f"{args.window_s:.0f} s", flush=True)
        await asyncio.sleep(args.after_s)
        load.stop()
        await task
    check = await shell(args.check_cmd, args.cmd_timeout_s) if args.check_cmd else None
    return load.rows, base_ok_s, base_bad, fault_at, recovered_at, cmd, check


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    endpoint.add_endpoint_args(p)
    p.add_argument("--fault-cmd", required=True)
    p.add_argument("--check-cmd", default="", help="run after recovery; must exit 0")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-prompt-tokens", type=int, default=24000)
    p.add_argument("--baseline-s", type=float, default=120)
    p.add_argument("--window-s", type=float, default=60)
    p.add_argument("--settle-s", type=float, default=120, help="no recovery is declared before this")
    p.add_argument("--recovered-ratio", type=float, default=0.9)
    p.add_argument("--recover-timeout-s", type=float, default=3600)
    p.add_argument("--after-s", type=float, default=30, help="load kept on after recovery")
    p.add_argument("--max-failed-requests", type=int, default=-1, help="-1 = no cap")
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--cmd-timeout-s", type=float, default=600)
    p.add_argument(
        "--unsupported-params",
        type=lambda t: [x for x in t.split(",") if x],
        default=[],
        help="sampling parameters the deployment rejects, left out of the mix (min_p with DSpark)",
    )
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="fault.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    endpoint.raise_nofile(args.concurrency)
    endpoint.resolve_model(args)
    rows, base_ok_s, base_bad, fault_at, recovered_at, cmd, check = asyncio.run(fault(args))
    lost = outage(rows, fault_at, recovered_at)
    gates = []
    if base_bad:
        gates.append(f"{base_bad} failures before the fault (baseline must be clean)")
    if not base_ok_s:
        gates.append("no completed requests in the baseline")
    if recovered_at is None:
        gates.append(f"not recovered within {args.recover_timeout_s:.0f} s")
    if 0 <= args.max_failed_requests < lost["failed_requests"]:
        gates.append(f"{lost['failed_requests']} requests failed (max {args.max_failed_requests})")
    if check and check["exit_code"] != 0:
        gates.append(f"check command exit {check['exit_code']}: {check['output_tail'][-1:]}")
    result = {
        "endpoint": endpoint.describe(args),
        "concurrency": args.concurrency,
        "baseline_ok_per_s": round(base_ok_s, 3),
        "fault_at_s": round(fault_at, 1),
        "fault_command": cmd,
        "recovered_after_s": round(recovered_at - fault_at, 1) if recovered_at is not None else None,
        "outage": lost,
        "check_command": check,
        "total": traffic.tally(rows),
        "gate_failures": gates,
        "passed": not gates,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if result['passed'] else 'FAIL'}  fault: recovered after {result['recovered_after_s']} s, "
          f"{lost['failed_requests']} requests failed -> {args.out}")
    for g in gates:
        print("  gate: " + g)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
