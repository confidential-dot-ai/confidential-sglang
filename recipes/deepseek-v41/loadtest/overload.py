"""Overload: offer more load than the endpoint can serve and check it sheds it cleanly.

Steps the number of concurrent clients (--levels) for --step-s each, then sends
a burst of --burst requests at once, then checks recovery. Each client sends
short requests back to back, so offered load climbs with the level until the
router's admission cap or the gateway's key limits refuse the excess.

Passes when, at every level and in the burst:
  - no 5xx, no connection errors and no wrong answers (429 is the correct refusal);
  - and afterwards a request succeeds within --recover-s (no wedged router).
Timeouts are reported and fail only with --fail-on-timeout: a queued request
can legitimately wait past a short client timeout.

`shed` in the result says whether any step was refused with 429. If not, the
endpoint queued everything and the refusal path was never exercised: raise
--levels and --burst until it is, or pass --require-shed to fail such a run.

  python overload.py --url https://GATEWAY --levels 64,128,256,512 --burst 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import endpoint
import traffic

HARD = ("5xx", "conn", "bad", "4xx")


async def level(session, url, model, clients, seconds, seed, timeout, t0):
    load = traffic.Load(session, url, model, clients, make=traffic.short, seed=seed,
                        timeout=timeout, t0=t0, end=time.time() + seconds)
    return await load.run()


async def burst(session, url, model, n, seed, timeout, t0):
    rnd = random.Random(seed)
    jobs = [traffic.send(session, url, model, *traffic.short(rnd), rnd, t0, timeout) for _ in range(n)]
    return await asyncio.gather(*jobs)


async def recover(session, url, model, seconds, t0):
    """Seconds until one short request succeeds, or None."""
    rnd, start = random.Random(0), time.time()
    while time.time() - start < seconds:
        row = await traffic.send(session, url, model, *traffic.short(rnd), rnd, t0, 60)
        if row["cls"] == "ok":
            return round(time.time() - start, 1)
        await asyncio.sleep(2)
    return None


async def overload(args):
    t0 = time.time()
    url = args.url.rstrip("/") + "/v1/chat/completions"
    steps = []
    async with endpoint.client_session(args, limit=0) as session:
        for i, clients in enumerate(args.levels):
            rows = await level(session, url, args.model, clients, args.step_s, i, args.request_timeout, t0)
            steps.append(_step(f"{clients} clients", rows, args.step_s))
            print(_line(steps[-1]), flush=True)
        if args.burst:
            rows = await burst(session, url, args.model, args.burst, 99, args.request_timeout, t0)
            span = max(r["end"] for r in rows) - min(r["start"] for r in rows)
            steps.append(_step(f"burst {args.burst}", rows, span))
            print(_line(steps[-1]), flush=True)
        recovered = await recover(session, url, args.model, args.recover_s, t0)
    return steps, recovered


def _step(name, rows, seconds):
    t = traffic.tally(rows)
    return {"step": name, "seconds": round(seconds, 1), "ok_per_s": round(t["classes"]["ok"] / max(seconds, 1e-9), 2),
            **t}


def _line(s):
    c = s["classes"]
    return (f"{s['step']:>14}: ok {c['ok']} ({s['ok_per_s']}/s)  429 {c['429']}  5xx {c['5xx']}  conn {c['conn']}  "
            f"timeout {c['timeout']}  bad {c['bad']}  4xx {c['4xx']}  p50 {s['latency_s'][50]} s")


def gates(steps, recovered, args):
    out = []
    for s in steps:
        hard = {k: s["classes"][k] for k in HARD if s["classes"][k]}
        if args.fail_on_timeout and s["classes"]["timeout"]:
            hard["timeout"] = s["classes"]["timeout"]
        if hard:
            out.append(f"{s['step']}: {hard}")
    if recovered is None:
        out.append(f"no successful request within {args.recover_s} s after the overload")
    if args.require_shed and not any(s["classes"]["429"] for s in steps):
        out.append("no 429 at any step: the endpoint was not overloaded (raise --levels / --burst)")
    return out


def ints(text):
    return [int(x) for x in text.split(",") if x]


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    endpoint.add_endpoint_args(p)
    p.add_argument("--levels", type=ints, default=ints("64,128,256,512"), help="concurrent clients per step")
    p.add_argument("--step-s", type=float, default=60)
    p.add_argument("--burst", type=int, default=2000, help="requests sent at once after the steps (0 = none)")
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--recover-s", type=float, default=120)
    p.add_argument("--fail-on-timeout", action="store_true")
    p.add_argument("--require-shed", action="store_true", help="fail unless some step got 429s")
    p.add_argument("--out", default="overload.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    endpoint.resolve_model(args)
    steps, recovered = asyncio.run(overload(args))
    failures = gates(steps, recovered, args)
    shed = any(s["classes"]["429"] for s in steps)
    result = {"endpoint": endpoint.describe(args), "steps": steps, "recovered_after_s": recovered, "shed": shed,
              "gate_failures": failures, "passed": not failures}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if result['passed'] else 'FAIL'}  overload: recovered after {recovered} s, "
          f"{'shed with 429' if shed else 'NOT saturated (no 429 at any step)'} -> {args.out}")
    for g in failures:
        print("  gate: " + g)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
