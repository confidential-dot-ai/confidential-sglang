"""Soak: the mixed traffic of traffic.py at a fixed concurrency for a long time.

Passes when no request failed (429s are counted, not failed) and the endpoint
still answers afterwards. Per-minute counts and latencies go into the result, so
a slow leak shows up as a trend: latency or 429s climbing minute over minute.

  python soak.py --url https://GATEWAY --minutes 240 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import endpoint
import traffic


def by_minute(rows, minutes):
    out = []
    for m in range(int(minutes) + 1):
        sel = [r for r in rows if m * 60 <= r["end"] < (m + 1) * 60]
        if sel:
            t = traffic.tally(sel)
            out.append({"minute": m, "requests": t["requests"], "failures": t["failures"],
                        "429": t["classes"]["429"], "latency_p50_s": t["latency_s"][50],
                        "latency_p99_s": t["latency_s"][99]})
    return out


async def soak(args):
    t0 = time.time()
    url = args.url.rstrip("/") + "/v1/chat/completions"
    async with endpoint.client_session(args, limit=0) as session:
        load = traffic.Load(
            session, url, args.model, args.concurrency,
            make=lambda rnd: traffic.mixed(rnd, args.max_prompt_tokens, traffic.samplers_without(args.unsupported_params)),
            seed=args.seed, timeout=args.request_timeout, t0=t0, end=t0 + args.minutes * 60,
        )

        async def ticker():
            while time.time() < load.end:
                await asyncio.sleep(min(60, max(1, load.end - time.time())))
                t = traffic.tally(load.rows)
                print(f"[{time.strftime('%H:%M:%S')}] requests {t['requests']}  failures {t['failures']}  "
                      f"429 {t['classes']['429']}  p50 {t['latency_s'][50]} s", flush=True)

        rows, _ = await asyncio.gather(load.run(), ticker())
        after = await traffic.send(session, url, args.model, *traffic.short(load.rnd), load.rnd, t0, 120)
    return rows, after


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    endpoint.add_endpoint_args(p)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-prompt-tokens", type=int, default=80000, help="largest long prompt in the mix")
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--max-failures", type=int, default=0, help="failed requests allowed, answers judged bad excluded")
    # A small model can miss a tool call at temperature 1.0; that is the model, not the serving stack.
    p.add_argument("--max-bad", type=int, default=0, help="answers judged bad (no tool call, malformed) allowed")
    p.add_argument(
        "--unsupported-params",
        type=lambda t: [x for x in t.split(",") if x],
        default=[],
        help="sampling parameters the deployment rejects, left out of the mix (min_p with DSpark)",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="soak.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    endpoint.raise_nofile(args.concurrency)
    endpoint.resolve_model(args)
    rows, after = asyncio.run(soak(args))
    total = traffic.tally(rows)
    kinds = {k: traffic.tally([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})}
    gates = []
    bad = total["classes"].get("bad", 0)
    if total["failures"] - bad > args.max_failures:
        gates.append(f"failures {total['failures'] - bad} (max {args.max_failures})")
    if bad > args.max_bad:
        gates.append(f"bad answers {bad} (max {args.max_bad})")
    if after["cls"] != "ok":
        gates.append(f"endpoint unhealthy after the soak: {after['cls']} {after['err']}")
    result = {
        "endpoint": endpoint.describe(args),
        "minutes": args.minutes,
        "concurrency": args.concurrency,
        "total": total,
        "by_kind": {k: {"requests": v["requests"], "failures": v["failures"], "latency_s": v["latency_s"]}
                    for k, v in kinds.items()},
        "by_minute": by_minute(rows, args.minutes),
        "healthy_after": after["cls"] == "ok",
        "gate_failures": gates,
        "passed": not gates,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if result['passed'] else 'FAIL'}  soak {args.minutes} min at {args.concurrency}: "
          f"{total['requests']} requests, {total['failures']} failed, {total['classes']['429']} x 429, "
          f"p50/p99 {total['latency_s'][50]}/{total['latency_s'][99]} s -> {args.out}")
    for g in gates:
        print("  gate: " + g)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
