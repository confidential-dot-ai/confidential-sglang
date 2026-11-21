"""Long-prompt recall: hide a code in filler text, ask for it back, fail on any miss.

Each case is fixed by (target tokens, depth, seed), so a miss repeats on every run
and every deployment: it is a correctness bug, not noise. This test found the
CED tail-graph bug (2-5 of 32 prompts wrong up to 60K tokens) that a functional
suite and a load test both passed.

  python needles.py --url https://GATEWAY                      # the 32-case set, x2 seeds
  python needles.py --url https://GATEWAY --sizes 64000,120000,250000 --tokens-per-sentence 0

The defaults are the exact cases of the 4x B200 runs (16 sizes from 14K to 60K
tokens, depths 0.1 and 0.5, seeds 1000 and 4000, 20.19 tokens per sentence on
the DeepSeek-V4.1 tokenizer), so results compare across deployments.
--tokens-per-sentence 0 calibrates on the served tokenizer instead (other models,
or sizes that must land near the context limit).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import endpoint

PHRASES = [
    "The committee reviewed the quarterly logistics report and noted minor delays in the northern routes.",
    "Engineers replaced the coolant pumps on the third floor after the scheduled inspection.",
    "A light rain fell over the harbor while the ferries continued their usual timetable.",
    "The library extended its weekend hours to accommodate students preparing for exams.",
    "Market analysts expected stable demand for industrial equipment through the next season.",
    "Volunteers planted two hundred saplings along the river trail on Saturday morning.",
    "The software team migrated the billing service to a new cluster without downtime.",
    "Local bakeries reported higher sales of rye bread during the cold spell.",
]
QUESTION = "\n\nQuestion: What is the secret vault access code mentioned in the text above? Reply with only the code."
DEFAULT_SIZES = "14000,17000,20000,23000,26000,29000,32000,35000,38000,41000,44000,47000,50000,53000,56000,60000"
DSV41_TOKENS_PER_SENTENCE = 20.19


def haystack(n_sent, needle, depth, seed):
    rnd = random.Random(seed)
    sents = [f"{rnd.choice(PHRASES)} (Ref {rnd.randint(1000, 9999)}.)" for _ in range(n_sent)]
    sents.insert(int(len(sents) * depth), needle)
    return " ".join(sents)


def case(target_tokens, depth, seed, tokens_per_sentence):
    """The prompt and the code it hides, both fixed by the arguments."""
    code = str(random.Random(seed).randint(100000, 999999))
    n = max(10, int(target_tokens / tokens_per_sentence))
    doc = haystack(n, f"IMPORTANT: The secret vault access code is {code}.", depth, seed)
    return doc + QUESTION, code


async def chat(session, args, content, max_tokens, timeout=900):
    import aiohttp

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    body = endpoint.with_thinking(args, body)
    async with session.post(
        args.url.rstrip("/") + "/v1/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as r:
        text = await r.text()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {text[:200]}")
        return json.loads(text)


async def calibrate(session, args):
    """Tokens per filler sentence on this model's tokenizer, from one 1-token request."""
    r = await chat(session, args, haystack(400, "x", 0.5, 0), max_tokens=1)
    return (r["usage"]["prompt_tokens"] - 10) / 401


async def run_case(session, args, target, depth, seed, tps):
    prompt, code = case(target, depth, seed, tps)
    t0 = time.time()
    row = {"target_tokens": target, "depth": depth, "seed": seed}
    try:
        r = await chat(session, args, prompt, max_tokens=64)
        answer = r["choices"][0]["message"].get("content") or ""
        row.update(ok=code in answer, prompt_tokens=r["usage"]["prompt_tokens"], answer=answer[:40])
    except Exception as e:  # noqa: BLE001  (a failed request is a failed case, reported as such)
        row.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    row["latency_s"] = round(time.time() - t0, 2)
    return row


async def scan(args):
    cases = [
        (t, d, s + t + int(d * 10))
        for s in args.seeds
        for t in args.sizes
        for d in args.depths
    ]
    async with endpoint.client_session(args) as session:
        tps = args.tokens_per_sentence or await calibrate(session, args)
        sem = asyncio.Semaphore(args.concurrency)

        async def bounded(c):
            async with sem:
                return await run_case(session, args, *c, tps)

        rows = await asyncio.gather(*(bounded(c) for c in cases))
    return tps, rows


def summarize(rows):
    misses = [r for r in rows if r["ok"] is False and "error" not in r]
    errors = [r for r in rows if "error" in r]
    return {
        "cases": len(rows),
        "misses": len(misses),
        "errors": len(errors),
        "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in rows), default=0),
        "failed_cases": [
            {k: r.get(k) for k in ("target_tokens", "depth", "seed", "prompt_tokens", "answer", "error")}
            for r in misses + errors
        ][:20],
        "passed": not misses and not errors,
    }


def ints(text):
    return [int(x) for x in text.split(",") if x]


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    endpoint.add_endpoint_args(p)
    p.add_argument("--sizes", type=ints, default=ints(DEFAULT_SIZES), help="target prompt tokens")
    p.add_argument("--depths", type=lambda t: [float(x) for x in t.split(",")], default=[0.1, 0.5])
    p.add_argument("--seeds", type=ints, default=ints("1000,4000"))
    p.add_argument(
        "--tokens-per-sentence",
        type=float,
        default=DSV41_TOKENS_PER_SENTENCE,
        help="filler sentence length in tokens; 0 = measure it on the server",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="needles.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    endpoint.resolve_model(args)
    tps, rows = asyncio.run(scan(args))
    summary = summarize(rows)
    with open(args.out, "w") as f:
        json.dump({"endpoint": endpoint.describe(args), "tokens_per_sentence": tps, **summary, "rows": rows}, f, indent=1)
    print(
        f"{'PASS' if summary['passed'] else 'FAIL'}  needles: {summary['misses']} missed, {summary['errors']} errors "
        f"of {summary['cases']} (up to {summary['max_prompt_tokens']} prompt tokens) -> {args.out}"
    )
    for r in summary["failed_cases"]:
        print("  " + json.dumps(r))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
