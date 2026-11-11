"""Concurrent replay of an Oracle dataset against an OpenAI-compatible chat endpoint.

Two formats: Tracelab (oracle-coding-together-*: steps carry `workload` and
`reset_history`; described below) and realmix (oracle-realmix-*: each step's
messages already include the trace's previous assistant turn, so history is
replayed verbatim and the model's reply is discarded; output length =
len(forced_token_ids)).

Each conversation is one session; its steps run in order, each session runs
concurrently with the others ("saturated": no think time between steps).
History: reset_history clears it, the step's messages are appended, the request
is sent with the whole history, and the model's reply is appended as the
assistant turn. Output length follows the dataset (max_tokens = target output
tokens, ignore_eos); the dataset's forced_token_ids are another tokenizer's
ids and are not replayed. Tool messages need a matching assistant tool call,
so each assistant turn followed by tool results carries synthetic tool_calls
(reset steps that open with a tool result get a short synthetic assistant
turn, ~20 tokens).

429s (router admission queue timeout) are retried with backoff, as an agent
client would; each retry is counted.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def tool_call(cid):
    return {"id": cid, "type": "function", "function": {"name": "exec", "arguments": "{}"}}


class Stats:
    def __init__(self):
        self.rows, self.retries_429, self.errors = [], 0, {}
        self.skipped_context = 0
        self.t0 = time.time()


async def run_step(session, args, sid, idx, history, max_tokens, stats, out):
    body = {
        "model": args.model,
        "messages": history,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload = json.dumps(body)
    retries = 0
    while True:
        t0 = time.time()
        ttft, text, usage = None, [], None
        try:
            async with session.post(
                args.url + "/v1/chat/completions",
                data=payload,
                headers={"content-type": "application/json"},
            ) as resp:
                if resp.status == 429:
                    retries += 1
                    stats.retries_429 += 1
                    await resp.read()
                    await asyncio.sleep(min(10.0, 0.5 * 2 ** min(retries, 5)) * random.uniform(0.5, 1.5))
                    continue
                if resp.status != 200:
                    msg = (await resp.text())[:200]
                    key = f"{resp.status}: {msg}"
                    stats.errors[key] = stats.errors.get(key, 0) + 1
                    return None
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices", []):
                        delta = ch.get("delta") or {}
                        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                        if piece or delta.get("tool_calls"):
                            if ttft is None:
                                ttft = time.time() - t0
                            text.append(piece)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            key = f"conn: {type(e).__name__}"
            stats.errors[key] = stats.errors.get(key, 0) + 1
            return None
        end = time.time()
        usage = usage or {}
        row = {
            "session": sid,
            "step": idx,
            "start": t0 - stats.t0,
            "end": end - stats.t0,
            "ttft": ttft,
            "latency": end - t0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "target_output": max_tokens,
            "retries_429": retries,
        }
        stats.rows.append(row)
        out.write(json.dumps(row) + "\n")
        return "".join(text)


async def run_session(session, args, sid, steps, stats, out, deadline, est_ratio):
    history, calls = [], 0
    for idx, step in enumerate(steps[: args.max_steps or None]):
        if time.time() > deadline:
            return
        if "workload" not in step:
            # Realmix format: each step carries the trace's previous assistant turn,
            # so history is replayed verbatim and the model's reply is discarded.
            if idx == 0:
                history = []
            history = history + [{"role": m["role"], "content": m["content"]} for m in step["messages"]]
            max_tokens = len(step["forced_token_ids"]) or step["max_new_tokens"]
            if args.context_limit:
                est = sum(len(m["content"] or "") for m in history) / 3.5 + 12 * len(history)
                if est + max_tokens > args.context_limit:
                    stats.skipped_context += 1
                    return
            if await run_step(session, args, sid, idx, history, max_tokens, stats, out) is None:
                return  # this format cannot resume past a failed turn coherently
            continue
        if step["reset_history"] or idx == 0:
            history = []
        msgs = step["messages"]
        new = []
        for m in msgs:
            if m["role"] == "tool":
                prev = (new or history)[-1] if (new or history) else None
                if not prev or prev["role"] != "assistant":
                    prev = {"role": "assistant", "content": "", "tool_calls": []}
                    new.append(prev)
                prev.setdefault("tool_calls", [])
                cid = f"call_{sid}_{calls}"
                calls += 1
                prev["tool_calls"].append(tool_call(cid))
                new.append({"role": "tool", "tool_call_id": cid, "content": m["content"]})
            else:
                new.append({"role": m["role"], "content": m["content"]})
        history = history + new
        max_tokens = step["workload"].get("target_output_tokens") or len(step["forced_token_ids"])
        if args.context_limit:
            est = sum(len(m.get("content") or "") for m in history) / est_ratio + 12 * len(history)
            if est + max_tokens > args.context_limit:
                stats.skipped_context += 1
                history = []  # the session cannot continue past an oversized prompt
                continue
        text = await run_step(session, args, sid, idx, history, max_tokens, stats, out)
        if text is None:
            history = []  # failed step: next step resumes from a fresh history
            continue
        history = history + [{"role": "assistant", "content": text}]


async def reporter(stats, stop, interval):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except asyncio.TimeoutError:
            pass
        el = time.time() - stats.t0
        done = len(stats.rows)
        inp = sum(r["prompt_tokens"] or 0 for r in stats.rows)
        print(
            f"[{el:7.0f}s] steps {done}  {done / el:.2f}/s  prompt tok/s {inp / el:,.0f}  "
            f"429 retries {stats.retries_429}  errors {sum(stats.errors.values())}",
            flush=True,
        )


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--url", default="http://127.0.0.1:8040")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--sessions", type=int, default=0, help="first N sessions (0 = all)")
    p.add_argument("--max-steps", type=int, default=0, help="steps per session (0 = all)")
    p.add_argument("--time-limit", type=float, default=0, help="stop starting steps after S seconds")
    p.add_argument("--context-limit", type=int, default=0, help="skip steps whose est. prompt+output exceeds this")
    p.add_argument("--output", default="oracle.jsonl")
    p.add_argument("--report-every", type=float, default=30)
    p.add_argument("--request-timeout", type=float, default=3600)
    args = p.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)
    convs = data["conversations"]
    if args.sessions:
        convs = convs[: args.sessions]
    stats = Stats()
    deadline = stats.t0 + (args.time_limit or 1e12)
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    stop = asyncio.Event()
    print(
        f"Oracle replay: {len(convs)} sessions, "
        f"{sum(len(c[: args.max_steps or None]) for c in convs)} steps, url {args.url}",
        flush=True,
    )
    with open(args.output, "w") as out:
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            rep = asyncio.create_task(reporter(stats, stop, args.report_every))
            await asyncio.gather(
                *(run_session(session, args, i, c, stats, out, deadline, 4.997) for i, c in enumerate(convs))
            )
            stop.set()
            await rep

    rows = stats.rows
    el = max((r["end"] for r in rows), default=time.time() - stats.t0)
    ttft = [r["ttft"] for r in rows if r["ttft"] is not None]
    lat = [r["latency"] for r in rows]
    inp = sum(r["prompt_tokens"] or 0 for r in rows)
    outp = sum(r["completion_tokens"] or 0 for r in rows)
    # Decode time per output token, excluding the first.
    tpot = [
        (r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1)
        for r in rows
        if r["ttft"] is not None and (r["completion_tokens"] or 0) > 1
    ]
    summary = {
        "sessions": len(convs),
        "steps_ok": len(rows),
        "duration_s": round(el, 1),
        "steps_per_s": round(len(rows) / el, 3) if el else None,
        "prompt_tokens_per_s": round(inp / el) if el else None,
        "output_tokens_per_s": round(outp / el) if el else None,
        "mean_prompt_tokens": round(inp / len(rows)) if rows else None,
        "ttft_s": {q: round(pct(ttft, q), 2) for q in (50, 90, 99)},
        "step_latency_s": {q: round(pct(lat, q), 2) for q in (50, 90, 99)},
        "tpot_ms": {q: round(1000 * pct(tpot, q), 1) for q in (50, 90, 99)},
        "retries_429": stats.retries_429,
        "skipped_context": stats.skipped_context,
        "errors": stats.errors,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
