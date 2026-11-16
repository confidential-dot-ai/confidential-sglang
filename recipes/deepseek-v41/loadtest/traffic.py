"""Mixed chat traffic and per-request outcome classes, shared by soak, overload and fault.

The mix: short chat, long prompts (6K-80K tokens), tool calls, thinking,
streams read to the end, and streams the client drops early, with varied
sampling. Every request ends in exactly one class:

  ok        200 and a usable answer
  bad       200 but the answer is wrong (a tool call request without a tool call)
  429       refused for load or a key limit (expected under overload)
  4xx       any other client error
  5xx       server error
  conn      connection refused, reset or closed early
  timeout   no complete answer within --request-timeout
"""

from __future__ import annotations

import asyncio
import json
import random
import time

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
WORDS = (
    "alpha river stone market signal cloud engine harbor ledger window orbit canvas "
    "thread copper garden vector quiet lantern summit meadow"
).split()
SAMPLERS = [
    {"temperature": 1.0, "top_p": 0.95},
    {"temperature": 0.0},
    {"temperature": 0.7, "top_k": 50},
    {"temperature": 0.8, "min_p": 0.05},
    {"temperature": 1.0, "top_p": 1.0},
    {"temperature": 0.6, "top_p": 0.9, "frequency_penalty": 0.3},
]
CITIES = ["Paris", "Lagos", "Lima", "Seoul", "Denver", "Perth", "Oslo", "Delhi"]
CLASSES = ("ok", "bad", "429", "4xx", "5xx", "conn", "timeout")
FAILURES = ("bad", "4xx", "5xx", "conn", "timeout")


def _text(n_words, rnd):
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def samplers_without(params):
    """SAMPLERS minus those using a parameter the deployment rejects (min_p under DSpark)."""
    return [sp for sp in SAMPLERS if not set(sp) & set(params)] or [{"temperature": 0.0}]


def mixed(rnd, max_prompt_tokens=80000, samplers=SAMPLERS):
    """(kind, stream, body) of one request from the soak mix."""
    k = rnd.random()
    sp = rnd.choice(samplers)
    if k < 0.40:
        return "short_chat", False, {
            "messages": [{"role": "user", "content": "Summarize: " + _text(rnd.randint(50, 1500), rnd)}],
            "max_tokens": rnd.randint(32, 768), **sp}
    if k < 0.55:
        sizes = [n for n in (6000, 12000, 24000, 48000, 80000) if n <= max_prompt_tokens] or [6000]
        n = rnd.choice(sizes)  # words, about one token each
        return f"long_{n // 1000}k", False, {
            "messages": [{"role": "user", "content": _text(n, rnd) + "\nCount how often 'harbor' appears, roughly."}],
            "max_tokens": rnd.randint(64, 512), **sp}
    if k < 0.70:
        return "tool_call", False, {
            "messages": [{"role": "user", "content": f"What's the weather in {rnd.choice(CITIES)}?"}],
            "tools": [WEATHER], "max_tokens": 512, **sp}
    if k < 0.80:
        return "thinking", False, {
            "messages": [{"role": "user", "content":
                          f"What is {rnd.randint(100, 999)} * {rnd.randint(100, 999)}? Think step by step."}],
            "chat_template_kwargs": {"thinking": True}, "max_tokens": 2048, **sp}
    if k < 0.92:
        return "stream_full", True, {
            "messages": [{"role": "user", "content": "Write a story about " + _text(8, rnd)}],
            "max_tokens": rnd.randint(256, 2048), **sp}
    return "stream_cancel", True, {
        "messages": [{"role": "user", "content": "Write an essay about " + _text(8, rnd)}],
        "max_tokens": 4096, "ignore_eos": True, **sp}


def short(rnd):
    """A cheap request: overload tests offered load, not prompt size."""
    return "short_chat", False, {
        "messages": [{"role": "user", "content": "Summarize: " + _text(rnd.randint(50, 400), rnd)}],
        "max_tokens": rnd.randint(16, 128), "temperature": 0.0}


async def _read_stream(r, row, start, cancel_at):
    """Read SSE events into row. A refusal after the 200 header is an error event in
    the stream; a stream that ends without [DONE] (and was not dropped) failed."""
    n, done = 0, False
    async for raw in r.content:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        ev = json.loads(data)
        if ev.get("error"):
            err = ev["error"] if isinstance(ev["error"], dict) else {"message": ev["error"]}
            code = err.get("code")
            row["cls"] = _status_class(code) if isinstance(code, int) else "5xx"
            row["err"] = f"in-stream {code}: {str(err.get('message'))[:180]}"
            break
        n += 1
        if row["ttft"] is None:
            row["ttft"] = round(time.time() - start, 3)
        if cancel_at is not None and n >= cancel_at:
            done = True  # the client drops the stream on purpose
            break
    if row["cls"] == "ok" and not done:
        row["cls"], row["err"] = "conn", "stream closed before [DONE]"
    row["ctoks"] = n


def _status_class(status):
    if status == 429:
        return "429"
    return "4xx" if 400 <= status < 500 else "5xx"


async def send(session, url, model, kind, stream, body, rnd, t0, timeout):
    """Send one request and return its row; never raises."""
    import aiohttp

    body = {"model": model, **body}
    if stream:
        body["stream"] = True
    start = time.time()
    row = {"start": start - t0, "kind": kind, "cls": "ok", "status": None, "ttft": None, "ctoks": None, "err": None}
    try:
        async with session.post(
            url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as r:
            row["status"] = r.status
            if r.status != 200:
                row["cls"], row["err"] = _status_class(r.status), (await r.text())[:200]
            elif stream:
                await _read_stream(r, row, start, cancel_at=rnd.randint(3, 60) if kind == "stream_cancel" else None)
            else:
                j = await r.json()
                row["ctoks"] = j["usage"]["completion_tokens"]
                m = j["choices"][0]["message"]
                if kind == "tool_call":
                    if not m.get("tool_calls"):
                        row["cls"], row["err"] = "bad", "no tool call: " + (m.get("content") or "")[:100]
                    else:
                        json.loads(m["tool_calls"][0]["function"]["arguments"])
    except asyncio.TimeoutError:
        row["cls"], row["err"] = "timeout", f"no answer in {timeout} s"
    except aiohttp.ClientError as e:
        row["cls"], row["err"] = "conn", f"{type(e).__name__}: {e}"[:200]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        row["cls"], row["err"] = "bad", f"malformed answer: {type(e).__name__}: {e}"[:200]
    row["end"] = time.time() - t0
    row["lat"] = round(row["end"] - row["start"], 3)
    return row


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q / 100 * len(xs)))], 3)


def tally(rows):
    """Counts per outcome class, and latency quantiles of the ok rows."""
    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r["cls"]] += 1
    lat = [r["lat"] for r in rows if r["cls"] == "ok"]
    ttft = [r["ttft"] for r in rows if r["cls"] == "ok" and r["ttft"] is not None]
    return {
        "requests": len(rows),
        "classes": counts,
        "failures": sum(counts[c] for c in FAILURES),
        "latency_s": {q: pct(lat, q) for q in (50, 95, 99)},
        "stream_ttft_s": {q: pct(ttft, q) for q in (50, 95, 99)},
        "first_errors": [r["err"] for r in rows if r["cls"] in FAILURES][:5],
    }


def per_second(rows, duration):
    """ok and failed requests per second of end time, as two lists."""
    n = int(duration) + 1
    ok, bad = [0] * n, [0] * n
    for r in rows:
        i = min(n - 1, max(0, int(r["end"])))
        if r["cls"] == "ok":
            ok[i] += 1
        elif r["cls"] in FAILURES:
            bad[i] += 1
    return ok, bad


class Load:
    """A closed loop of `concurrency` clients sending until stop() or `end`."""

    def __init__(self, session, url, model, concurrency, make, seed, timeout, t0, end=float("inf")):
        self.session, self.url, self.model = session, url, model
        self.concurrency, self.make, self.timeout = concurrency, make, timeout
        self.rnd, self.t0, self.end = random.Random(seed), t0, end
        self.rows, self.stopped = [], False

    def stop(self):
        self.stopped = True

    async def _client(self):
        while not self.stopped and time.time() < self.end:
            kind, stream, body = self.make(self.rnd)
            self.rows.append(
                await send(self.session, self.url, self.model, kind, stream, body, self.rnd, self.t0, self.timeout)
            )

    async def run(self):
        await asyncio.gather(*(self._client() for _ in range(self.concurrency)))
        return self.rows
