#!/usr/bin/env python3
"""Functional battery: tool calling, API features, long context and limits.

Groups (--only, comma-separated): tools (11 cases x --trials), api (7 cases x 4,
plus a long output and n=2), needle (long-context recall up to the context
length), limits (a prompt over the context length must get a 4xx; a storm of
client disconnects must leave the server healthy).

Writes one JSON line per case to --out-jsonl and a summary to --out. Exits 1 if
any case failed that is not listed in --known-failures (for example t_n2 on a
PD deployment, where n>1 is not supported).

  python func_tests.py --url https://GATEWAY --context-length 131072
"""
import argparse, asyncio, json, random, re, sys, time
import aiohttp

import endpoint
from needles import PHRASES, haystack  # noqa: F401  (PHRASES: shared filler)

args = None  # set by main()
URL = None
SAMPLING = [  # recommended (temp 1.0 / top_p 0.95) and greedy, alternating
    {"temperature": 1.0, "top_p": 0.95},
    {"temperature": 0.0},
]

# ---------------------------------------------------------------- tools
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
STOCK = {"type": "function", "function": {
    "name": "get_stock_price", "description": "Get the latest stock price for a ticker symbol.",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}},
                   "required": ["ticker"]}}}
ORDER = {"type": "function", "function": {
    "name": "create_order", "description": "Create a purchase order.",
    "parameters": {"type": "object", "properties": {
        "customer": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name", "email"]},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
            "required": ["sku", "qty"]}},
        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        "gift": {"type": "boolean"}},
        "required": ["customer", "items", "priority"]}}}


def filler_tools(n):
    verbs = ["list", "delete", "update", "archive", "export", "sync", "audit", "tag"]
    nouns = ["invoices", "users", "tickets", "repos", "buckets", "alerts", "contracts",
             "shipments", "leads", "campaigns"]
    out = []
    for i in range(n):
        v, o = verbs[i % len(verbs)], nouns[(i // len(verbs)) % len(nouns)]
        out.append({"type": "function", "function": {
            "name": f"{v}_{o}_{i}", "description": f"{v.capitalize()} {o} in system #{i}.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "limit": {"type": "integer"}}}}})
    return out


def check_args(tc, required, types=None):
    a = json.loads(tc["function"]["arguments"])
    for k in required:
        assert k in a, f"missing {k} in {a}"
    for k, t in (types or {}).items():
        if k in a:
            assert isinstance(a[k], t), f"{k} has type {type(a[k]).__name__}"
    return a


def tcalls(r):
    return r["choices"][0]["message"].get("tool_calls") or []


async def chat(s, body, timeout=600):
    body = endpoint.with_thinking(args, {"model": args.model, **body})
    t = time.time()
    async with s.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        txt = await r.text()
        dt = time.time() - t
        if r.status != 200:
            raise AssertionError(f"HTTP {r.status}: {txt[:300]}")
        return json.loads(txt), dt


async def chat_stream(s, body, timeout=600):
    body = endpoint.with_thinking(args, {"model": args.model, "stream": True,
                                         "stream_options": {"include_usage": True}, **body})
    t, ttft = time.time(), None
    content, reasoning, tcs, usage, finish = "", "", {}, None, None
    async with s.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        assert r.status == 200, f"HTTP {r.status}: {(await r.text())[:300]}"
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.time() - t
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    e = tcs.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    e["name"] += f.get("name") or ""
                    e["arguments"] += f.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    return {"content": content, "reasoning": reasoning, "tool_calls": tcs,
            "usage": usage, "finish": finish, "ttft": ttft}, time.time() - t


# ---------------------------------------------------------------- test cases
async def t_single_tool(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "What's the weather like in Berlin? Use celsius."}],
                           "tools": [WEATHER], "max_tokens": 1024, **sp})
    tc = tcalls(r)
    assert len(tc) == 1, f"{len(tc)} tool calls; content={r['choices'][0]['message'].get('content')!r:.200}"
    assert tc[0]["function"]["name"] == "get_weather"
    a = check_args(tc[0], ["city"])
    assert "berlin" in a["city"].lower()
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    return dt


async def t_parallel_tools(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        "I need three things at once: weather in Paris, weather in Tokyo, and the stock price of NVDA."}],
        "tools": [WEATHER, STOCK], "max_tokens": 1024, **sp})
    tc = tcalls(r)
    names = sorted(t["function"]["name"] for t in tc)
    assert names == ["get_stock_price", "get_weather", "get_weather"], f"got {names}"
    for t in tc:
        json.loads(t["function"]["arguments"])
    return dt


async def t_tool_choice_none(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "What's the weather in Rome?"}],
                           "tools": [WEATHER], "tool_choice": "none", "max_tokens": 512, **sp})
    assert not tcalls(r), "tool call despite tool_choice=none"
    assert r["choices"][0]["message"].get("content"), "empty content"
    return dt


async def t_tool_choice_required(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Hi there! How are you?"}],
                           "tools": [WEATHER, STOCK], "tool_choice": "required", "max_tokens": 512, **sp})
    tc = tcalls(r)
    assert tc, "no tool call despite tool_choice=required"
    json.loads(tc[0]["function"]["arguments"])
    return dt


async def t_tool_choice_named(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Tell me about Apple."}],
                           "tools": [WEATHER, STOCK],
                           "tool_choice": {"type": "function", "function": {"name": "get_stock_price"}},
                           "max_tokens": 512, **sp})
    tc = tcalls(r)
    assert tc and tc[0]["function"]["name"] == "get_stock_price", f"got {[t['function']['name'] for t in tc]}"
    check_args(tc[0], ["ticker"])
    return dt


async def t_no_tool_needed(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                           "tools": [WEATHER, STOCK], "max_tokens": 256, **sp})
    assert not tcalls(r), "unnecessary tool call"
    assert "paris" in (r["choices"][0]["message"].get("content") or "").lower()
    return dt


async def t_multi_turn_tool_result(s, sp):
    msgs = [{"role": "user", "content": "What's the weather in Oslo?"}]
    r, dt1 = await chat(s, {"messages": msgs, "tools": [WEATHER], "max_tokens": 1024, **sp})
    tc = tcalls(r)
    assert tc, "no tool call in turn 1"
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "",
                 "tool_calls": tc})
    msgs.append({"role": "tool", "tool_call_id": tc[0]["id"],
                 "content": json.dumps({"city": "Oslo", "temp_c": -7, "condition": "heavy snow"})})
    r2, dt2 = await chat(s, {"messages": msgs, "tools": [WEATHER], "max_tokens": 1024, **sp})
    c = (r2["choices"][0]["message"].get("content") or "").lower()
    assert not tcalls(r2), "re-called tool after result"
    assert "-7" in c or "minus 7" in c or "−7" in c, f"answer ignores tool result: {c[:200]!r}"
    assert "snow" in c, f"answer ignores condition: {c[:200]!r}"
    return dt1 + dt2


async def t_stream_tool(s, sp):
    r, dt = await chat_stream(s, {"messages": [{"role": "user", "content":
        "Get the weather for Lisbon and Madrid in fahrenheit."}], "tools": [WEATHER], "max_tokens": 1024, **sp})
    tcs = list(r["tool_calls"].values())
    assert len(tcs) == 2, f"{len(tcs)} streamed tool calls"
    for t in tcs:
        assert t["name"] == "get_weather", t["name"]
        a = json.loads(t["arguments"])
        assert "city" in a
    assert r["finish"] == "tool_calls", r["finish"]
    assert r["usage"] and r["usage"]["completion_tokens"] > 0
    return dt


async def t_many_tools(s, sp):
    tools = filler_tools(60)
    tools.insert(37, STOCK)
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "What's MSFT trading at?"}],
                           "tools": tools, "max_tokens": 1024, **sp})
    tc = tcalls(r)
    assert tc and tc[0]["function"]["name"] == "get_stock_price", f"got {[t['function']['name'] for t in tc]}"
    a = check_args(tc[0], ["ticker"])
    assert "msft" in a["ticker"].lower()
    return dt


async def t_complex_schema(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        "Place a high-priority order for Dana Kim (dana@example.com): 3 of SKU A-100 and 1 of SKU B-7. It's a gift."}],
        "tools": [ORDER], "max_tokens": 1024, **sp})
    tc = tcalls(r)
    assert tc, "no tool call"
    a = check_args(tc[0], ["customer", "items", "priority"], {"items": list, "customer": dict})
    assert a["priority"] == "high", a["priority"]
    skus = {(i["sku"], i["qty"]) for i in a["items"]}
    assert skus == {("A-100", 3), ("B-7", 1)}, skus
    assert isinstance(a["items"][0]["qty"], int)
    assert a.get("gift") is True, a.get("gift")
    return dt


async def t_thinking_tool(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        "Which is warmer right now, Cairo or Helsinki? Check both."}], "tools": [WEATHER],
        "chat_template_kwargs": {"thinking": True}, "max_tokens": 4096, **sp})
    m = r["choices"][0]["message"]
    tc = tcalls(r)
    assert len(tc) >= 2, f"{len(tc)} tool calls"
    assert "<think>" not in (m.get("content") or "") and "</think>" not in (m.get("content") or "")
    return dt


async def t_thinking_plain(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "A train leaves at 14:35 and arrives at 17:10. How many minutes is the trip? Answer with just the number."}],
                           "chat_template_kwargs": {"thinking": True}, "max_tokens": 4096, **sp})
    m = r["choices"][0]["message"]
    assert m.get("reasoning_content"), "no reasoning_content"
    assert "155" in (m.get("content") or ""), f"content={m.get('content')!r:.200}"
    return dt


async def t_json_schema(s, sp):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
              "langs": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "langs"],
              "additionalProperties": False}
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        "Extract: 'Priya is 31 and codes in Rust and Go.' Return JSON."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
        "max_tokens": 256, **sp})
    a = json.loads(r["choices"][0]["message"]["content"])
    assert a["age"] == 31 and set(map(str.lower, a["langs"])) == {"rust", "go"}, a
    return dt


async def t_stop_seq(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Count from 1 to 20 separated by spaces."}],
                           "stop": [" 7"], "max_tokens": 128, **sp})
    c = r["choices"][0]["message"]["content"]
    assert "8" not in c.split()[-1:] and " 7" not in c, f"stop ignored: {c!r}"
    return dt


async def t_max_tokens(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
                           "max_tokens": 50, **sp})
    assert r["usage"]["completion_tokens"] <= 50 and r["choices"][0]["finish_reason"] == "length", r["usage"]
    return dt


async def t_long_output(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Write a detailed 5000-word technical essay on the history of distributed databases."}],
                           "max_tokens": 8192, "ignore_eos": True, **sp})
    assert r["usage"]["completion_tokens"] == 8192, r["usage"]
    return dt


async def t_unicode(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        "Translate to Japanese and add a fitting emoji at the end: 'Good morning, the coffee is ready.'"}],
        "max_tokens": 256, **sp})
    c = r["choices"][0]["message"]["content"]
    assert re.search(r"[぀-ヿ一-鿿]", c), c
    assert "�" not in c, "replacement char in output"
    return dt


async def t_n2(s, sp):
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Name a random fruit."}],
                           "n": 2, "max_tokens": 64, **sp}, timeout=60)
    assert len(r["choices"]) == 2, len(r["choices"])
    return dt


async def t_logprobs_rejected(s, sp):
    """Known limitation: speculative decoding has no logprobs. Pass = clean 4xx, server unaffected."""
    body = {"model": args.model, "messages": [{"role": "user", "content": "Say hello."}],
            "logprobs": True, "top_logprobs": 3, "max_tokens": 16, **sp}
    t = time.time()
    async with s.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=60)) as r:
        assert 400 <= r.status < 500, f"HTTP {r.status}"
    return time.time() - t


async def t_system_and_history(s, sp):
    msgs = [{"role": "system", "content": "You are a pirate. Always say 'Arrr' somewhere."},
            {"role": "user", "content": "My name is Quentin."},
            {"role": "assistant", "content": "Arrr, welcome aboard, Quentin!"},
            {"role": "user", "content": "What is my name?"}]
    r, dt = await chat(s, {"messages": msgs, "max_tokens": 128, **sp})
    c = r["choices"][0]["message"]["content"]
    assert "quentin" in c.lower(), c
    return dt


# --------------------------------------------------- long context / limits
SENT_TOKENS = 22.5  # measured approx tokens per filler sentence; refined via usage


async def t_needle(s, target_tokens, depth, seed):
    code = f"{random.Random(seed).randint(100000, 999999)}"
    needle = f"IMPORTANT: The secret vault access code is {code}."
    n = max(10, int(target_tokens / SENT_TOKENS))
    doc = haystack(n, needle, depth, seed)
    r, dt = await chat(s, {"messages": [{"role": "user", "content":
        doc + "\n\nQuestion: What is the secret vault access code mentioned in the text above? Reply with only the code."}],
        "max_tokens": 64, "temperature": 0.0}, timeout=900)
    c = r["choices"][0]["message"]["content"] or ""
    pt = r["usage"]["prompt_tokens"]
    assert code in c, f"needle miss at {pt} tokens: {c!r:.100}"
    return dt, pt


async def t_over_limit(s):
    doc = haystack(int(args.context_length * 1.1 / SENT_TOKENS), "x", 0.5, 1)
    body = {"model": args.model, "messages": [{"role": "user", "content": doc}], "max_tokens": 16}
    async with s.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        txt = await r.text()
        assert 400 <= r.status < 500, f"expected 4xx for >context prompt, got {r.status}: {txt[:200]}"
        return r.status, txt[:200]


async def t_disconnects(s, n=24):
    """Open n streams, drop each after a few chunks, then confirm the server is healthy."""
    async def one(i):
        body = {"model": args.model, "stream": True, "max_tokens": 2048, "ignore_eos": True,
                "messages": [{"role": "user", "content": f"Write a long essay #{i} about rivers."}]}
        async with s.post(URL, json=body) as r:
            k = 0
            async for _ in r.content:
                k += 1
                if k > 5 + i % 20:
                    break  # leaving the context manager closes the connection
    await asyncio.gather(*(one(i) for i in range(n)))
    await asyncio.sleep(5)
    r, dt = await chat(s, {"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8, "temperature": 0})
    assert r["choices"][0]["message"]["content"], "no content after disconnect storm"
    return dt


TOOL_TESTS = [t_single_tool, t_parallel_tools, t_tool_choice_none, t_tool_choice_required,
              t_tool_choice_named, t_no_tool_needed, t_multi_turn_tool_result, t_stream_tool,
              t_many_tools, t_complex_schema, t_thinking_tool]
API_TESTS = [t_thinking_plain, t_json_schema, t_stop_seq, t_max_tokens, t_unicode,
             t_logprobs_rejected, t_system_and_history]


def needle_targets(context_length):
    """Prompt sizes up to just under the context length (8K left for the answer)."""
    fixed = [t for t in (4000, 16000, 32000, 64000, 100000) if t < context_length - 8192]
    return fixed + [context_length - 8192]


async def run_all():
    out = open(args.out_jsonl, "w")
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    def rec(group, name, ok, dt=None, err=None, **kw):
        row = {"group": group, "test": name, "ok": ok, "latency_s": round(dt, 3) if dt else None,
               "err": err, "ts": time.time(), **kw}
        results.append(row)
        out.write(json.dumps(row) + "\n"); out.flush()

    async with endpoint.client_session(args, limit=256) as s:
        async def run(group, fn, sp, trial):
            async with sem:
                try:
                    dt = await fn(s, sp)
                    rec(group, fn.__name__, True, dt, sampling=sp, trial=trial)
                except Exception as e:
                    rec(group, fn.__name__, False, err=f"{type(e).__name__}: {e}"[:400], sampling=sp, trial=trial)

        if not args.only or "tools" in args.only:
            await asyncio.gather(*(run("tools", fn, SAMPLING[t % 2], t)
                                   for fn in TOOL_TESTS for t in range(args.trials)))
        if not args.only or "api" in args.only:
            await asyncio.gather(*(run("api", fn, SAMPLING[t % 2], t)
                                   for fn in API_TESTS for t in range(4)))
            await run("api", t_long_output, {"temperature": 1.0, "top_p": 0.95}, 0)
            await run("api", t_n2, {"temperature": 1.0, "top_p": 0.95}, 0)
        if not args.only or "needle" in args.only:
            # calibrate tokens per sentence
            global SENT_TOKENS
            r, _ = await chat(s, {"messages": [{"role": "user", "content": haystack(400, "x", 0.5, 0)}], "max_tokens": 1})
            SENT_TOKENS = (r["usage"]["prompt_tokens"] - 10) / 401
            print(f"calibrated {SENT_TOKENS:.2f} tokens/sentence", flush=True)
            for tgt in needle_targets(args.context_length):
                for depth in [0.1, 0.5, 0.9]:
                    seed = tgt + int(depth * 10)
                    try:
                        dt, pt = await t_needle(s, tgt, depth, seed)
                        rec("needle", f"needle_{tgt//1000}k_d{depth}", True, dt, prompt_tokens=pt, depth=depth)
                    except Exception as e:
                        rec("needle", f"needle_{tgt//1000}k_d{depth}", False, err=str(e)[:300], depth=depth)
            try:
                st, txt = await t_over_limit(s)
                rec("limits", "over_context", True, status=st, body=txt)
            except Exception as e:
                rec("limits", "over_context", False, err=str(e)[:300])
        if not args.only or "disconnect" in args.only:
            try:
                dt = await t_disconnects(s)
                rec("limits", "client_disconnect_storm_24", True, dt)
            except Exception as e:
                rec("limits", "client_disconnect_storm_24", False, err=str(e)[:300])

    return results


def summarize(results, known):
    agg = {}
    for r in results:
        a = agg.setdefault((r["group"], r["test"]), [0, 0, [], []])
        a[0] += r["ok"]; a[1] += 1
        if r["latency_s"]: a[2].append(r["latency_s"])
        if not r["ok"]: a[3].append(r["err"])
    tests = []
    for (g, t), (ok, n, lat, errs) in agg.items():
        lat.sort()
        tests.append({"group": g, "test": t, "passed": ok, "runs": n,
                      "p50_s": round(lat[len(lat) // 2], 2) if lat else None,
                      "known_failure": t in known, "first_error": errs[0] if errs else None})
    unexpected = [x for x in tests if x["passed"] < x["runs"] and not x["known_failure"]]
    return {
        "cases": sum(x["runs"] for x in tests),
        "passed_cases": sum(x["passed"] for x in tests),
        "known_failures": [x["test"] for x in tests if x["passed"] < x["runs"] and x["known_failure"]],
        "unexpected_failures": [x["test"] for x in unexpected],
        "tests": tests,
        "passed": not unexpected,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    endpoint.add_endpoint_args(p)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--context-length", type=int, default=131072, help="the deployment's context length")
    p.add_argument("--only", default="", help="tools,api,needle,disconnect (default: all)")
    p.add_argument("--known-failures", default="", help="comma-separated test names that may fail")
    p.add_argument("--out", default="func.json")
    p.add_argument("--out-jsonl", default="func.jsonl")
    return p


def main(argv=None):
    global args, URL
    args = parser().parse_args(argv)
    endpoint.resolve_model(args)
    URL = args.url.rstrip("/") + "/v1/chat/completions"
    known = {x for x in args.known_failures.split(",") if x}
    summary = summarize(asyncio.run(run_all()), known)
    with open(args.out, "w") as f:
        json.dump({"endpoint": endpoint.describe(args), "context_length": args.context_length, **summary}, f, indent=1)
    print(f"\n{'group':8} {'test':32} {'pass':>7} {'p50 s':>7}  first failure")
    for x in summary["tests"]:
        mark = " (known)" if x["known_failure"] and x["passed"] < x["runs"] else ""
        print(f"{x['group']:8} {x['test']:32} {x['passed']:>3}/{x['runs']:<3} {x['p50_s'] or '-':>7}  "
              f"{(x['first_error'] or '')[:100]}{mark}")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}  functional: {summary['passed_cases']}/{summary['cases']} "
          f"cases, unexpected failures: {summary['unexpected_failures'] or 'none'} -> {args.out}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
