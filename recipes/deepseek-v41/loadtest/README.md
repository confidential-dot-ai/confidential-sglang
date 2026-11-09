# DeepSeek-V4.1 load test: Oracle replay

The standard load test for a PD pair (2026-11-09). It replays multi-turn agent
conversations concurrently, one session per conversation, with no think time.

## Datasets

| Dataset | Content | Prompts | Output |
|---|---|---|---|
| [oracle-coding-together-256-saturated-v1](https://f004.backblazeb2.com/file/fileblob/oracle-datasets/oracle-coding-together-256-saturated-v1.json) (637 MB) | Tracelab agentic coding, 256 sessions | 45K–203K tokens; needs `--context-length` ≥ 210K (`dsv41-262k.env`) | per step target |
| [oracle-realmix-256-r36-m3-modeled-long25-38k-v3](https://f004.backblazeb2.com/file/fileblob/oracle-datasets/oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json) (119 MB) | Real agent traces, 256 sessions | ~36K mean, ~95K max | ~1.2K mean |

`oracle_replay.py` reads both formats:

- **Tracelab** steps carry `workload` and `reset_history`. Each step's messages
  are appended to the history, and the model's reply becomes the next assistant
  turn. The router rejects tool messages without a matching assistant tool call,
  so the driver adds synthetic `tool_calls`.
- **Realmix** steps already include the trace's previous assistant turn, so the
  history is replayed verbatim and the model's reply is discarded.

Output lengths follow the dataset (`max_tokens` with `ignore_eos`). The
`forced_token_ids` are another tokenizer's ids and are not replayed. A 429 from
the router's admission queue is retried with backoff and counted.

## Running

```sh
# One level, inside the pd container (the image's python has aiohttp):
python oracle_replay.py DATASET.json --sessions 32 --time-limit 720 --output s32.jsonl

# Sweep 16..256 sessions, 12 min each; stops at the first traceback/OOM or failed chat check:
DATASET=DATASET.json LOGS=/data/dsv41-bringup/recipes/container-pd-<ts> ./oracle_sweep.sh
```

Each run prints progress every `--report-every` seconds and ends with a line
`SUMMARY {...}`. The summary has:

- steps/s;
- prompt and output tok/s;
- TTFT, step latency and TPOT at p50/p90/p99;
- 429 retries, context skips and errors.

With `LOGS` set, the sweep also prints the prefill prefix-cache hit for each level.

## Results (2026-11-09)

Setup: one pair on 4x B200 (non-CED), `dsv41-262k.env` settings without HiCache (no host RAM for it).

| Realmix sessions | Turns/s | Output tok/s | Prefill cache hit | TTFT p50 / p99 |
|---|---|---|---|---|
| 16 | 2.50 | 3085 | 94.4% | 0.34 s / 2.1 s |
| 32 | 3.83 | 4751 | 91.1% | 0.44 s / 9.4 s |
| 64 | 2.40 | 3050 | 72.5% | 1.7 s / (p90 67 s) |
| 128 | 2.43 | | 61.5% | 49 s |
| 256 | 2.99 | | 65.1% | 64 s / 130 s |

| Coding sessions | Steps/s | Prompt tok/s | Prefill cache hit | TTFT p50 / p99 |
|---|---|---|---|---|
| 16 | 1.49 | 155K | 84.6% | 3.0 s / 45 s |
| 32 | 1.23 | 130K | 80% | 3.7 s / 125 s |
| 64 | 0.37 | 29K | 0.9% | 147 s / 281 s |

No worker errors occurred at any level. The 256-session realmix level had 893
429-retries, and every request was eventually served.

The knee is about 32 realmix conversations, or 16–32 coding sessions, per pair.
Past it, the sessions' working set no longer fits in prefill's prefix cache
(1M tokens), so turns re-prefill their whole history. HiCache (a host-memory
tier) is the lever for that. `dsv41-262k.env` enables it, but these results were
measured without it: the test VM's host RAM was held by the Engram cache.
