# Production readiness: what every topology change reports

Status: draft (2026-11-09). A topology change (GPU count, PD split, TP/EP,
context length, pool sizes, HiCache, speculative settings) is not ready until
it has results from both tests below. Served tokens per minute alone is not
enough: it hides cache misses, preemption, queueing and tail latency.

## Test 1: live metrics (any traffic)

`live_metrics.py` reads the workers' and router's Prometheus endpoints and
reports rates per interval plus a whole-run summary. It only reads metrics, so
it runs beside a load test, a soak, or production. The workers need
`--enable-metrics`, which the launcher now passes by default
(`--no-worker-metrics` turns it off).

```sh
python3 live_metrics.py --gpus 8 --interval 10 --out live.jsonl      # PD defaults: 8041/8042, router 29001
# 2P2D: the launcher puts prefill i on PORT+1+i, decode j on PORT+1+num_prefill+j
python3 live_metrics.py --prefill http://127.0.0.1:8041 --prefill http://127.0.0.1:8042 \
    --decode http://127.0.0.1:8043 --decode http://127.0.0.1:8044 --router http://127.0.0.1:29001 --gpus 8
python3 live_metrics.py --router-only --router http://127.0.0.1:29001   # workers without metrics
```

For Prometheus, `--serve PORT` publishes the latest window as gauges at
`/metrics`, so dashboards and alerts get the marquee numbers without PromQL over
the raw counters (the raw counters stay scrapeable from the workers and the
router themselves):

```sh
python3 live_metrics.py --gpus 4 --serve 29100 --serve-host 0.0.0.0 \
    --label topology=4xB200-1P1D --label model=dsv41
# inference_traffic_output_tok_s, inference_traffic_served_tpm_per_gpu,
# inference_cache_prefix_hit, inference_cache_prefill_cache_fill, inference_cache_decode_kv_use,
# inference_preemption_retracted_req_s, inference_queues_router_inflight,
# inference_latency_seconds{stage="ttft|itl|prefill_queue|router_request",quantile="0.5|0.9|0.99"},
# inference_health_router_non2xx_s, inference_up (0 = last scrape failed)
```

Each gauge is a rate or quantile over the last `--interval`, so set the
Prometheus scrape interval to match it.

In a deployment it runs as the `metrics` service in `container/compose.yaml`
(entrypoint mode `metrics`): it reads the pair's `PORT`, `GPUS` and
`DSV41_LAUNCH_ARGS` to find the workers, waits for them, and serves on
`METRICS_PORT` (29100) at `METRICS_HOST` (127.0.0.1; 0.0.0.0 for a remote
Prometheus), with `METRICS_LABELS`. With `--serve`, the tool waits for the
endpoints at start instead of exiting, and reports `inference_up 0` meanwhile.

If a worker's `/metrics` is missing (a stack started without metrics), the
tool exits 2 with that hint. `--router-only` still gives request rate, non-2xx
rate, admission in flight and rejections, and request duration, but no token,
cache, KV or TTFT/ITL numbers. Router admission in flight returns to 0 when
idle. Before the router fix in this PR, `smg_admission_inflight` was set only
when a request was admitted, so after load it kept the last admission's count
(32 after a 32-session replay) while `GET /admission` reported 0. If it stays
nonzero while idle, compare it with `GET /admission` before suspecting a leak.

## Test 2: Oracle scorecard (fixed workload, pass/fail)

`oracle_scorecard.py` replays an Oracle dataset with Oracle decoding, reads the
server counters at the start and end of a fixed window after a warmup, and
gates the result (below). The model is the M3 scorecard in
`sglang-private/benchmark/oracle_replay`.

Oracle decoding (`forced_text`) makes each completion the trace's own text.
The server tokenizes it with the served model's tokenizer, so no dataset is
converted ahead of time. Every turn's history is then exactly what the server
generated, the prefix-cache hit and output share are properties of the
dataset, and runs are comparable. Without it, the model's own text changes the
next turn's prompt.

The text comes from the dataset: realmix steps take the next step's opening
assistant turn; the remaining steps (each conversation's last, and all Tracelab
steps) decode the dataset's `forced_token_ids` with the tokenizer that built it
(`--source-tokenizer`, MiniMax-M3's `tokenizer.json`).

Run on a deployment launched with `--oracle-decode` (benchmark only: requests
can then dictate their output) and, for DSpark, `--oracle-commit-len` set to the
production accept length:

```sh
python oracle_scorecard.py DATASET.json --source-tokenizer TOKENIZERS/minimax-m3 \
    --concurrency 32 --gpus 8 --gpu-model B200 --topology "1 node; 2P TP2 + 2D TP2" \
    --image ghcr.io/...@sha256:... --warmup-s 120 --measure-s 600 \
    --cache-hit-band 0.93,0.96 --output-share-band 0.030,0.035 --out scorecard.json
```

How Oracle decoding works here: the tokenizer manager encodes `forced_text`
and sets `max_new_tokens` to its length; token ids cannot be sent directly. The
server keeps each forced request's tokens
in a device pool indexed by request slot. Prefill and plain decode replace the
sampled token. DSpark decode verifies the forced tokens in place of the drafts,
so the target runs on the tokens the request emits, then commits
`--oracle-commit-len` of them per step (or the target's own acceptance when 0),
with the bonus token forced as well. The router forwards
`forced_text` through a vendored `openai-protocol` (`sgl-model-gateway/vendor/`).

## Marquee metrics

| Group | Metric | Source |
|---|---|---|
| Traffic | requests/s; input tok/s (total, cached, uncached); output tok/s; served TPM total and per physical GPU; output share | `sglang:num_requests_total`, `prompt_tokens_total` and `cached_tokens_total` (prefill), `generation_tokens_total` (decode) |
| Cache | prefix-cache hit; KV pool use per role (cache pressure); evicted tok/s; HiCache host use; write-back and load-back tok/s | `sglang:token_usage`, `kv_used_tokens`, `evicted_tokens_total`, `hicache_host_used_tokens`, `hicache_backup_tokens_total`, `load_back_tokens_total` |
| Preemption | retracted requests/s and tokens/s ("swaps" back to the queue); paused and aborted requests | `sglang:num_retracted_requests_total`, `num_retracted_{input,output}_tokens_total`, `num_paused_reqs`, `num_aborted_requests_total` |
| Queues | running and waiting per role; PD bootstrap, prealloc and transfer queues; router admission in flight; 429s | `sglang:num_running_reqs`, `num_queue_reqs`, `num_prefill_bootstrap_queue_reqs`, `num_decode_{prealloc,transfer}_queue_reqs`, `smg_admission_*` |
| Latency | TTFT, ITL, prefill queue time and router request duration at p50/p90/p99 | `sglang:time_to_first_token_seconds`, `inter_token_latency_seconds`, `queue_time_seconds`, `smg_http_request_duration_seconds` |
| Health | non-2xx/s; KV-transfer and bootstrap failures; prefill retries; speculative accept length | `smg_http_responses_total`, `sglang:num_transfer_failed_reqs_total`, `num_bootstrap_failed_reqs_total`, `num_prefill_retries_total`, `spec_accept_length` |

In PD, input counters come from prefill and output counters from decode. Both
roles count every prompt, so adding them together would double the input rate.

## Scorecard gates

- Zero failed requests, KV-transfer failures and 5xx in the window. 429s are
  reported, not failed.
- Prefix-cache hit and output share within each dataset's band. With Oracle
  decoding these are fixed by the dataset, so a miss means the cache is too
  small, not that the traffic changed. Realmix (`oracle-realmix-256-r36-m3-
  modeled-long25-38k-v3`, sha256 `5b615893…`): hit 0.93-0.96, output share
  0.030-0.035, from the runs below.
- TTFT p50 at or below 3 s (the M3 interactive target). TTFT p99 and ITL
  p50/p99 are recorded; set their limits per topology from its baseline.
- Retractions are zero in steady state: a retraction re-runs a prefill.
- The report includes per-physical-GPU served TPM, the image digest, the launch
  plan, and the dataset's sha256.
- Run with `--oracle-commit-len` at the topology's production accept length
  (5 for DSpark block 5), so decode steps are a property of the dataset too;
  0 lets the model's own acceptance vary them.

## First results (2026-11-15, 4x B200, one 1P1D pair)

Prefill TP2 + decode TP2 (DSpark), HiCache off, 131K context, prefill pool
1.5M tokens; dsv41-pd 1805cee0ec plus the Oracle slot fix (a98446b). Realmix,
`--concurrency C`, warmup 120 s, window 600 s. Every run passed, with 0 failed
requests, 429s and retractions.

| Run | req/s | served TPM/GPU | hit | output share | TTFT p50 / p99 (s) | ITL p50 / p99 (ms) | accept |
|---|---|---|---|---|---|---|---|
| C=16, Oracle | 2.97 | 1.70M | 0.946 | 0.032 | 0.33 / 3.02 | 3.1 / 21 | 5.44 |
| C=32, Oracle | 4.78 | 2.77M | 0.946 | 0.033 | 0.37 / 0.80 | 5.0 / 26 | 5.03 |
| C=48, Oracle | 5.47 | 3.20M | 0.904 | 0.033 | 0.53 / 11.1 | 5.2 / 31 | 5.40 |
| C=64, Oracle | 5.82 | 3.44M | 0.875 | 0.033 | 1.49 / 36.4 | 5.3 / 31 | 5.65 |
| C=32, Oracle, commit length 5 | 4.55 | 2.66M | 0.946 | 0.033 | 0.37 / 0.79 | 5.1 / 8.4 | 4.99 |
| C=32, not forced, server `--oracle-decode` | 4.91 | 2.83M | 0.946 | 0.033 | 0.36 / 0.79 | 5.0 / 14 | 5.47 |
| C=32, not forced, server without it | 4.91 | 2.83M | 0.946 | 0.033 | 0.36 / 0.78 | 5.0 / 14 | 5.23 |

- The hit and output share stay at the dataset's values (0.946, 0.033) up to
  C=32. At 48 the sessions' working set outgrows the 1.5M-token prefix cache:
  the hit falls and TTFT p99 rises, the cache-hit gate's job. The operating
  point for this pair is C=32.
- Oracle off costs nothing: the server with and without `--oracle-decode`
  served the same traffic at the same rates and latencies.
- Forcing tokens costs about 2% of throughput at C=32 (2.77M vs 2.83M), and
  its ITL p99 is higher with commit length 0: rows with forced tokens take
  the eager DSpark accept, not the folded in-graph one.

## Open work

1. Done on a 4x B200 (2026-11-15): forced completions come back exactly
   (Qwen3.5-0.8B plain server 8/8; DSV4.1 PD + DSpark with decode CUDA graphs
   64/64), and Oracle off has no overhead. Not run: prefill without CUDA graphs.
2. Baseline the 8x B200 (2P2D): TTFT p99 and ITL limits for that topology, and
   bands for the Tracelab coding dataset.
3. Pin the router's dependencies: `sgl-model-gateway` has no `Cargo.lock`, and
   a fresh resolve now picks `notify-types` 2.3.0, which needs Rust 1.91; the
   image builds with 1.90.
