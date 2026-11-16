# Stress-testing a candidate cluster

`cluster_stress.py` runs every check from outside the cluster, through the
gateway, the way a customer reaches it, and writes one report. Exec into pods
is off on a C8s cluster, so nothing here needs a shell on a node: requests go
through the gateway with an API key, server counters come from Prometheus,
and faults go through the control plane.

## What you need

- The gateway URL (or the local TEErminator URL that forwards to it), and
  `--ca-cert` if its certificate chains to a private CA.
- A test API key with raised limits, in an environment variable (default
  `STRESS_API_KEY`). The gateway's per-key rate limit and token budget would
  otherwise turn the load phases into 429s. The key is never put on a command
  line or in a result file.
- Read access to the Prometheus the cluster remote-writes to, and label
  matchers that select the workers (and the router) of this deployment.
- For the fault phase, a command that injects the fault through the control
  plane (a kubeconfig minted with the operator key).
- Python 3.10+ with `aiohttp`; the load phase also needs `tokenizers` with
  `--oracle`. Any machine with network access to the gateway works.

## Phases

| Phase | Tool | Passes when |
|---|---|---|
| func | `func_tests.py` | Tool calls (11 cases x 10), API features (JSON schema, stop, max tokens, thinking, unicode, history), needles up to the context length, over-context prompt gets a 4xx, a storm of client disconnects leaves the server healthy. Only `--known-failures` may fail. |
| needles | `needles.py` | Every one of the fixed long-prompt recall cases (default: the 64 cases that caught the CED tail-graph bug, 14K-60K tokens) returns its code. |
| load | `oracle_scorecard.py` | Realmix agent traces at `--sessions` in flight; in the measured window: 0 failed requests and KV-transfer failures, prefix hit and output share inside the dataset's bands, TTFT p50 at most 3 s, 0 retractions. Reports served TPM per GPU. |
| overload | `overload.py` | 64 to 512 clients of short requests, then a burst: only 429s refuse the excess (no 5xx, connection errors or wrong answers), and a request succeeds within 2 minutes afterwards. The report says whether any step was refused at all (`shed`); if none was, the levels never reached capacity and the refusal path is untested (`--require-shed` makes that a failure). |
| soak | `soak.py` | Mixed traffic (short chat, 6K-80K prompts, tool calls, thinking, full and dropped streams) at `--sessions` for `--soak-minutes`: 0 failed requests and healthy at the end. Per-minute latency and 429s show drift. |
| fault | `fault.py` | Load at 16 clients, then `--fault-cmd` (for example deleting a worker pod): the endpoint recovers (60 s with no failures at 90% of baseline throughput) within an hour, and `--check-cmd` exits 0. Reports the requests lost and the time to recover. |

## Running it

```sh
export STRESS_API_KEY=...        # test key, raised limits
export PROM_TOKEN=...            # if Prometheus needs a bearer token
python cluster_stress.py --url https://GATEWAY --context-length 131072 --sessions 32 \
    --dataset oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json \
    --gpus 8 --gpu-model B200 --topology "2 pairs, 1P TP2 + 1D TP2 each" --image ghcr.io/...@sha256:... \
    --prom-url https://PROMETHEUS --prom-token-env PROM_TOKEN \
    --prom-prefill 'environment="candidate",role="prefill"' --prom-decode 'environment="candidate",role="decode"' \
    --fault-cmd 'kubectl -n inference delete pod inference-pair-0 --wait=false' \
    --check-cmd './check-node-gpus.sh' \
    --known-failures t_n2 --unsupported-params min_p \
    --soak-minutes 240 --out-dir stress-candidate
```

A colocated (non-PD) deployment passes `--prom-worker` instead of the prefill
and decode matchers. Run a single phase with `--phases needles`, and pass extra
options to one phase with `--<phase>-args "..."`, for example
`--overload-args "--levels 128,256 --burst 4000"`.

The result directory holds each phase's JSON and log, `report.json`, and
`report.md`, a table for a PR or a release note. The exit code is 1 if any phase
failed.

## Choices to make per deployment

- **Load bands.** Without Oracle decoding (production workers do not run
  `--oracle-decode`), each turn's history depends on what the model wrote, so
  the prefix hit and output share drift from the dataset's values. Run the load
  phase without `--oracle` and without bands (`--allow-failure` is implied) to
  record the numbers, or on a benchmark candidate started with
  `--oracle-decode` and with the realmix bands (hit 0.93-0.96, output share
  0.030-0.035). `--oracle-decode` lets a request dictate the output, so it must
  never be in a production allowlist.
- **Known failures and rejected parameters.** PD deployments do not support
  `n > 1` (`--known-failures t_n2`); speculative decoding returns no logprobs,
  which `t_logprobs_rejected` expects as a clean 4xx. DSpark also rejects
  `min_p`: pass `--unsupported-params min_p` so the soak and fault mixes leave
  it out instead of failing on it.
- **Errors inside a stream.** A streamed request refused after the 200 header
  arrives as an error event in the stream. Every tool here counts that, and a
  stream that ends without `[DONE]`, as a failed request.
- **Fault expectations.** With one replica every request during the reload
  fails, so leave `--max-failed-requests` unset and judge the recovery time.
  With two pairs the router should carry the traffic on the other pair: set
  `--fault-args "--max-failed-requests 0"` (or the in-flight count you accept).
- **Prometheus lag.** The scorecard waits `--prom-lag-s` (60 s) before each
  query, for scrape and remote-write delay. A measured window must not include a
  worker restart: the old pod's series stay visible for 5 minutes and would be
  added to the new one's.
