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
| func | `func_tests.py` | Tool calls (11 cases x 10), API features (JSON schema, stop, max tokens, thinking, unicode, history), needles up to the context length, over-context prompt gets a 4xx, a storm of client disconnects leaves the server healthy. `t_logprobs`, `t_logprobs_stream` and `t_min_p` check the declared behavior: a parameter in `--unsupported-params` must get a clean client error (4xx other than 408/429 with an error body, or a 200 stream that holds only a 4xx error event), never a 5xx, a dropped connection or a timeout, and the server must answer the next request; any other parameter must work (logprobs with 3 alternatives). Only `--known-failures` may fail. |
| needles | `needles.py` | Every one of the fixed long-prompt recall cases (default: the 64 cases that caught the CED tail-graph bug, 14K-60K tokens) returns its code. |
| load | `oracle_scorecard.py` | Realmix agent traces at `--sessions` in flight; in the measured window: 0 failed requests and KV-transfer failures, prefix hit and output share inside the dataset's bands, TTFT p50 at most 3 s, 0 retractions. Reports served TPM per GPU. |
| overload | `overload.py` | 64 to 512 clients of short requests, then a burst: only 429s refuse the excess (no 5xx, connection errors or wrong answers), and a request succeeds within 2 minutes afterwards. The report says whether any step was refused at all (`shed`); if none was, the levels never reached capacity and the refusal path is untested (`--require-shed` makes that a failure). `--shed-codes` names the error codes of a capacity shed (the gateway's `inference_capacity`): then only those count, so a 429 of a per-address cap or of the key's own limit does not prove a shed. Each step lists its 429 codes (`codes_429`). |
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
    --known-failures t_n2 --unsupported-params logprobs,min_p --require-metrics \
    --soak-minutes 240 --out-dir stress-candidate
```

A colocated (non-PD) deployment passes `--prom-worker` instead of the prefill
and decode matchers. Run a single phase with `--phases needles`, and pass extra
options to one phase with `--<phase>-args "..."`, for example
`--overload-args "--levels 128,256 --burst 4000"`.

The result directory holds each phase's JSON and log, `report.json`, and
`report.md`, a table for a PR or a release note. The exit code is 1 if any phase
failed.

`report.json` names the suite's source (`suite.commit`, from the checkout or
`$STRESS_SUITE_COMMIT`, and `suite.dirty`). With `--prom-url` (or the direct
`--metrics-*` URLs) each phase's result holds `metrics`: the server's window
over that phase (TTFT and ITL quantiles, served tokens, KV use, preemptions,
router non-2xx, KV-transfer failures), read after the run at the recorded phase
boundaries once `--prom-lag-s` has passed. `metrics.errors` lists any window
that could not be read; `--require-metrics` makes that a failure.

Every model behavior is an option (`--context-length`, `--thinking-kwarg`,
`--thinking-default`, `--known-failures`, `--unsupported-params`). A validation
(confidential-inference-internal `bin/validate-deployment --stress`) passes each
one from the release recipe and the environment's `validation.yaml` and uses no
profile, so the same suite validates any model.

## Profiles

`--profile` sets the defaults for one kind of deployment. Any option on the
command line still wins over the profile.

| Profile | For | Phases | Settings | Run time |
|---|---|---|---|---|
| (none) | the tools' own defaults | all | 131072 context, 32 sessions, 60-minute soak | hours |
| `qwen-short` | a 1-GPU Qwen3.5-0.8B staging deployment (`recipes/qwen35-0.8b`) | func, needles, overload, soak | 262144 context; func with 4 trials, needles up to 253952 tokens (context - 8192); needles phase at 16K-240K tokens, calibrated on the served tokenizer; overload at 16-128 clients (30 s steps), burst 500; 8 sessions, 5-minute soak, prompts up to 48K | about 15-20 minutes (estimate; not yet run on staging) |
| `dsv41-full` | DeepSeek-V4.1 PD with DSpark | all (load needs `--dataset`, fault needs `--fault-cmd`) | 131072 context, 32 sessions, `--known-failures t_n2 --unsupported-params logprobs,min_p`, 240-minute soak | about 5 hours |

`qwen-short` leaves out the load phase: its realmix dataset and bands are
DeepSeek-V4.1 traces. Qwen3.5-0.8B has no speculative decoding, so it serves
logprobs and min_p, and `t_logprobs` and `t_min_p` require that. The 0.8B model's tool-call and reasoning accuracy was not measured
before this profile was written: add `--known-failures` for any case the first
staging run shows to be a model limit, not a serving fault.

## Running from the engine image

The engine image (`ghcr.io/confidential-dot-ai/confidential-sglang/engine`)
carries this directory at `/opt/sglang/recipes/deepseek-v41/loadtest`, and its
Python environment (`python3`) has every dependency of the tools (aiohttp; and
tokenizers for `--oracle`). It does not carry the realmix dataset or kubectl:
mount a dataset for the load phase, and give the fault phase a command that
exists in the container. The image runs as uid 1001 with a read-only `/opt`, so
write results to a mounted directory.

```sh
mkdir -p out && chmod 777 out
IMAGE=ghcr.io/confidential-dot-ai/confidential-sglang/engine@sha256:...

# Qwen3.5-0.8B staging, ~15-20 minutes
docker run --rm -e STRESS_API_KEY -v "$PWD/out:/out" --entrypoint python3 "$IMAGE" \
    /opt/sglang/recipes/deepseek-v41/loadtest/cluster_stress.py --profile qwen-short \
    --url https://GATEWAY --api-key-env STRESS_API_KEY --out-dir /out/qwen-short

# DeepSeek-V4.1, full profile (load phase with a mounted dataset)
docker run --rm -e STRESS_API_KEY -e PROM_TOKEN -v "$PWD/out:/out" -v "$PWD/data:/data-in:ro" \
    --entrypoint python3 "$IMAGE" \
    /opt/sglang/recipes/deepseek-v41/loadtest/cluster_stress.py --profile dsv41-full \
    --url https://GATEWAY --dataset /data-in/oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json \
    --gpus 8 --gpu-model B200 --topology "2 pairs, 1P TP2 + 1D TP2 each" --image "$IMAGE" \
    --prom-url https://PROMETHEUS --prom-token-env PROM_TOKEN \
    --prom-prefill 'environment="candidate",role="prefill"' --prom-decode 'environment="candidate",role="decode"' \
    --out-dir /out/dsv41-full
```

As a Kubernetes Job beside the cluster under test (any cluster that can reach
the gateway; the key comes from a Secret, results go to the pod's log and an
emptyDir — copy `report.md` out of the log, or mount a volume you can read):

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cluster-stress-qwen-short
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      securityContext:
        runAsUser: 1001
        runAsGroup: 1001
        runAsNonRoot: true
        seccompProfile: {type: RuntimeDefault}
      containers:
        - name: stress
          image: ghcr.io/confidential-dot-ai/confidential-sglang/engine@sha256:...
          command: ["python3", "/opt/sglang/recipes/deepseek-v41/loadtest/cluster_stress.py"]
          args: ["--profile", "qwen-short", "--url", "https://GATEWAY",
                 "--api-key-env", "STRESS_API_KEY", "--out-dir", "/out/qwen-short"]
          env:
            - name: STRESS_API_KEY
              valueFrom: {secretKeyRef: {name: stress-api-key, key: key}}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: ["ALL"]}
          resources:
            requests: {cpu: "2", memory: 2Gi}
          volumeMounts:
            - {name: out, mountPath: /out}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - {name: out, emptyDir: {}}
        - {name: tmp, emptyDir: {}}
```

The Job needs no GPU. Run it outside the cluster under test (or from any pod
that is allowed to reach the gateway), never on a production cluster.

## Running from the cluster-stress image

`container/build.sh DATASET TOKENIZER_DIR` builds `cluster-stress:<revision>`:
the tools, hash-pinned aiohttp and tokenizers, kubectl, and the realmix
dataset with its MiniMax-M3 tokenizer (`--dataset` and `--source-tokenizer`
default to them). The unit tests run during the build. It runs as uid 1000
and writes results to `/work`:

```sh
mkdir -p out && chmod 777 out
docker run --rm -e STRESS_API_KEY -e PROM_TOKEN -v "$PWD/out:/work" \
    -v "$HOME/.kube/config:/home/stress/.kube/config:ro" \
    cluster-stress:REVISION --url https://GATEWAY ...   # same options as above
```

Mount a CA bundle and pass `--ca-cert` for a private CA, and use
`--network host` when the endpoint is on the machine running the container.

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
  `n > 1`: behind a gateway that refuses it (confidential-inference #369),
  pass `n` in `--unsupported-params` and `t_n2` requires a clean client error;
  straight to the router, `--known-failures t_n2`. Speculative decoding returns
  no logprobs, and DSpark also rejects `min_p`: pass `--unsupported-params
  logprobs,min_p`. `t_logprobs` and `t_min_p` then require a clean client error for them, and
  the soak and fault mixes leave min_p out instead of failing on it.
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

## First run (2026-11-16, 4x B200, one 1P1D pair)

High-RAM 4x B200, prefill TP2 + decode TP2 with DSpark, 131K context, image
`dsv41-pd@sha256:d3a97f8c...` with `--oracle-decode --oracle-commit-len 5`,
through the router with a placeholder Bearer key (`--known-failures t_n2
--unsupported-params min_p`). Every phase passed:

| Phase | Numbers |
|---|---|
| func | 159/160 (the `n=2` case, known on PD) |
| needles | 0 of 64 missed, up to 60,287 prompt tokens |
| load | realmix at 32 sessions, bands enforced: 2.65M served TPM/GPU, hit 0.945, TTFT p50/p99 0.45/0.84 s |
| overload | 512, 1024 and 2048 clients, then 4000 at once: 44-50 completed req/s held, the excess refused with 429 only, 0 5xx or connection errors, healthy 0.4 s later |
| soak | 15 min at 32: 5,485 requests, 0 failed, flat latency per minute |

The fault phase ran only against the fake gateway in `test_cluster_stress.py`.
At 64-256 clients the router queued everything (no 429): its admission cap is
192 in flight plus a queue of 256, so shedding needs more clients than that.
