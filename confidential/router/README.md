# Router image

`ghcr.io/confidential-dot-ai/confidential-sglang/router`: this checkout's
SGLang router (`sgl-model-gateway/`, Python package `sglang_router`) in a
slim Python image. It holds no CUDA and no engine. The same wheel (the
`router-wheel` stage) is installed in the engine image, where the
DeepSeek-V4.1 launcher runs the router inside the pair pod.

| File | What it is |
| --- | --- |
| `Dockerfile` | The image. Context: the repository root. Build args `SOURCE_REVISION` (40-hex commit) and `SOURCE_DATE_EPOCH` (commit time) |
| `Dockerfile.dockerignore` | Keeps only `sgl-model-gateway/` and this directory in the context |
| `requirements.lock` | The runtime Python dependencies of `sglang_router`, every file by SHA-256 (from `requirements.in`) |
| `canonicalize_wheel.py` | Sorts the wheel's entries so two builds give the same wheel |
| `private_router.py` | `/usr/local/bin/ci-sglang-router`, the entrypoint (below) |
| `check-image.sh` | Checks a built image: uid 1001, read-only root, launcher present, no CUDA/torch, size |
| `run_router_tests.sh` | Runs the Rust tests (`cargo test --locked --lib` and `--test reliability_tests`) in the pinned build image |

Pins: the Rust 1.90.0 build image and the Python runtime image by digest,
Debian packages by one `snapshot.debian.org` timestamp (`DEBIAN_SNAPSHOT`),
crates by `sgl-model-gateway/Cargo.lock` (committed; the build uses
`--locked`), maturin by version and SHA-256, Python wheels by SHA-256.
`SOURCE_DATE_EPOCH` sets the router's build time. Build with BuildKit
`rewrite-timestamp=true`; two clean builds give the same digest.

Run:

```sh
python3 /usr/local/bin/ci-sglang-router ARGS     # what the chart runs (private output)
python3 -m sglang_router.launch_router ARGS       # the router itself
```

`ci-sglang-router` starts `python3 -m sglang_router.launch_router ARGS
--log-level warn --json-log`, reads the child's output in memory, and writes
only reviewed JSON records (fixed event names, role, HTTP status, exit code)
to stdout. It refuses `--log-dir`, `--enable-trace` and
`--otlp-traces-endpoint`.

## Changes to upstream's router

Upstream base: the fork's `main` (upstream `314bce7` plus this repository's
router commits). In this fork:

- Worker withdrawal, model routes, admission caps, the PD relay changes and
  a reproducible build time (`SOURCE_DATE_EPOCH`) came with the
  DeepSeek-V4.1 stack.
- An HTTP worker gets no router tokenizer: the router would otherwise load the
  worker's model path, which is not in its filesystem, from the Hub.
- The regular router does not count a worker's 4xx as a worker failure (408
  and 429 still count). A refused bad request must not open every worker's
  circuit breaker.
- PD request errors log fixed event names (`ci_event`), roles and status
  numbers, never worker error bodies, URLs or error text.
- Discovered prefill/decode pairs (below).

## Prefill/decode pairs from discovery

In prefill/decode mode with `--service-discovery`, a pod that carries both
annotations

```yaml
metadata:
  annotations:
    sglang.ai/prefill-port: "8041"
    sglang.ai/decode-port: "8043"
```

and matches `--prefill-selector` or `--decode-selector` (the paired layout
gives one label set to both) is one pair. Discovery registers its prefill
worker at `http://<pod IP>:<prefill-port>` (with the
`sglang.ai/bootstrap-port` annotation, as upstream) and its decode worker at
`http://<pod IP>:<decode-port>`, both with the worker label
`pd_pair=<pod name>`. The PD router then:

- selects a paired prefill only while its own decode worker is available;
- sends the request's decode leg only to that prefill's decode worker, also
  when decode is dispatched later (the decode gate);
- never gives an unpaired prefill a paired decode, or the reverse.

A pod with only one of the two annotations, the same port twice, or a port
that is not a number from 1 to 65535 is not registered at all (fail closed).
Pods without the annotations keep upstream discovery: one worker on
`--service-discovery-port`, typed by the selector it matches. In regular mode
(`--service-discovery --selector ...`) the annotations mean nothing.

A worker added through `POST /workers` with `labels: {"pd_pair": "<name>"}`
pairs the same way. This replaces the public image's `CI_ROUTER_PD_PAIRS`
environment value (exactly two fixed URL pairs).
