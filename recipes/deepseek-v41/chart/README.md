# DeepSeek-V4.1-Flash worker image for confidential-inference

`../container` is the docker recipe: one root image that `compose.yaml` and
`deploy-dsv41.sh` run on a GPU VM. This directory builds the same engine as the
worker image of the confidential-inference Helm chart, which runs every
container as uid 1001 with its own command line.

```
IMAGE=ghcr.io/confidential-dot-ai/sglang-dsv41-flash recipes/deepseek-v41/chart/build.sh
recipes/deepseek-v41/chart/check-image.sh ghcr.io/confidential-dot-ai/sglang-dsv41-flash:<commit>
```

## What is in it

- The base is the PD recipe image (`BASE_IMAGE`, default
  `dsv41-pd@sha256:8e1b54fb…`): Ubuntu 24.04, CUDA 13.0.3, the
  `requirements.lock` environment in `/opt/dsv41` (torch, FlashInfer 0.7.3 with
  sm100a/sm103a kernels, deep-gemm, Mooncake). This image changes none of it.
- `/opt/sglang/python` and `/opt/sglang/recipes/deepseek-v41` come from this
  checkout. `SOURCE_RECORD.json` records the commit.
- The chart's helpers, copied from confidential-inference
  (`vendor/SOURCE`): `wait-for-model`, `fetch-model`, `sglang-pair` with
  `pair_diagnostics.py` and `private_log_sink.py`, and `dcgm-exporter` with its
  counters.
- The HTTP server is hardened with confidential-inference's
  `harden_http.py --variant pd`, as its own images are. Only the serving routes
  remain (health, OpenAI, generate and the other inference endpoints). The
  admin, profiling, weight-update, cache-flush and docs routes are removed, and
  `launch_server` refuses every frontend except the Python HTTP one. The script
  checks the SHA-256 of `http_server.py` and `launch_server.py` first, so a
  change to either fails the build until someone reviews the policy against
  it. The vendored copy pins this branch's files (the upstream 314bce7 sync).
  confidential-inference pins upstream a7fd2e1, which its own image builds
  from. `vendor/SOURCE` records the review.
- uid 1001, `HOME=/tmp`, `/opt` read-only to the runtime user, no entrypoint.

## Chart contracts

| Chart mode | What the chart runs from this image |
| --- | --- |
| aggregated (`inference.pd.enabled: false`) | `wait-for-model … -- python3 -m sglang.launch_server …`, `fetch-model`, `dcgm-exporter` |
| PD (`inference.pd.enabled: true`) | `/opt/dsv41/bin/python` (`inference.pd.python`), `launcher/hold_engram_cache.py` (`inference.pd.engramHolder.script`), `/bin/sh` and `curl` for the start gate |
| paired PD (`inference.pd.paired.enabled: true`) | `/opt/dsv41/bin/python /usr/local/bin/sglang-pair`, and the per-rank progress records of `sglang.srt.utils.pair_progress` |

Paired PD is the v0 layout: one pod per prefill/decode pair, both workers in
one `sglang-pair` container, KV over NVLink (CUDA IPC) inside the pod. CUDA
IPC needs the importing process to share the exporter's PID namespace and see
its GPU, which separate containers or pods do not have under C8s. `sglang-pair`
marks a worker ready only after its scheduler ranks write progress records, so
the image's SGLang must carry `pair_progress` (this branch's base, #5); without
it prefill comes up healthy and decode is never started.

`check-image.sh` checks all of them without a GPU. It runs as uid 1001, with a
read-only root filesystem and no network. It also checks the hardened route
table and the source record.

## Tested and not tested

- Engine: this branch's SGLang ran on the 4x B200 in the docker recipe
  (functional suite and realmix load).
- The image itself has not booted on a GPU under the chart. The first staging
  deploy is that test. confidential-inference#300 selects it in the 8-GPU B200
  PD layer. Every worker argv of that layer parses with this branch's
  `ServerArgs`.
