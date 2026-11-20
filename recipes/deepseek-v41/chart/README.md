# DeepSeek-V4.1-Flash on the confidential-inference chart

`../container` is the docker recipe: one root image that `compose.yaml` and
`deploy-dsv41.sh` run on a GPU VM. Under the confidential-inference Helm chart
the same engine runs from the one engine image,
`ghcr.io/confidential-dot-ai/confidential-sglang/engine`, which CI builds per
`main` commit from `confidential/engine/` (pin it by digest from the release
record `images-<commit>`). The chart runs every container as uid 1001 with its
own command line; the image has no entrypoint.

```
confidential/engine/check-image.sh ghcr.io/confidential-dot-ai/confidential-sglang/engine@sha256:<digest>
```

## What the engine image holds for this recipe

- The `/opt/dsv41` environment (torch, FlashInfer with the sm100a/sm103a
  prebuilt kernels and cubins, deep-gemm, Mooncake), this repository's SGLang
  at `/opt/sglang/python`, and `/opt/sglang/recipes/deepseek-v41`.
  `/opt/sglang/SOURCE_RECORD.json` records the commit.
- The chart's helpers from `confidential/helpers/`: `wait-for-model`,
  `fetch-model`, `sglang-pair` with `pair_diagnostics.py` and
  `private_log_sink.py`, and `dcgm-exporter` with its counters.
- The hardening of `confidential/helpers/`: `harden_http.py --variant pd`
  (serving routes only, no docs, only the Python HTTP frontend, and startup
  settings that log or dump request data refused), `harden_media.py` (inline
  request media only) and `harden_dist.py` (internal listeners on loopback,
  single node). Each script checks the SHA-256 of the files it patches, so a
  change to any of them fails the build until the policy is reviewed again
  (`confidential/helpers/CHANGES.md` records the reviews).
- The baked launch plans of `../container/*.env` in `/opt/sglang/layouts/`.
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
the image's SGLang must carry `pair_progress`; without it prefill comes up
healthy and decode is never started. Because prefill and decode share one pod,
`harden_dist.py` keeps the KV bootstrap server and the distributed stores on
loopback.

`check-image.sh` checks all of them without a GPU. It runs as uid 1001, with a
read-only root filesystem and no network. It also checks the hardened route
table, the media and listener policies, the FlashInfer kernels, the baked
layouts and the source record.

## Tested and not tested

- Engine: this repository's SGLang ran on the 4x B200 in the docker recipe
  (functional suite and realmix load).
- The hardened engine image has not booted on a GPU under the chart yet; the
  first staging deploy is that test.
