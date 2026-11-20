# Qwen3.5-0.8B on one GPU (staging)

The staging worker for confidential-inference: this repository's SGLang serving
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision
`2fc06364715b967f1860aea9cf38778875588b17` on one GPU. It is a real model
on the real engine and kernels, small enough that a worker starts in a few
minutes, so new kernels, optimizations and topology changes go through the
whole confidential-inference path here before they reach a production model.

| File | What it is |
| --- | --- |
| `container/serve.sh` | `docker run` helper: fetch the model, or verify it and serve (the chart's argv) |
| `container/model-manifest.json` | Byte manifest of the pinned revision (`fetch-model manifest` output) |
| `chart-values.yaml` | confidential-inference values layer for this model on one GPU |

## Image

This recipe runs on the one engine image,
`ghcr.io/confidential-dot-ai/confidential-sglang/engine`, built per `main`
commit by CI from `confidential/engine/` (see its README and
`docs/confidential-changes.md`). Pin it by digest from the release record
`images-<commit>`. The image holds this repository's SGLang (HTTP, media and
listener hardening applied), both recipes under `/opt/sglang/recipes`, and the
chart's helpers (`confidential/helpers/`): `wait-for-model`, `fetch-model` and
`dcgm-exporter` in `/usr/local/bin`, `python3` with SGLang on `PATH`, user 1001
and `HOME=/tmp`. It has no entrypoint; the chart sets every command.
`confidential/engine/check-image.sh IMAGE` checks those contracts without a GPU.

## Run with Docker

```sh
img=ghcr.io/confidential-dot-ai/confidential-sglang/engine@sha256:<digest>
serve=/opt/sglang/recipes/qwen35-0.8b/container/serve.sh
mkdir -p /data/models/qwen35-0.8b
# Download the pinned revision; every file is checked against the manifest.
docker run --rm --user 0 -v /data/models/qwen35-0.8b:/models/qwen35-0.8b "$img" "$serve" fetch
# Verify every byte again, then serve on one GPU.
docker run -d --name qwen35 --gpus device=0 --shm-size 8g -p 30000:30000 \
  -v /data/models/qwen35-0.8b:/models/qwen35-0.8b:ro "$img" "$serve" serve
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-0.8B","messages":[{"role":"user","content":"Say hi"}],"max_tokens":64}'
```

`docker run "$img" "$serve" args` prints the `sglang.launch_server` argv. Extra
arguments after `serve` are appended to it. `GPU_METRICS_PORT=29000` also
starts the DCGM exporter, as the chart does.

## Deploy with confidential-inference

Render the staging chart with `chart-values.yaml` after the staging layer:

```sh
helm template confidential-inference helm/confidential-inference \
  --values release/values.yaml --values release/staging/values.yaml \
  --values <confidential-sglang>/recipes/qwen35-0.8b/chart-values.yaml \
  --values deployment-values.yaml
```

The deployment values add `images.sglangWorker` (the engine image, by digest) and
the model source. That is either a C8s model volume with this revision
(`inference.model.c8sVolumeSource`, `volumeNames`) or the plain cache
(`inference.model.cache.enabled`, with `container/model-manifest.json` as
`cache.manifest`). The release's allowlist must admit the image and the
rendered argv, as for any worker change.

The worker flags follow `docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx`:
`qwen3_coder` tool calls, the `qwen3` reasoning parser, and the default
262,144-token context. They add the chart's fixed flags
(`--mem-fraction-static=0.90 --chunked-prefill-size=4096`). There is no
speculative decoding and no prefill/decode split.

## Updating the model

Change the revision and the hashes in `serve.sh` and `chart-values.yaml`, and
regenerate the manifest:

```sh
python3 confidential/helpers/fetch_model.py manifest \
  --repository Qwen/Qwen3.5-0.8B --revision <sha> > recipes/qwen35-0.8b/container/model-manifest.json
sha256sum recipes/qwen35-0.8b/container/model-manifest.json
```
