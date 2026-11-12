# Qwen3.5-0.8B on one GPU (staging)

The staging worker for confidential-inference: this repository's SGLang serving
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision
`2fc06364715b967f1860aea9cf38778875588b17` on one GPU. It is a real model
on the real engine and kernels, small enough that a worker starts in a few
minutes, so new kernels, optimizations and topology changes go through the
whole confidential-inference path here before they reach a production model.

| File | What it is |
| --- | --- |
| `container/Dockerfile` | The image: the DeepSeek-V4.1 image's Python environment, this checkout's SGLang, and the chart's helpers |
| `container/build.sh` | Builds it from a clean checkout, tagged by commit |
| `container/serve.sh` | Entrypoint for `docker run`: fetch the model, or verify it and serve |
| `container/model-manifest.json` | Byte manifest of the pinned revision (`fetch-model manifest` output) |
| `container/vendor/` | `wait-for-model`, `fetch-model` and the DCGM counters, copied from confidential-inference (`vendor/SOURCE`) |
| `chart-values.yaml` | confidential-inference values layer for this model on one GPU |

## Image

Published as `ghcr.io/confidential-dot-ai/sglang-qwen35-0.8b`, tagged with the
first 12 characters of the source commit. `/opt/sglang/SOURCE_RECORD.json` in
the image records the commit.

The base is `ghcr.io/confidential-dot-ai/dsv41-pd` by digest, the environment
(`recipes/deepseek-v41/container/requirements.lock`) tested on upstream
314bce7. Its layers are shared, so this image adds only a few hundred MB on
GHCR. The image replaces the base's SGLang source with this checkout and
drops the V4.1 recipe.

The image meets the chart's aggregated-worker contract: `wait-for-model`,
`fetch-model` and `dcgm-exporter` in `/usr/local/bin`, `python3` with SGLang on
`PATH`, user 1001 and `HOME=/tmp`.

```sh
recipes/qwen35-0.8b/container/build.sh                # sglang-qwen35-0.8b:<commit>
IMAGE=ghcr.io/confidential-dot-ai/sglang-qwen35-0.8b \
  recipes/qwen35-0.8b/container/build.sh
```

## Run with Docker

```sh
img=ghcr.io/confidential-dot-ai/sglang-qwen35-0.8b:<tag>
mkdir -p /data/models/qwen35-0.8b
# Download the pinned revision; every file is checked against the manifest.
docker run --rm --user 0 -v /data/models/qwen35-0.8b:/models/qwen35-0.8b "$img" fetch
# Verify every byte again, then serve on one GPU.
docker run -d --name qwen35 --gpus device=0 --shm-size 8g -p 30000:30000 \
  -v /data/models/qwen35-0.8b:/models/qwen35-0.8b:ro "$img"
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-0.8B","messages":[{"role":"user","content":"Say hi"}],"max_tokens":64}'
```

`docker run "$img" args` prints the `sglang.launch_server` argv. Extra
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

The deployment values add `images.sglangWorker` (this image, by digest) and
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
python3 recipes/qwen35-0.8b/container/vendor/fetch_model.py manifest \
  --repository Qwen/Qwen3.5-0.8B --revision <sha> > recipes/qwen35-0.8b/container/model-manifest.json
sha256sum recipes/qwen35-0.8b/container/model-manifest.json
```
