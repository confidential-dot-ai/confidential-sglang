# Engine image

`ghcr.io/confidential-dot-ai/confidential-sglang/engine`: one image per `main`
commit, tagged `sha-<commit>`, used by digest. It runs every model and layout
of confidential-inference: Qwen3.5-0.8B on one GPU (staging), DeepSeek-V4.1-Flash
PD pairs on B200 and B300, and the CPU simulator.

Built only by `.github/workflows/images.yml` (two clean builds on two runners,
equal digests, `check-image.sh`, then push and the release record). A local
build is for development, not a release.

## Contents

| Path | What |
| --- | --- |
| `/opt/dsv41` | Python 3.12 environment, `requirements.lock` (by hash). `python3` on `PATH` is `/opt/dsv41/bin/python`. |
| FlashInfer 0.7.3 | `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` with the `sm100a` (B200) and `sm103a` (B300) providers. The build fails unless both providers have `jit_cache/sampling/sampling.so`. |
| `/opt/sglang/python` | This checkout's SGLang, editable. HTTP server hardened with `confidential/helpers/harden_http.py --variant pd` (serving routes only). |
| `/opt/sglang/recipes/deepseek-v41` | Launcher, probes, docker recipe, load tests and stress suite. |
| `/opt/sglang/recipes/qwen35-0.8b` | Qwen3.5-0.8B staging recipe. |
| `/opt/sglang/layouts/<layout>.json` | Launch plans of the DeepSeek-V4.1 layouts (`confidential/layouts/README.md`). |
| `/opt/sglang/SOURCE_RECORD.json` | The source commit (`launch.py` reads it). |
| `/usr/local/bin/wait-for-model` | Model check and private supervisor of `python3 -m sglang.launch_server ...`. |
| `/usr/local/bin/fetch-model` | Model download into a plain cache. |
| `/usr/local/bin/sglang-pair` | Paired PD supervisor, with `pair_diagnostics.py` and `private_log_sink.py`. |
| `/usr/local/bin/dcgm-exporter` | NVIDIA DCGM exporter, `/etc/dcgm-exporter/counters.csv`. |
| `sglang_router` | The router wheel built from this checkout's `sgl-model-gateway` (the launcher runs it in the pod). The cluster router is the separate router image. |
| `/opt/confidential-inference/sglang-simulator` | CPU simulator for `inference.mode: simulator`. |

uid 1001, `HOME=/tmp`, nothing writable under `/opt`, no `ENTRYPOINT`, no
`CMD`. The chart names every command, for example
`/usr/local/bin/wait-for-model ... -- python3 -m sglang.launch_server ...`.

## Pins

| Input | Pin |
| --- | --- |
| CUDA base | `docker.io/nvidia/cuda@sha256:7d56eb...` (13.0.3-devel-ubuntu24.04, amd64) |
| Ubuntu packages | `snapshot.ubuntu.com` `20261104T000000Z` |
| uv | `ghcr.io/astral-sh/uv@sha256:265d07...` (0.11.16) |
| DCGM exporter | `nvcr.io/nvidia/k8s/dcgm-exporter@sha256:148b0c...` (4.8.4) |
| Python | `requirements.lock`: every PyPI file hash; FlashInfer kernels by URL and SHA-256 (`flashinfer-wheels.txt`) |
| Router wheel | Rust base by digest, `cargo --locked` (`sgl-model-gateway/Cargo.lock`) |
| Build | BuildKit by digest, `SOURCE_DATE_EPOCH` = commit time, `rewrite-timestamp=true`, no provenance or SBOM |

Change a Python pin in `requirements.in`, then run `lock.sh`.

## Check

`check-image.sh IMAGE` runs the image as the chart does (uid 1001, read-only
root, no network) and checks the routes, files, helpers, FlashInfer kernels,
layouts and permissions. CI runs it on build #1 before publishing.
