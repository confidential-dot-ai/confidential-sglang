# confidential/helpers: what moved here, from where, and why

The chart's worker helpers and the build-time hardening of the engine image
live only in this repository now. They came from
`confidential-dot-ai/confidential-inference` `images/sglang-confidential/` at
public `main` `0d27717` (#334, "Worker: report run-time kernel compiles
(jit_compile)", which merged while this was being done; it includes #315
structured log events and #316 refused logging settings), and from this
repository's own `recipes/deepseek-v41/chart/vendor/`.

## Files

| Here | From | SHA-256 here | Change |
| --- | --- | --- | --- |
| `wait_for_model.py` -> `/usr/local/bin/wait-for-model` | public `0d27717` | `45efebc0…ccc37a6c` | unchanged (has the `jit_compile` watcher) |
| `fetch_model.py` -> `/usr/local/bin/fetch-model` | public `0d27717` | `84d2144c…3f002b23e` | unchanged |
| `dcgm-counters.csv` -> `/etc/dcgm-exporter/counters.csv` | public `0d27717` | `adf12881…d8329aef` | unchanged |
| `harden_http.py` (build time, `--variant pd`) | public `0d27717` (`6796a87c…`) | `19eff6b4…31955ee` | `pd` hashes re-pinned to this fork's files; the unused `ordinary` variant (upstream 71de97b) removed |
| `harden_media.py` (build time) | public `0d27717` (`22e161e7…`) | `ea18f7a0…bfb357d` | ported to upstream's split of `srt/utils/common.py` (#48781); hashes re-pinned |
| `harden_dist.py` (build time) | public `0d27717` (`a119ff2f…`) | `925bc1db…a7bade49b` | hashes re-pinned to this fork's files; the old `release`/`diagnostic-fork` sets removed |
| `pair_supervisor.py` -> `/usr/local/bin/sglang-pair` | `recipes/deepseek-v41/chart/vendor/` (public a59231a `images/sglang/`) | `51769868…40ddb80a5` | unchanged |
| `pair_diagnostics.py` -> `/usr/local/bin/pair_diagnostics.py` | same | `a5e0b806…99f78fa61` | unchanged |
| `private_log_sink.py` -> `/usr/local/bin/private_log_sink.py` | same (public `images/sglang-router/`) | `1a865cfd…f06bd09` | unchanged |
| `../simulator/upstream/` | public `0d27717` `images/sglang-confidential/simulator-upstream` (sgl-project/sglang PR 33824 at `5e6af31`) | same git tree as public | unchanged |
| `../simulator/patches/sglang-simulator-{replay-only,parser-flags-noop,prefix-length}.patch` | public `0d27717` `images/sglang-confidential/patches/` | `7b911611…`, `9c654801…`, `e819be5b…` | unchanged |
| `../simulator/patches/sglang-simulator-fork-imports.patch` | new | `1a419145…7cf51f1` | see "Simulator" |
| `../tests/helpers/` | public `0d27717` `tests/images/sglang-confidential/`, and the pair tests public removed in #310 (`10bb59d^` `tests/images/sglang/test_pair_*.py`, `pair_fake_worker.py`) | | paths changed; sources read from this checkout instead of downloaded |
| `../engine/check-image.sh` | `recipes/deepseek-v41/chart/check-image.sh` | | extended to the one engine image |

Removed, because the one engine image (`confidential/engine/`) replaces them:
`recipes/deepseek-v41/chart/{Dockerfile,Dockerfile.dockerignore,build.sh,check-image.sh,vendor/}`
and `recipes/qwen35-0.8b/container/{Dockerfile,build.sh,vendor/}`. The two
vendor directories held duplicate (older) copies of the helpers above. The
recipe READMEs now point at `ghcr.io/confidential-dot-ai/confidential-sglang/engine`.
`recipes/qwen35-0.8b/container/serve.sh` stays as a `docker run` helper; the
image has no entrypoint, so it is run as
`docker run IMAGE /opt/sglang/recipes/qwen35-0.8b/container/serve.sh serve`.

Not moved: public `test_recipe.py` (it tests the public Dockerfile, which this
repository does not use) and `test_no_model_weights_in_git.py` (it reads the
public repository's deployment fixtures). `smoke.py` (CPU simulator smoke on a
built image, needs Docker), `inference_check.py` and `mooncake_integrity.py`
(GPU tools) moved as they are; pytest does not collect them.

## Review: harden_http.py against this fork

Files: `python/sglang/srt/entrypoints/http_server.py` `fcd651b9…` and
`python/sglang/launch_server.py` `0d47833b…` (the hashes the fork's vendored
copy already pinned after the upstream 314bce7 sync; the files are unchanged
since). The current public policy adds the `launch_server` refusal of startup
settings that log or dump request data (#316, `REFUSED_SERVER_ARGS`). All ten
refused fields exist in this fork (`srt/arg_groups/fields/observability.py`,
`spec.py`) and default to off; `ServerArgs` is still the flat dataclass, so
`getattr(server_args, name)` reads them.

Applied to this fork: 91 route declarations go to 41, the serving set
(health, ready, model/server info, generate/encode/classify, OpenAI
completions/chat/embeddings/tokenize/detokenize/audio/realtime/models/score/
responses/rerank, Ollama, Anthropic messages, SageMaker `/ping`, Vertex).
Upstream 314bce7 no longer has the `get_load` and `sagemaker_chat_completions`
(`/invocations`) handlers. New admin routes such as `/plugin_weight_update` and
`/get_hicache_l3_cache_layout` are removed. `app.include_router(v1_loads_router)`
stays (`/v1/loads`, as in public); the elastic EP router is removed. OpenAPI,
docs and redoc are off. `launch_server.run_server` refuses every frontend
except the Python HTTP one (Rust server, gRPC, encoder-only, Ray,
`DUMPER_SERVER_PORT`).

## Review: harden_media.py against this fork

Upstream #48781 (in 314bce7) split `srt/utils/common.py`: the media loaders
(`_assert_media_url_allowed`, `download_remote_media`, `get_mm_http_session`,
`load_audio`, `_load_image`, `load_image`, `get_image_bytes`,
`_normalize_video_input`, `get_video_bytes`) moved unchanged to
`srt/multimodal/media_io.py`, and the processors import them from there. The
set of files that fetch URLs or open request-named files is otherwise the same
as at a7fd2e1 (checked by grepping both trees for downloads, `file://`,
`urlopen`, sessions and video/audio loaders). Port:

- every `srt/utils/common.py` key is now `srt/multimodal/media_io.py`; the
  injected code imports `inline_media_bytes`/`inline_media_container` from
  `sglang.srt.multimodal.media_io`; the injected block imports `re`, which
  `media_io.py` does not;
- the import-line edits of `mimo_v2.py`, `moss_vl.py`, `inkling.py`,
  `bailing_mm.py` and `mimo_audio.py` (`media_io.` instead of `common.`) match
  the new import lines;
- `inkling.py` now resolves URLs inside `try/except CLIENT_MEDIA_EXCEPTIONS`;
  the edit replaces the data:/http(s) branches inside that `try` with
  `return inline_media_bytes(url)`;
- `inkling/image_processing.py` and `inkling/feature_extraction.py` now wrap
  the file read in `try/except OSError`; the edit replaces the whole
  read-and-handle block with the inline-only `ValueError`.

The policy itself is unchanged: inline bytes, `data:` URL or base64 only;
Pillow opens only JPEG/PNG/WEBP/GIF; FFmpeg gets only known binary containers
without playlist or manifest markers; the Transformers backend refuses video.
All 13 files apply; `pyflakes` reports no undefined names in the results.

## Review: harden_dist.py against this fork

All six files differ from a7fd2e1 (`bootstrap.py` NPU init and EPLB warmup;
`network.py` the ZMQ peer allowlist and source-IP pin; `disagg_service.py` the
Rust-server bootstrap helpers; large PD/CED changes in `decode.py` and
`common/conn.py`). Every edit applies unchanged: each AST anchor and text edit
is found exactly once. The edited lines have the same meaning:
`_resolve_dist_init_method`, `init_parallel_runtime`,
`_launch_subprocesses -> server_args.resolve_once()`, `get_local_ip_auto`,
`bind_port`, the bootstrap server `host=`, decode's
`NetworkAddress(req.bootstrap_host, …)` and prefill's
`register_to_bootstrap`. Every field `_refuse_non_local_listeners` reads exists
(`nnodes`, `node_rank`, `dist_init_addr`, `gated_launch_port`,
`elastic_ep_backend`, `moe_a2a_backend`, `weight_cache_mode`,
`remote_instance_weight_loader_start_seed_via_transfer_engine`,
`kv_events_config`, `forward_pass_metrics_ipc_name`,
`SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE`).

Pair launch: `recipes/deepseek-v41/launcher/launch.py` gives every worker
`--host 127.0.0.1`, its own `--nccl-port` and bootstrap port, and no
`--dist-init-addr`, `--nnodes`, kv-events, forward-pass-metrics IPC, NIXL or
elastic EP, so the refusals do not trigger. Prefill and decode run in one pod
(`sglang-pair`) or one container (docker recipe), so loopback bootstrap,
stores and local IP reach each other. Not covered by a GPU run yet: Mooncake
with `get_local_ip_auto` = 127.0.0.1 (intra-node NVLink and RDMA handshake on
loopback). The first GPU boot of the engine image is that test. Listeners that
the fork adds outside these six files (the live metrics sidecar, the Engram
holder) are not part of this policy.

## Simulator

The simulator's hooks were checked statically against this fork: every hooked
module and class exists, with the same methods as at a7fd2e1 (the generic
host-pool hook now matches 15 classes instead of 12). One import broke: the
CPU allocator imports `get_num_new_pages` from `sglang.srt.utils`, which
upstream #48781 moved to `sglang.srt.mem_cache.utils` (same signature).
`sglang-simulator-fork-imports.patch` changes that import. A CPU run of the
simulator on the built image (`confidential/tests/helpers/smoke.py`) has not
been done here; it needs the image.

## Tests

`python3 -m pytest confidential/tests/helpers` (no GPU, no network, no
SGLang import; pytest, fastapi, httpx, pillow; the two ASGI tests skip without
fastapi/httpx): 174 passed (wait-for-model, fetch-model, server output and
signals, HTTP/media/listener policy against this checkout's source, simulator
parser no-op, pair supervisor/diagnostics/log and snapshot delivery against
this checkout's `pair_progress` and `pair_nccl_diagnostics`).
