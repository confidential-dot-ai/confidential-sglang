# DeepSeek-V4.1 launch plans (`/opt/sglang/layouts`)

The engine image carries one launch plan per DeepSeek-V4.1 layout of the docker
recipe. A plan is the exact output of `launch.py --dry-run` for that layout:
every worker's and the router's command line and environment, with the
engine image's container paths. It lets a deployment (or a reviewer) see what
a layout runs without a GPU, without the model, and without the host the
recipe was tuned on.

Built at image build time, from the repository root:

```sh
python3 -I -B confidential/layouts/build_layouts.py --out /opt/sglang/layouts
```

`build_layouts.py` uses only the standard library. It never imports SGLang,
reads no GPU, no `/proc/meminfo`, and no host path. The output depends only on
the source tree and the source revision, so two builds of one commit are
byte-identical (`confidential/tests/layouts/test_build_layouts.py` checks it).
The revision is `--revision`, else `$SOURCE_REVISION`, else
`SOURCE_RECORD.json` at the repository root, else `git rev-parse HEAD`.

## Layouts

| Layout (`<layout>.json`) | Source profile | Target | Topology |
| --- | --- | --- | --- |
| `dsv41` | `recipes/deepseek-v41/container/dsv41.env` | 4x B200 | 1 pair: TP2 prefill + TP2 decode, 128K context |
| `dsv41-262k` | `dsv41-262k.env` | 4x B200 | 1 pair, 262144-token context |
| `dsv41-1m` | `dsv41-1m.env` | 4x B200, ~880 GiB host RAM | 1 pair, 1M context, MXFP4 Engram, B200 memory estimate in the plan (unqualified) |
| `dsv41-b200-1m` | `dsv41-b200-1m.env` | 8x B200 | 2x TP2 prefill + 2x TP2 decode, 1M context |
| `dsv41-b300-1m` | `dsv41-b300-1m.env` | 8x B300 | 2x TP2 prefill + 2x TP2 decode, 1M context, B300 memory profile |
| `dsv41-b300-1m-ced` | `dsv41-b300-1m-ced.env` | 8x B300 | 2x PP2xTP1 CED prefill + 2x TP2 decode, 1M context |

Every `*.env` in that directory becomes a layout; a new profile needs no change
here unless it uses `--prefill-hicache-ratio auto` (see `host_available_gib`).

## How a profile becomes a plan

The arguments are those the docker recipe passes (`entrypoint.sh pd` ->
`tp2-pd.sh` -> `_common.sh run_launch`), in the same order:

1. `--model`, `--port`, `--bootstrap-port`, `--nccl-port`,
   `--router-metrics-port` (the profile's `PORT`, `BOOTSTRAP_PORT`, `NCCL_PORT`,
   `ROUTER_METRICS_PORT`);
2. `--cpu-engram-staging --split-cuda-graphs --output-dir` (`tp2-pd.sh`);
3. `--resident-engram-pid` (the recipe always runs with the Engram holder);
4. the profile's `DSV41_LAUNCH_ARGS` and `DSV41_EXTRA_ARGS`, minus the flags in
   step 5;
5. container values: `--python`, `--install-root`, `--kernel-cache-dir`,
   `--source-revision`, and `--host-available-gib` when HiCache is `auto`.

`--gpus` is not passed: inside a pod the visible GPUs are `0..N-1`, which is
the launcher's default. The profile's host GPU indices are kept only as
`target.host_gpus`.

## Container paths

| Path | What |
| --- | --- |
| `/opt/sglang` | Install root; workers run with `PYTHONPATH=/opt/sglang/python` |
| `/opt/dsv41/bin/python` | The image's Python environment (also `python3` on `PATH`) |
| `/opt/sglang/recipes/deepseek-v41/launcher/launch.py` | The launcher; `launch.py <launch_argv>` reruns the plan |
| `/opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py` | The Engram holder the plans assume is running |
| `/mnt/c8s-data/models/deepseek-v41-flash` | The model. The chart mounts each worker's model at `inference.model.volumeDirectory` (`/mnt/c8s-data/models`) / volume name; the plans use volume name `deepseek-v41-flash`. A deployment with another name replaces every occurrence of `paths.model` |
| `/tmp/dsv41/kernel-cache` | Triton, inductor, FlashInfer, CUDA and SGLang kernel caches (writable; the chart runs with `HOME=/tmp` and a read-only root) |
| `/tmp/dsv41/run` | The launcher's output directory (launch plan, preflight, logs) |

No plan contains `/data` (the recipe's host paths); the build fails if one does.

## Schema (`confidential-sglang/dsv41-layout/v1`)

JSON object, keys sorted, two-space indent, trailing newline.

| Key | Type | Meaning |
| --- | --- | --- |
| `schema` | string | `confidential-sglang/dsv41-layout/v1` |
| `layout` | string | Layout name = profile file stem |
| `source_env` | string | Profile path relative to the repository root |
| `revision` | string | 40-hex source commit the plan was built from |
| `target.gpu_count` | int | GPUs the layout uses (prefill workers x TP + decode workers x TP) |
| `target.gpu_models` | list of string | GPU name substrings the launcher's preflight accepts (`--gpu-models`) |
| `target.memory_profile` | string or null | `--memory-profile` (`B200`/`B300`) used for `auto` pool sizes, or null |
| `target.host_available_gib` | number or null | Host RAM assumed for `--prefill-hicache-ratio auto` (null when not auto). B200 layouts: 768 GiB (an 8x B200 host with 1 TiB, less the ~198 GiB Engram copy the holder pins; for `dsv41-1m`, a 4x B200 with ~880 GiB less the ~98 GiB MXFP4 Engram copy and the guest) |
| `target.host_gpus` | list of int or null | The profile's host GPU indices (docker recipe only; not used in the plan) |
| `paths.*` | string | `install_root`, `python`, `launcher`, `engram_holder`, `model`, `kernel_cache_dir`, `output_dir` (table above) |
| `launch_argv` | list of string | `launch.py` arguments (without `--dry-run`) that produce `plan` |
| `plan` | object | `launch.py --dry-run` output, below |

`plan` (from `recipes/deepseek-v41/launcher/launch.py build_plan`):

| Key | Type | Meaning |
| --- | --- | --- |
| `revision` | string | Same as the top-level `revision` |
| `status` | string | Always `unqualified` (a plan is not a measured profile) |
| `physical_gpus` | list of int | GPU indices, `0..gpu_count-1` |
| `topology` | string | Human summary, e.g. `2x(PP2xTP1 CED prefill) + 2x(TP2 decode); CPU Engram; DP=1 per worker` |
| `collectives` | object | `custom_all_reduce` (`on`/`off`), `flashinfer_allreduce_fusion` (`off`) |
| `workers` | list of object | Prefill workers first, then decode. Each: `name` (`prefill`, `prefill0`, `decode1`, ...), `command` (argv, `python -m sglang.launch_server ...`), `env` (the worker's full environment additions: `CUDA_VISIBLE_DEVICES`, caches, `SGLANG_*`, `NCCL_*`, Mooncake), `url` (`http://127.0.0.1:<port>`), and `bootstrap_port` (int, prefill only, when there are several prefills) |
| `router` | object | `name` (`router`), `command` (`python -m sglang_router.launch_router --pd-disaggregation ...`), `env`, `url` |
| `hicache_auto` | object, optional | Present when HiCache is `auto`: `host_available_gib`, `engram_gib`, `host_reserve_gib`, `fits_ratio`, `ratio` (null = HiCache off) |
| `memory` | object, optional | Present with `--memory-profile`: `profile`, `profile_measured`, `per_rank.{prefill,decode}` (HBM budget per rank: `hbm_gib`, `weights_gib`, `graphs_gib`, `transient_reserve_gib`, `kv_bytes_per_token`, `pool_tokens`, `pool_gib`, `max_pool_tokens`, `free_gib`, `contexts_resident`, and `stages` for PP prefill), `hicache_host_gib_all_prefill`, `note` |

Example (abridged, `dsv41.json`):

```json
{
  "launch_argv": ["--model", "/mnt/c8s-data/models/deepseek-v41-flash", "--port", "8040", "..."],
  "layout": "dsv41",
  "paths": {"install_root": "/opt/sglang", "kernel_cache_dir": "/tmp/dsv41/kernel-cache", "model": "/mnt/c8s-data/models/deepseek-v41-flash", "python": "/opt/dsv41/bin/python", "...": "..."},
  "plan": {
    "physical_gpus": [0, 1, 2, 3],
    "router": {"command": ["/opt/dsv41/bin/python", "-m", "sglang_router.launch_router", "--pd-disaggregation", "--prefill", "http://127.0.0.1:8041", "8999", "--decode", "http://127.0.0.1:8042", "..."], "env": {"PYTHONPATH": "/opt/sglang/python"}, "name": "router", "url": "http://127.0.0.1:8040"},
    "status": "unqualified",
    "topology": "1x(TP2 prefill + TP2 decode); CPU Engram; DP=1 per worker",
    "workers": [{"name": "prefill", "command": ["/opt/dsv41/bin/python", "-m", "sglang.launch_server", "--model-path", "/mnt/c8s-data/models/deepseek-v41-flash", "..."], "env": {"CUDA_VISIBLE_DEVICES": "0,1", "PYTHONPATH": "/opt/sglang/python", "...": "..."}, "url": "http://127.0.0.1:8041"}, "..."]
  },
  "revision": "<40 hex>",
  "schema": "confidential-sglang/dsv41-layout/v1",
  "source_env": "recipes/deepseek-v41/container/dsv41.env",
  "target": {"gpu_count": 4, "gpu_models": ["B200"], "host_available_gib": 768, "host_gpus": [4, 5, 6, 7], "memory_profile": null}
}
```

The router command runs `sglang_router` from the engine environment, as the
docker recipe does. The plans describe the docker recipe's single-node PD
layouts; they are not chart values.
