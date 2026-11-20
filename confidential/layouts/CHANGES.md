# Changes: launch plans and stress profiles (branch sa/layouts)

Notes for docs/confidential-changes.md.

## Baked DeepSeek-V4.1 launch plans

- `confidential/layouts/build_layouts.py` (new). Writes one
  `<layout>.json` per `recipes/deepseek-v41/container/*.env` profile by running
  `launch.py --dry-run` with the docker recipe's arguments (entrypoint.sh `pd`
  -> tp2-pd.sh -> _common.sh) and container paths. Why: the engine image must
  carry each layout's exact worker and router command lines, reviewable without
  a GPU, and the recipe's plans held host paths (`/data/...`) and host GPU
  indices. Standard library only, no SGLang import, no host reads (GPU,
  `/proc/meminfo`); byte-identical for one revision, so it does not break the
  image's reproducible build. Build: `python3 -I -B
  confidential/layouts/build_layouts.py --out /opt/sglang/layouts` (revision
  from `--revision`, `$SOURCE_REVISION`, `SOURCE_RECORD.json` or git).
- Container conventions chosen: install root `/opt/sglang`, Python
  `/opt/dsv41/bin/python`, model `/mnt/c8s-data/models/deepseek-v41-flash`
  (the chart's `inference.model.volumeDirectory` with volume name
  `deepseek-v41-flash`; substitute `paths.model` otherwise), kernel caches
  `/tmp/dsv41/kernel-cache`, launcher output `/tmp/dsv41/run`, GPUs `0..N-1`.
  HiCache `auto` (B200 layouts) is planned for 768 GiB of host RAM (8x B200 with
  1 TiB less the pinned Engram copy), recorded in `target.host_available_gib`,
  instead of reading the build machine's memory.
- `confidential/layouts/README.md` (new): layouts, argument order, container
  paths, and the JSON schema `confidential-sglang/dsv41-layout/v1`.
- `recipes/deepseek-v41/launcher/launch.py`: two flags. `--install-root`
  (workers' and router's `PYTHONPATH=<root>/python`; default the launcher's own
  checkout, unchanged) and `--source-revision` (plan revision without
  `SOURCE_RECORD.json` or `.git`, as in an image build). Why: the plan must name
  `/opt/sglang/python` when built from a checkout elsewhere. Default behavior
  is unchanged. Test added in `test_launch.py`.
- `confidential/tests/layouts/test_build_layouts.py` (new): all five layouts
  build, two builds are byte-identical, no `/data` path, schema keys, container
  paths in every command and environment, HiCache-auto host RAM recorded.

## Cluster stress profiles

- `recipes/deepseek-v41/loadtest/cluster_stress.py`: `--profile qwen-short`
  (1-GPU Qwen3.5-0.8B staging: func, needles to 240K within the 262144
  context, overload at 16-128 clients, 5-minute soak; ~15-20 minutes) and
  `--profile dsv41-full` (DeepSeek-V4.1 PD: all phases, 131072 context, n=2
  and min_p known to fail, 4-hour soak). Explicit options override a profile;
  without `--profile` the defaults are unchanged. `report.json`/`report.md`
  record the profile. Why: one suite, runnable from the engine image, with a
  short profile for staging and the full one for DeepSeek.
- `CLUSTER-STRESS.md`: profiles table, and how to run from the engine image
  (`docker run --entrypoint python3 ...` and a Kubernetes Job).
- `test_cluster_stress.py`: profile selection and override tests (no server,
  no aiohttp), and a qwen-short suite run against the fake gateway.
- The engine environment (`recipes/deepseek-v41/container/requirements.lock`)
  already holds every loadtest dependency (aiohttp, tokenizers; the
  cluster-stress image's other pins differ only in patch versions). Not in
  the engine image: the realmix dataset (mount it for the load phase) and
  kubectl (fault phase).
