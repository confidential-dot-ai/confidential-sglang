# Confidential changes to this fork

This file lists every change made to confidential-sglang for the
confidential-inference images, and why. Newest first. Detailed provenance
(source commits, SHA-256 of each moved file, policy reviews) is in the notes
it links to.

## 2026-11-20: one engine image, one router image, reproducible CI

Goal: confidential-inference deploys images built from this repository only,
reproducibly, with the FlashInfer prebuilt kernels. A staging deploy failed
because the public repo's own engine image had no prebuilt kernels: the first
top-k request JIT-compiled the sampling module for ~67 s and the liveness
probe killed the workers. Daniyal's hand-built images had the kernels but were
tagged `:latest`, not reproducible, and installed the FlashInfer wheels
without hashes. The public repo's `images/sglang-confidential/` and
`images/sglang-router/` will be deleted; everything they guaranteed is here.

### Merged Daniyal's stack onto `main`

- PRs #1 (`dsv41-release`), #2 (`dsv41-readiness`), #3 (`recipe-qwen35-0.8b`),
  #5 (`dsv41-pair-progress`), #4 (`recipe-dsv41-flash`), #7 (`cluster-stress`)
  merged bottom-up with merge commits (no conflicts); `main` was
  `314bce7cf` (upstream), now holds the whole stack. #6 had already merged into
  `dsv41-release`. GitHub's stacked PRs refused a base change, so the base
  branches (`dsv41-release`, `dsv41-readiness`, `dsv41-pair-progress`) were
  fast-forwarded to the merged `main`, which closed each PR as merged. No
  force-push.
- Why: `main` is the source of the images; one branch, one image per commit.

### Removed upstream's GitHub workflows

- All 122 files of upstream `.github/workflows/` are deleted; only `images.yml`
  is ours. Why: with Actions enabled on the fork they would run upstream's CI
  (nightly GPU jobs, bots, PR bots) here. The upstream sync never takes them
  back (below).

### Upstream sync (confidential-sglang-sync)

- `sync.sh` no longer replays `main..dsv41-release` onto upstream (our `main`
  is not an upstream commit any more, so the old script stops at its first
  check and pushes nothing). It now fast-forwards the `upstream` branch to
  upstream `main` (a plain mirror), and writes `upstream-nightly` = the merge
  of the newest cleanly merging upstream commit into our `main`
  (`upstream-nightly-staging` = that upstream commit). It never writes
  `main`. Our `.github/workflows` are kept as they are in the merge, so
  upstream workflow changes never conflict. A real conflict fails the run and
  names the commit and paths. Dry run on 2026-11-20: upstream head
  `741d14cec`, clean up to `df4ec2e07`, blocked by `e11c9583e` in
  `python/sglang/srt/speculative/dspark_components/dspark_config.py`.
- The workstation timer runs `run-local.sh` from its own checkout of
  confidential-sglang-sync: pull it to pick up the new `sync.sh`.

### Engine image (`confidential/engine/`)

`ghcr.io/confidential-dot-ai/confidential-sglang/engine`, one per `main`
commit, tagged `sha-<commit>`. See `confidential/engine/README.md`.

- Folds in Daniyal's `dsv41-pd` environment
  (`recipes/deepseek-v41/container/Dockerfile`, now deleted with its `build.sh`
  and unhashed `requirements.lock`). Same CUDA 13.0.3 devel base, now by
  digest; same apt packages, now from `snapshot.ubuntu.com` `20261104T000000Z`;
  same Python versions, now `requirements.lock` with the SHA-256 of every
  PyPI file (`requirements.in` + `lock.sh`). Added only the build tools the
  editable install needs (`setuptools-scm`, `vcs-versioning`, `wheel`).
- FlashInfer 0.7.3 prebuilt kernels by URL + SHA-256 (`flashinfer-cubin`,
  `flashinfer-jit-cache`, `-sm100a`, `-sm103a`; GitHub release assets, hashes
  from flashinfer.ai's index). The build asserts both providers have
  `jit_cache/sampling/sampling.so`. Why: the 67 s JIT compile above.
- The PyPI `sglang-router` 0.3.2 is no longer in the environment: the image
  builds this checkout's router wheel (the router image's `router-wheel`
  stage, pasted verbatim) because the DeepSeek-V4.1 launcher runs the router
  inside the pod.
- This checkout's SGLang (`python/`), editable at `/opt/sglang/python`,
  hardened at build time with `harden_http.py --variant pd`, `harden_media.py`
  and `harden_dist.py` (`confidential/helpers/CHANGES.md` has the reviews).
- Both recipes, whole (`recipes/deepseek-v41` with its loadtest, so the stress
  suite runs from the image; `recipes/qwen35-0.8b`).
- Helpers: `wait-for-model`, `fetch-model`, `sglang-pair` (+
  `pair_diagnostics.py`, `private_log_sink.py`), `dcgm-exporter` with its
  counters; the CPU simulator at `/opt/confidential-inference/sglang-simulator`
  (the public chart's simulator mode).
- Launch plans of every DeepSeek-V4.1 layout at
  `/opt/sglang/layouts/<layout>.json` (below).
- uid 1001, `HOME=/tmp`, `/opt` not writable by the runtime user, no
  `ENTRYPOINT`/`CMD` (the CUDA base's entrypoint is cleared): the public chart
  runs `/usr/local/bin/wait-for-model ... -- python3 -m sglang.launch_server ...`.
- No bytecode is compiled into the image (Daniyal's chart image ran
  `compileall`). Why: a `.pyc` records its source mtime and marshal output is
  not guaranteed stable, which would break the two-build digest match. Cost:
  SGLang is compiled in memory at each process start.
- Replaced Daniyal's hand-built `recipes/deepseek-v41/chart/` and
  `recipes/qwen35-0.8b/container/` images (Dockerfiles, `build.sh`, vendored
  helper copies); their READMEs point at the engine image.
- The DeepSeek-V4.1 docker recipe (`compose.yaml`, `*.env`) now runs the
  engine image by digest (`DSV41_IMAGE` must be set), with the recipe's
  entrypoint and root user set in compose.

### Helpers moved from confidential-inference (`confidential/helpers/`)

- `wait_for_model.py` (public `main` `0d27717`: structured log events #315,
  refused logging settings #316, `jit_compile` watcher #334), `fetch_model.py`,
  `dcgm-counters.csv` unchanged; `harden_http.py`, `harden_media.py`,
  `harden_dist.py` re-pinned/ported to this fork's source; their tests in
  `confidential/tests/helpers/` (183 pass). They live only here now. Details,
  hashes and the policy reviews: `confidential/helpers/CHANGES.md`.

### Launch plans (`confidential/layouts/`)

- `build_layouts.py` runs `launch.py --dry-run` for each
  `recipes/deepseek-v41/container/*.env` (`dsv41`, `dsv41-262k`,
  `dsv41-b200-1m`, `dsv41-b300-1m`, `dsv41-b300-1m-ced`) with container paths
  (model `/mnt/c8s-data/models/deepseek-v41-flash`, `/opt/dsv41/bin/python`,
  `/opt/sglang`), deterministic output. Schema in
  `confidential/layouts/README.md`. `launch.py` gained `--install-root` and
  `--source-revision` (defaults unchanged). Details:
  `confidential/layouts/CHANGES.md`.

### Stress suite profiles

- `cluster_stress.py --profile qwen-short` (1-GPU Qwen3.5-0.8B staging:
  functional, needles to 240K tokens within Qwen's 262144 context, overload,
  5-minute soak) and `--profile dsv41-full` (the full DeepSeek suite). How to
  run from the engine image: `recipes/deepseek-v41/loadtest/CLUSTER-STRESS.md`.

### Router (`sgl-model-gateway/`, `confidential/router/`)

`ghcr.io/confidential-dot-ai/confidential-sglang/router`: this checkout's
`sgl-model-gateway` as a maturin wheel in a slim Python image (no CUDA),
`python3 -m sglang_router.launch_router` and `/usr/local/bin/ci-sglang-router`
(the public chart's private launcher). Details: `confidential/router/CHANGES.md`.

- Ported from the public patches: `http-no-tokenizer`, `client-errors` (a
  worker 4xx is not a worker fault, with tests), `private-errors` (fixed event
  names, no error text or URLs in logs). `worker-withdrawal`,
  `admission-caps`, `pd-router` were already in the stack.
- Replaced `pd-pairs` (fixed `CI_ROUTER_PD_PAIRS`, exactly two pairs) with
  discovery-based pairing: in PD mode with `--service-discovery`, a pod's
  prefill and decode ports form one pair and the router never crosses pairs.
  Regular `--service-discovery --selector` is unchanged.
- `sgl-model-gateway/Cargo.lock` is committed; builds use `--locked`.

### CI (`.github/workflows/images.yml`, `confidential/ci/`)

- Builds each image twice, clean, on two fresh runners (pinned BuildKit,
  `SOURCE_DATE_EPOCH` = commit time, `rewrite-timestamp=true`, no provenance
  or SBOM), compares the digests, runs `check-image.sh` on build #1, runs the
  CPU tests and the router's `cargo test --lib` / `--test reliability_tests`,
  then pushes build #1's bytes by digest, tagged `sha-<commit>`, and writes
  the release record `images-<12 hex>.json` to the GitHub release
  `images-<12 hex>` (`confidential/ci/release_record.py`, schema in its
  docstring). Modeled on confidential-inference's `release-images.yml`.
- Each build also uploads a per-file listing of its image
  (`confidential/ci/oci_listing.py`: path, mode, owner, mtime, SHA-256), so a
  digest mismatch can be diffed. The first engine builds differed only in
  uv's `uv_cache.json` (an install-time ctime) in the router wheel's
  dist-info; it is removed like the SGLang one. With that, the two clean
  engine builds match (run 44708494033 of confidential-sglang-ci).
- Build #1 of each image also runs `check-image.sh`, on its own runner, after
  removing the builder's state to free disk. A separate check job spent ~22 of
  its 24 min downloading and loading the ~20 GB engine image.
- Triggers: push to `main` and `workflow_dispatch`. The workflow runs only in
  this repository. Until 2026-11-20, Actions were disabled here, and the
  workflow ran from `confidential-dot-ai/confidential-sglang-ci`, which is
  archived after the first release made here. That repo made the first record,
  `images-d0169679925d`. Run 44785601702 here rebuilt the same commit without
  publishing: engine `sha256:8fa664ce…4f1d` and router `sha256:609ab202…f3f6`,
  both builds equal to that record.
- Actions settings: only GitHub's actions, `confidential-dot-ai/*`, and the
  three Docker actions this workflow uses (each allowed at one commit hash);
  hash pinning required; a read-only default token; approval required for
  outside contributors. The upstream workflows that remain on other branches
  (for example `upstream-nightly`) use actions by tag, so they fail at
  startup and run no step.

## Cold-path warmup (2026-11-20)

- `python/sglang/srt/entrypoints/warmup.py`: a custom warmup,
  `confidential_cold_path`, for `--warmups`. It runs upstream's
  `prefill_shapes` (prefill lengths up to 32K), then every sampling path
  (greedy, defaults, top-k, top-p/min-p, penalties, stop strings, JSON
  schema; with and without logprobs), alone and mixed in one batch, then
  decode batch sizes 1 to 64. Custom warmups run in the server lifespan,
  before it listens, so the probes and the router see the worker only
  afterwards.
- Why: staging `staging-39086210fdc6` (Qwen3.5-0.8B, B200) logged 20
  `jit_compile` events after the worker was ready (Triton, SGLang JIT,
  inductor). Each burst stalled the worker; the router's `/health_generate`
  check failed, and clients got 503.
- `confidential/engine/check-image.sh` checks that the warmup is registered.

## Stress suite: the model's reasoning switch (2026-11-21)

- `recipes/deepseek-v41/loadtest/endpoint.py`: `--thinking-kwarg` (the
  `chat_template_kwargs` key, default `thinking`) and `--thinking-default`
  (`on`, `off`, or empty for the model's default). `func_tests.py` and
  `needles.py` apply them to every chat request; `cluster_stress.py` passes
  them to every phase.
- Why: Qwen3.5 thinks by default under `enable_thinking`. On
  `staging-d0d1b05e01e8` 50 of 72 func cases and every needle got empty
  content: the reasoning used the token budget, and `{"thinking": true}` did
  nothing. A Qwen profile passes `--thinking-kwarg enable_thinking
  --thinking-default off`.

## zstd layers: the containerd image store on the build runner (2026-11-21)

- #18 compresses image layers with zstd and loads build #1 with
  `docker load -i` for its check. The runner's classic Docker store cannot
  load an OCI archive with zstd layers (run 44874491789: `open
  .../blobs/json: no such file or directory`), and skopeo refuses zstd for
  `docker-daemon:`.
- `confidential/ci/free-disk.sh` turns on Docker's containerd image store
  (`features.containerd-snapshotter`), which loads it, and bind-mounts
  containerd's root on /mnt, where the image store now lives.

## Images per input commit, not per main commit (2026-11-21)

- `confidential/ci/image-inputs.txt` lists every path that can change the
  bytes of either image: both Dockerfiles and their `.dockerignore`, every
  `COPY` and bind-mount source, and `images.yml` itself.
  `confidential/tests/ci/test_image_inputs.py` fails if a Dockerfile reads a
  path the list does not cover.
- The `source` job computes the input commit
  (`confidential/ci/image_inputs.py`: the newest first-parent commit that
  changed an input). The images are built from it, labelled, tagged
  `sha-<input commit>` and recorded as `images-<12 hex of the input commit>`;
  `SOURCE_DATE_EPOCH` is its commit time. The record format is unchanged.
  The CI scripts (`free-disk.sh`, `check-image.sh`, `oci_listing.py`,
  `release_record.py`) and the tests come from the pushed commit.
- If that release exists (and its record checks for the input commit), the
  run builds and publishes nothing; the summary names the record. A
  `workflow_dispatch` with `publish=false` always builds both images twice,
  and never publishes.
- Why: on 2026-11-21, five full engine builds (~60 min each) ran for four
  small fixes, two of which changed only CI scripts.

## The CUDA graph input copy compiles once per dtype layout (2026-11-21)

- `python/sglang/kernels/ops/memory/small_copy.py`: `_small_copy_kernel` no
  longer specializes on the alignment of its source and destination
  pointers. With the kernel names in `jit_compile` (#22),
  `staging-6569fd307829` showed that almost every compile after ready was
  this kernel: CUDA graph input slices start at batch-dependent offsets, and
  each alignment pattern was another variant. The copies are at most 8,192
  elements, so the generic code costs nothing measurable.
- `recipes/deepseek-v41/loadtest/soak.py`: `--max-bad` allows answers judged
  bad (no tool call, malformed) apart from failed requests (`--max-failures`).
  A small model can miss a tool call at temperature 1.0.

## Stress suite: the load phase takes the reasoning switch (2026-11-22)

- `recipes/deepseek-v41/loadtest/oracle_replay.py` takes its endpoint options
  from `endpoint.add_endpoint_args` (model default still `dsv41`) and applies
  `endpoint.with_thinking` to every request. `oracle_scorecard.py` builds on
  that parser, now a function of its own (`parser()`).
- Why: since `--thinking-kwarg`/`--thinking-default` (2026-11-21),
  `cluster_stress.py` passes both to every phase, but the scorecard declared
  its own endpoint options and exited 2 on them. Every run with the load
  phase failed there (seen on a 4x B200 run of the `dsv41-full` profile).
- `test_cluster_stress.py` parses the load phase's command line with the
  scorecard's parser.

## Stress suite: declared parameters, capacity sheds, metrics and the suite commit (2026-11-23)

- `func_tests.py`: `t_logprobs_rejected` is replaced by `t_logprobs`,
  `t_logprobs_stream` and `t_min_p`, which check the behavior declared with
  the new `--unsupported-params`: a listed parameter must get a clean client
  error (4xx other than 408/429 with an error body, or a 200 stream holding
  only a 4xx error event; never a 5xx, a dropped connection or a timeout) and
  the server must answer the next request; an unlisted one must work.
  `cluster_stress.py` passes `--unsupported-params` to func too. Why: staging
  listed `t_logprobs_rejected` as a known failure because Qwen serves
  logprobs, so the case could not catch a 5xx under DSpark either.
- `overload.py`: `--shed-codes` names the 429 error codes of a capacity shed;
  `--require-shed` then counts only those. Each step records `codes_429`.
  Why: a 429 of a per-address cap or of the key's limit is not a capacity shed.
- `cluster_stress.py`: with `--prom-url` (or direct `--metrics-*` URLs) each
  phase result holds the server's metrics window over that phase
  (`live_metrics.window`), read after the run at the phase boundaries
  (`prom_metrics.PromScraper.snapshot_at`); `--require-metrics` fails the run
  if a window could not be read. `report.json` records `suite.commit`.
- `prom_metrics.py` names each series exactly and joins them with `or`, each
  tagged with `label_replace`. Why: Azure Monitor managed Prometheus refuses a
  regex on `__name__`, so the scorecard's `--prom-url` never worked against
  it, and `or` drops series whose labels are equal apart from the name.
- The `qwen-short` profile drops its known failure; `dsv41-full` declares
  `logprobs,min_p` unsupported.

## Cold-path warmup under DSpark; dsv41-262k runs it on both workers (2026-11-23)

- `python/sglang/srt/entrypoints/warmup.py`: `cold_path_plan()` gives the
  cold-path sampling paths and whether to ask for logprobs, minus what the
  scheduler refuses under the server's speculative algorithm. DSpark refuses
  `min_p` (scheduler.py, any speculative algorithm) and `return_logprob`
  (`validate_dflash_request`); a refused warmup request aborts, the warmup
  raises, and the server never listens. Plain DFlash also drops grammars, UNO
  also drops penalties. A path that loses a parameter still runs the others
  (`top_p` without `min_p`). Without speculation nothing changes.
- `recipes/deepseek-v41/launcher/launch.py`: `--warmups NAMES` passes SGLang
  custom warmups to every worker. `dsv41-262k.env` sets
  `--warmups confidential_cold_path`, so the baked `dsv41-262k.json` plan has
  it on prefill and decode, and `sglang-pair` (which waits for each worker's
  `/health`) reports the pair ready only once both are warm.
- Tests: `test/registered/unit/entrypoints/test_cold_path_warmup.py`,
  `test_launch.py::test_warmups_are_off_by_default_and_reach_both_roles`.

## Engram holder without mlock (2026-11-23)

- `recipes/deepseek-v41/launcher/hold_engram_cache.py`: `--lock auto`
  (default) tries `mlock` and, when it is refused (`EPERM`/`ENOMEM`/`EAGAIN`:
  no `RLIMIT_MEMLOCK` headroom, no `CAP_IPC_LOCK`), releases any partial locks
  and holds the shards unlocked: read-only mappings faulted in with
  `MADV_POPULATE_READ`, populated again every `--refresh-seconds` (300) if
  the kernel reclaimed pages. It logs `mode=locked` or `mode=populated` and
  the resident bytes (`mincore`). `--lock required` keeps the old behavior;
  `--lock off` never locks. `--reserve-gib` (default 32) is the RAM headroom
  check that was hard-coded.
- Why: under C8s the worker's containerd sets no `LimitMEMLOCK`, and the
  chart runs the holder as uid 1001 with no capabilities, so the ~198 GiB
  `mlock` fails and the pair never starts.
- `recipes/deepseek-v41/container/entrypoint.sh` passes `--lock required`:
  the docker recipe's launcher credits only locked pages.
- Tests: `recipes/deepseek-v41/launcher/test_hold_engram_cache.py`.

## The PD decode warmup skips grammars; warmups run once per server (2026-11-23)

- `python/sglang/srt/entrypoints/warmup.py` `cold_path_plan(spec, mode)`: on a
  PD decode server the cold path leaves out the grammar paths (`json_schema`,
  `regex`, `ebnf`, `structural_tag`). A decode warmup request takes the fake
  bootstrap, so no prefill samples its first token under the grammar; the
  grammar refused it ("Grammar accept_token failed ... Tokens not accepted",
  HTTP 500), the warmup raised and decode exited (4x B200, `dsv41-262k`, run
  R1, 10:21 UTC). Prefill still warms the grammar paths.
- `execute_warmups`: with `--tokenizer-worker-num N` every tokenizer process
  runs the lifespan, so each ran the whole warmup (N=4 on `dsv41-262k`). The
  processes of one server (main process ID and start time) now elect one
  with `flock` on a file in `/dev/shm`; it runs the warmups and writes a done
  marker, and the others wait on the lock (none answers `/health` before the
  warmup ends), then skip. `http_server.py` is unchanged, so
  `harden_http.py`'s pinned hash still matches.
- Tests: `test_cold_path_warmup.py` (decode plan; four forked processes warm
  up once and none finishes before the warmup ends).

## sglang-pair: parallel start (2026-11-23)

- `confidential/helpers/pair_supervisor.py`: optional config key
  `parallelStart` (boolean, default `false`). When true, prefill and decode
  start together once the model gate is ready (prefill is spawned first);
  the pair is `starting` until every rank is ready and fails on any worker
  failure or the startup timeout, as before. The chart's pair template sets
  it for `dsv41-262k`. Measured times and peak host RAM: see the PR.
- Tests: `test_pair_supervisor.py` (the key is an optional boolean; decode
  runs while prefill still loads, and the pair is ready only once both are).
