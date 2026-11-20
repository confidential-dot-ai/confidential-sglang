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
