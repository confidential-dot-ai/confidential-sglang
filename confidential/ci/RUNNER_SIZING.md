# CI Runner Sizing: Blacksmith CI vs GitHub Larger Runners

With zstd layer compression enabled (#18, #19), full `images.yml` runs on free 4-vCPU `ubuntu-24.04` runners dropped from **67.7 min** (gzip, run `44811154899`) to **38.1 min** (zstd, run `44926647632`), and `publish engine` dropped from **10.8 min** to **4.0 min**.

This document analyzes where the remaining ~38 minutes go and compares **Blacksmith CI** and **GitHub Larger Runners** for cutting build time roughly in half again (~17–22 min total).

## 1. Current Bottlenecks (Free 4-vCPU `ubuntu-24.04`, Run `44926647632`)

| Job | Gzip (`44811154899`) | Zstd (`44926647632`) | Main Remaining Bottleneck |
| --- | --- | --- | --- |
| `build engine #1` (build + listing + upload + load + check) | 55.9 min | **33.3 min** | Rust `router-wheel` compile (~14 min), zstd export (~4.5 min), `free-disk.sh` (~2 min), load/upload (~7 min) |
| `build engine #2` (build + listing) | 47.5 min | **24.0 min** | Rust `router-wheel` compile (~14 min), zstd export (~4.5–6.6 min), `free-disk.sh` (~2 min) |
| `build router #1` / `#2` | 16.9 / 15.2 min | **20.1 / 17.5 min** | Rust `maturin build` (~14 min, >80% of job) |
| `router tests` | 13.7 min | **19.0 min** | Rust `cargo test` compile + run |
| `publish engine` | 10.8 min | **4.0 min** | Network + OCI unpack (9.0 GB zstd archive) |
| `python tests` / `compare` / `publish router` / `release` | ~2.5 min | **~2.5 min** | Already fast |
| **Total wall-clock run** | **67.7 min** | **38.1 min** | Critical path: `build engine #1` (33.3 min) + `publish engine` (4.0 min) |

Because zstd removed the 20-minute single-core gzip export bottleneck, **CPU-bound Rust compilation (`router-wheel`, `router`, `router tests`)** is now the #1 bottleneck and scales cleanly with 16 vCPUs.

## 2. Runner Comparison (16-vCPU with NVMe)

Note: `confidential-sglang` is a public repository, so standard `ubuntu-24.04` (4 vCPU) is **$0**. Paid runners trade a small per-run cost for ~2x faster builds and no `free-disk.sh` workaround.

| Option | `build engine` | `build router` & `router tests` | Full `images.yml` Wall-Clock | Est. Cost / Run | Est. Monthly (~10 runs) | Disk Cleanup (`free-disk.sh`) | Trust Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Free GitHub `ubuntu-24.04` (4 vCPU)** | 24–33 min | 18–20 min | **~38 min** | **$0** | **$0** | Required (~2 min) | GitHub-hosted |
| **Blacksmith (16 vCPU, NVMe)** | **~12–15 min** | **~5–6 min** | **~17–20 min** | **~$0.75–$1.20** (`~$0.016/min`) | **~$10/mo** | **Not needed** (0 min) | 3rd-party ephemeral runner |
| **GitHub Larger Runner (16 vCPU)** | **~14–17 min** | **~6–7 min** | **~19–22 min** | **~$3.00–$4.00** (`$0.064/min`) | **~$35/mo** | **Not needed** (0 min) | GitHub-hosted |

### Scope: Only `confidential-sglang` Needs Larger Runners

- `confidential-sglang` builds both the ~20 GB `engine` image and the Rust-heavy `router` image + `router tests`.
- All remaining images in `confidential-inference` (`gateway`, `maintenance-gateway`, `kube-state-metrics`, `rollout-client`) are under 18 MB and build in under 2.5 minutes on free runners.
- Staging deploys (`Deploy and validate staging`) wait on Confidential Cloud VMs and remote cluster tests, not CI runner CPU.

## 3. Recommended Setup: Fast Path + Async Free GitHub Verification

There is only **one** image format (**zstd**). Both paths build the exact same zstd image; we never build or publish gzip.

```text
push to main
  ├── FAST PATH (~17–20 min; publishes images + release)
  │     ├── build engine #1, #2   (blacksmith-16vcpu-ubuntu-2404)
  │     ├── build router #1, #2   (blacksmith-16vcpu-ubuntu-2404)
  │     ├── router tests          (blacksmith-16vcpu-ubuntu-2404)
  │     ├── python tests          (ubuntu-24.04)
  │     ├── compare engine/router (ubuntu-24.04)
  │     ├── publish engine/router (ubuntu-24.04 — write tokens stay on GitHub)
  │     └── release record        (ubuntu-24.04 — creates images-<short>.json)
  │
  └── ASYNC VERIFICATION (~31 min, separate workflow, NEVER blocks publish)
        ├── verify engine         (free ubuntu-24.04, 1 clean zstd build)
        ├── verify router         (free ubuntu-24.04, 1 clean zstd build)
        └── verify-match          (asserts free GitHub digest == published digest;
                                   fails loud on mismatch, never publishes)
```

### Why This Works Without Weakening Reproducibility

1. **Cross-CPU determinism verified:** Tested pinned `moby/buildkit@sha256:28a89871...` with `rewrite-timestamp=true,compression=zstd` across `GOMAXPROCS=1`, `4`, and `8` — all three produced the exact same manifest SHA-256 bit-for-bit. A free 4-vCPU GitHub runner can therefore independently verify a 16-vCPU Blacksmith build.
2. **Write tokens stay on GitHub:** `build` jobs only have `contents: read`. Keeping `publish` (`packages: write`) and `release` (`contents: write`) on GitHub-hosted `ubuntu-24.04` ensures no push or release token runs on a 3rd-party runner.
3. **2-vendor integrity at $0 extra:** The async free GitHub workflow never slows down dev/staging artifacts, never publishes anything, and fails loudly if Blacksmith's published digest ever disagrees with an independent GitHub-hosted rebuild. Production promotion can optionally require that check to be green.

## 4. Rollout Checklist

1. Choose runner provider:
   - **Blacksmith** (~$1/run, fastest, with async free GitHub verification), or
   - **GitHub 16-vCPU Larger Runner** (~$3.50/run, 100% GitHub-hosted, no 3rd-party app needed).
2. If using Blacksmith, install the GitHub App scoped **only** to `confidential-dot-ai/confidential-sglang`, and keep `images.yml` restricted to `push: [main]` and `workflow_dispatch` (never `pull_request`).
3. Update `runs-on:` for `build` and `router-tests` in `.github/workflows/images.yml` (and skip `free-disk.sh` on 16-vCPU runners).
4. Add `.github/workflows/verify-reproducibility.yml` on free `ubuntu-24.04` to independently rebuild and assert exact digest equality against `images-<short>.json`.
