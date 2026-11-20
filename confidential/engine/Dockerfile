# The confidential-sglang engine image: one image per main commit, for every
# model and layout (Qwen3.5-0.8B staging on one GPU, DeepSeek-V4.1-Flash PD
# pairs on B200/B300).
#
# Build context: the repository root (Dockerfile.dockerignore keeps only
# python/, recipes/, sgl-model-gateway/ and confidential/). Build it with the
# workflow .github/workflows/images.yml, or locally:
#   docker buildx build -f confidential/engine/Dockerfile \
#     --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
#     --build-arg SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD) \
#     --output type=oci,dest=engine.tar,rewrite-timestamp=true --provenance=false .
#
# Every input is pinned: base images by linux/amd64 digest, Ubuntu packages by
# one snapshot.ubuntu.com timestamp, Python wheels by SHA-256
# (requirements.lock, FlashInfer's prebuilt kernels by URL + SHA-256), Rust
# crates by Cargo.lock (--locked). SGLang is this checkout's python/.
#
# What the image holds (confidential/engine/README.md):
#   /opt/dsv41                     the Python environment; python3 on PATH
#   /opt/sglang/python             this checkout's SGLang (editable install),
#                                  HTTP server hardened (harden_http.py --variant pd)
#   /opt/sglang/recipes            recipes/qwen35-0.8b and recipes/deepseek-v41
#                                  (launcher, loadtest, docker recipe)
#   /opt/sglang/layouts/*.json     launch plans of the DeepSeek-V4.1 layouts
#   /opt/sglang/SOURCE_RECORD.json the source commit
#   /usr/local/bin/wait-for-model, fetch-model, sglang-pair, dcgm-exporter
#   /opt/confidential-inference/sglang-simulator   the CPU simulator (staging)
# uid 1001, HOME=/tmp, nothing writable under /opt, no ENTRYPOINT and no CMD:
# the chart (or `docker run --entrypoint`) names every command.

FROM ghcr.io/astral-sh/uv@sha256:265d074d08ed8080bc578087ca68a8e94611f9c7be671d40e18b3d3b1ad0dad4 AS uv
FROM nvcr.io/nvidia/k8s/dcgm-exporter@sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003 AS dcgm-exporter

# The router wheel of this checkout's sgl-model-gateway: the same stage as
# confidential/router/Dockerfile (keep the two identical).
FROM docker.io/library/rust@sha256:3914072ca0c3b8aad871db9169a651ccfce30cf58303e5d6f2db16d1d8a7e58f AS router-wheel
ARG DEBIAN_SNAPSHOT=20261104T000000Z
ARG SOURCE_DATE_EPOCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARGO_NET_RETRY=10
# Rust 1.90.0 on Debian 12, as sgl-model-gateway/rust-toolchain.toml asks.
RUN test -n "${SOURCE_DATE_EPOCH}" \
    && printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' \
        "${DEBIAN_SNAPSHOT}" "${DEBIAN_SNAPSHOT}" > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/* \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv clang protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/build \
    && printf 'maturin==1.9.6 --hash=sha256:%s\n' \
        0246202377c49449315305209f45c8ecef6e2d6bd27a04b5b6f1ab3e4ea47238 > /tmp/maturin.lock \
    && /opt/build/bin/pip install --require-hashes --only-binary=:all: -r /tmp/maturin.lock
WORKDIR /src
COPY sgl-model-gateway /src/sgl-model-gateway
COPY confidential/router/canonicalize_wheel.py /inputs/canonicalize_wheel.py
ENV RUSTFLAGS="--remap-path-prefix=/src=/usr/src/sglang --remap-path-prefix=/usr/local/cargo/registry=/usr/src/cargo-registry"
RUN test -f sgl-model-gateway/Cargo.lock \
    && /opt/build/bin/maturin build --locked --release --features vendored-openssl \
        --manifest-path sgl-model-gateway/bindings/python/Cargo.toml --out /out \
    && /opt/build/bin/python3 /inputs/canonicalize_wheel.py /out/*.whl \
    && ls /out/sglang_router-*.whl


# CUDA 13.0.3 devel (nvidia/cuda:13.0.3-devel-ubuntu24.04, linux/amd64): the
# toolkit Daniyal's dsv41-pd image was tested with. FlashInfer, Triton and the
# SGLang JIT kernels compile against it at run time.
FROM docker.io/nvidia/cuda@sha256:7d56ebe2b7cd864a60dca3c8b2d0a39f8fc110417e8253e32505c3387f59119c AS base
ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu packages from one snapshot.ubuntu.com timestamp, never the moving
# archive; the base image's NVIDIA apt source is not read. Mooncake links
# ibverbs/mlx5, libnl, numa and curl even for intra-node NVLink; FFmpeg 6.1
# keeps torchcodec's import quiet; the OpenSSL headers serve the HiCache hash
# extension's JIT build; git serves setuptools-scm.
RUN printf 'deb https://snapshot.ubuntu.com/ubuntu/20261104T000000Z %s main universe\n' \
        noble noble-updates noble-security > /tmp/snapshot.list \
    && apt-get -o Dir::Etc::SourceList=/tmp/snapshot.list -o Dir::Etc::SourceParts=/dev/null \
        -o Acquire::Check-Valid-Until=false update \
    && apt-get -o Dir::Etc::SourceList=/tmp/snapshot.list -o Dir::Etc::SourceParts=/dev/null \
        upgrade -y --no-install-recommends \
    && apt-get -o Dir::Etc::SourceList=/tmp/snapshot.list -o Dir::Etc::SourceParts=/dev/null \
        install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential ca-certificates curl git procps libssl-dev \
        ibverbs-providers libibverbs1 libnuma1 libcurl4t64 libnl-3-200 libnl-route-3-200 \
        libavcodec60 libavdevice60 libavfilter9 libavformat60 libavutil58 libswresample4 libswscale7 \
    && rm -rf /tmp/snapshot.list /var/lib/apt/lists/* /var/cache/apt/*.bin /var/cache/apt/archives/*.deb \
        /var/log/apt /var/log/dpkg.log /var/log/alternatives.log /var/cache/ldconfig/aux-cache \
        /var/cache/fontconfig

COPY --from=uv /uv /usr/local/bin/uv
# No bytecode from installs: a .pyc records its source's mtime, and the runtime
# user cannot write the root-owned environment.
ENV VIRTUAL_ENV=/opt/dsv41 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV PATH=/opt/dsv41/bin:${PATH}
# The whole environment, by hash, without resolution. Epoch mtimes: FlashInfer's
# JIT cache compares source and object mtimes, so a fresh install would
# otherwise rebuild every op. flashinfer-python and torch-c-dlpack-ext both ship
# a top-level build_backend.py (their PEP 517 build shim, never imported at run
# time); uv installs wheels in parallel, so which copy survives varies between
# builds: the file and both RECORD lines go.
# FlashInfer's prebuilt kernels: without them FlashInfer JIT-compiles modules on
# first use inside the scheduler (the top-k/top-p sampling module took ~65-86 s
# on B200, enough for a liveness probe or a 90 s watchdog to kill the worker).
# The build fails unless both providers ship sampling.so.
RUN --mount=type=bind,source=confidential/engine/requirements.lock,target=/inputs/requirements.lock,readonly \
    uv venv --python /usr/bin/python3.12 /opt/dsv41 \
    && uv pip install --no-deps --require-hashes -r /inputs/requirements.lock \
    && site=/opt/dsv41/lib/python3.12/site-packages \
    && rm "$site/build_backend.py" \
    && sed -i '/^build_backend\.py,/d' "$site"/flashinfer_python-*.dist-info/RECORD "$site"/torch_c_dlpack_ext-*.dist-info/RECORD \
    && ! grep -l '^build_backend\.py,' "$site"/*.dist-info/RECORD \
    && python -I -B -c "import flashinfer, flashinfer_jit_cache as j, flashinfer_cubin as c, pathlib; \
assert flashinfer.__version__ == '0.7.3', flashinfer.__version__; \
root = pathlib.Path(j.__file__).parent / 'providers'; \
missing = [a for a in ('sm100a', 'sm103a') if not (root / a / 'jit_cache/sampling/sampling.so').is_file()]; \
assert not missing, f'jit-cache providers missing sampling.so: {missing}'; \
print('flashinfer', flashinfer.__version__, 'jit-cache', j.__version__, 'cubin', c.__version__)" \
    && rm -rf /root/.cache /root/.triton /tmp/* \
    && find /opt/dsv41 -xdev -exec touch -h -d @0 {} +
# Mooncake's transfer engine links the CUDA 12 runtime from its wheel.
ENV LD_LIBRARY_PATH=/opt/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}

FROM base
ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
# The router this checkout builds (sglang_router): the DeepSeek-V4.1 launcher
# runs it inside the pod for a PD pair, and its preflight imports it.
RUN --mount=type=bind,from=router-wheel,source=/out,target=/run-router,readonly \
    uv pip install --no-deps /run-router/sglang_router-*.whl \
    && python -I -B -c 'import sglang_router.sglang_router_rs'

# DCGM exporter with its embedded host engine, as confidential-inference's
# images installed it.
RUN --mount=type=bind,from=dcgm-exporter,source=/,target=/run-dcgm,readonly \
    --mount=type=bind,source=confidential/helpers/dcgm-counters.csv,target=/run-src/dcgm-counters.csv,readonly \
    install -D -m 0555 /run-dcgm/usr/bin/dcgm-exporter /usr/local/bin/dcgm-exporter \
    && install -d -m 0755 /opt/dcgm/lib \
    && cp -a /run-dcgm/usr/lib/x86_64-linux-gnu/libdcgm* /run-dcgm/usr/lib/x86_64-linux-gnu/libnvperf_* /opt/dcgm/lib/ \
    && cp -a /run-dcgm/licenses /opt/dcgm/licenses \
    && install -D -m 0644 /run-src/dcgm-counters.csv /etc/dcgm-exporter/counters.csv

# This checkout's SGLang, editable at /opt/sglang/python (workers and the
# launcher import it from there). No .git reaches the build, so the version is
# fixed. uv_cache.json holds the build-time ctime of pyproject.toml (uv's
# reinstall check, never used in the image): it leaves the dist-info and its
# RECORD line.
WORKDIR /opt/sglang
COPY python /opt/sglang/python
RUN --mount=type=bind,source=confidential/helpers,target=/run-src/helpers,readonly \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 SGLANG_BUILD_RUST_EXTS=none \
        uv pip install --no-deps --no-build-isolation -e /opt/sglang/python \
    && dist=/opt/dsv41/lib/python3.12/site-packages/sglang-0.0.0.dev0.dist-info \
    && rm -f "$dist/uv_cache.json" \
    && sed -i '/^sglang-0\.0\.0\.dev0\.dist-info\/uv_cache\.json,/d' "$dist/RECORD" \
    && python -I -B /run-src/helpers/harden_http.py --root /opt/sglang/python/sglang --variant pd \
    && python -I -B /run-src/helpers/harden_media.py --root /opt/sglang/python/sglang \
    && python -I -B /run-src/helpers/harden_dist.py --root /opt/sglang/python/sglang \
    && HOME=/tmp/import-check python -B -c "import sglang.launch_server, sglang.srt.entrypoints.http_server, sglang.srt.models.deepseek_v4, sglang.srt.models.qwen3_5, mooncake" \
    && rm -rf /tmp/* /root/.cache /root/.triton \
    && find /opt/sglang/python /opt/dsv41 -name __pycache__ -prune -exec rm -rf {} + \
    && chmod -R go-w /opt/sglang/python

# Both recipes, whole: the DeepSeek-V4.1 launcher, probes, docker recipe and
# load tests (the stress suite runs from this image), and the Qwen3.5-0.8B
# staging recipe.
COPY recipes/deepseek-v41 /opt/sglang/recipes/deepseek-v41
COPY recipes/qwen35-0.8b /opt/sglang/recipes/qwen35-0.8b

# The chart's helpers, at the paths the chart runs them by
# (confidential/helpers/CHANGES.md records where each came from).
RUN --mount=type=bind,source=confidential/helpers,target=/run-src,readonly \
    install -D -m 0555 /run-src/wait_for_model.py /usr/local/bin/wait-for-model \
    && install -D -m 0555 /run-src/fetch_model.py /usr/local/bin/fetch-model \
    && install -D -m 0555 /run-src/pair_supervisor.py /usr/local/bin/sglang-pair \
    && install -D -m 0444 /run-src/pair_diagnostics.py /usr/local/bin/pair_diagnostics.py \
    && install -D -m 0444 /run-src/private_log_sink.py /usr/local/bin/private_log_sink.py \
    && python -I -B /usr/local/bin/pair_diagnostics.py

# CPU staging (inference.mode: simulator) uses the same image. The simulator is
# outside the production import path; only the staging PYTHONPATH enables it.
RUN --mount=type=bind,source=confidential/simulator,target=/run-src/simulator,readonly \
    install -d -m 0755 /opt/confidential-inference/sglang-simulator \
    && cp -a /run-src/simulator/upstream/. /opt/confidential-inference/sglang-simulator/ \
    && cd /opt/confidential-inference/sglang-simulator \
    && for p in replay-only parser-flags-noop prefix-length fork-imports; do \
         git apply --check "/run-src/simulator/patches/sglang-simulator-$p.patch" \
         && git apply "/run-src/simulator/patches/sglang-simulator-$p.patch" || exit 1; \
       done \
    && chmod -R go-w /opt/confidential-inference

# The launch plan of every DeepSeek-V4.1 layout, with container paths
# (confidential/layouts/README.md). The planner is the launcher's --dry-run:
# standard library only, no GPU.
RUN --mount=type=bind,source=confidential/layouts,target=/run-src/confidential/layouts,readonly \
    --mount=type=bind,source=recipes/deepseek-v41,target=/run-src/recipes/deepseek-v41,readonly \
    python -I -B /run-src/confidential/layouts/build_layouts.py --out /opt/sglang/layouts \
    && python -I -B -c 'import json, pathlib; ps = sorted(pathlib.Path("/opt/sglang/layouts").glob("*.json")); assert ps; [json.loads(p.read_text()) for p in ps]; print(*[p.name for p in ps])'

# The source record launch.py reads (it has no .git), and the build's labels.
RUN test -n "${SOURCE_REVISION}" \
    && printf '{"diff_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","image":"ghcr.io/confidential-dot-ai/confidential-sglang/engine","revision":"%s","status":""}\n' \
        "${SOURCE_REVISION}" > /opt/sglang/SOURCE_RECORD.json \
    && python -I -B -c 'import json; json.load(open("/opt/sglang/SOURCE_RECORD.json"))' \
    && chmod -R go-w /opt/sglang /opt/dsv41 /opt/dcgm \
    && test -z "$(find /opt/sglang /opt/dsv41 /opt/dcgm /opt/confidential-inference -xdev -perm /022 ! -type l -print -quit)"

LABEL org.opencontainers.image.title="confidential-sglang engine"
LABEL org.opencontainers.image.source="https://github.com/confidential-dot-ai/confidential-sglang"
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
LABEL org.opencontainers.image.version="sha-${SOURCE_REVISION}"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda"
LABEL org.opencontainers.image.base.digest="sha256:7d56ebe2b7cd864a60dca3c8b2d0a39f8fc110417e8253e32505c3387f59119c"
LABEL ai.confidential.build.source-date-epoch="${SOURCE_DATE_EPOCH}"
LABEL ai.confidential.ubuntu.snapshot="20261104T000000Z"
ENV CI_DIAGNOSTIC_SOURCE_REVISION=${SOURCE_REVISION} \
    HOME=/tmp
USER 1001:1001
# The CUDA base image's entrypoint script does not apply: no ENTRYPOINT, no CMD.
ENTRYPOINT []
CMD []
