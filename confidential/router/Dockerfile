# The SGLang router (sgl-model-gateway) of this checkout, built into a small
# Python image: no CUDA, no engine. Build context: the repository root
# (Dockerfile.dockerignore keeps only sgl-model-gateway/ and this directory).
#
#   docker buildx build -f confidential/router/Dockerfile \
#     --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
#     --build-arg SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) \
#     --output type=oci,dest=router.tar,rewrite-timestamp=true .
#
# Every input is pinned: both base images by digest, Debian packages by one
# snapshot.debian.org timestamp, Rust crates by sgl-model-gateway/Cargo.lock
# (--locked), maturin by version and hash, Python wheels by SHA-256
# (requirements.lock). SOURCE_DATE_EPOCH (the commit time) sets the router's
# build time; the wheel's entries are sorted (canonicalize_wheel.py), and
# no bytecode is written, so two clean builds give the same image digest.
#
# The router-wheel stage is also the engine image's router
# (confidential/engine/Dockerfile copies it verbatim): the DeepSeek-V4.1
# launcher runs the router inside the pair pod.

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

FROM docker.io/library/python@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
LABEL org.opencontainers.image.title="confidential-sglang router"
LABEL org.opencontainers.image.source="https://github.com/confidential-dot-ai/confidential-sglang"
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
LABEL org.opencontainers.image.version="sha-${SOURCE_REVISION}"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/python"
LABEL org.opencontainers.image.base.digest="sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
LABEL ai.confidential.build.source-date-epoch="${SOURCE_DATE_EPOCH}"
ENV CI_DIAGNOSTIC_SOURCE_REVISION=${SOURCE_REVISION} \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
# No pip cache: with one, pip writes its self-check state with the wall-clock
# time under /root/.cache/pip. No bytecode: a .pyc records the mtime of its
# source at import time, which is the install time.
RUN --mount=type=bind,source=confidential/router/requirements.lock,target=/inputs/requirements.lock,readonly \
    --mount=type=bind,from=router-wheel,source=/out,target=/router-wheels,readonly \
    python3 -m pip install --no-cache-dir --disable-pip-version-check --no-compile \
        --require-hashes -r /inputs/requirements.lock \
    && python3 -m pip install --no-cache-dir --disable-pip-version-check --no-compile --no-deps \
        /router-wheels/sglang_router-*.whl \
    && python3 -B -c 'import sglang_router.sglang_router_rs, sglang_router.launch_router'
COPY --chmod=0555 confidential/router/private_router.py /usr/local/bin/ci-sglang-router
USER 1001:1001
EXPOSE 30000 29000
# The chart runs `python3 /usr/local/bin/ci-sglang-router ARGS`; the launcher
# runs `python3 -m sglang_router.launch_router ARGS --log-level warn` and
# keeps its output private (README.md).
ENTRYPOINT ["python3", "/usr/local/bin/ci-sglang-router"]
