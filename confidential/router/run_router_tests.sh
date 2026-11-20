#!/usr/bin/env bash
# Run the router's Rust tests in the router image's pinned Rust build image.
#
#   confidential/router/run_router_tests.sh [WORK_DIR]
#
# The source is this checkout's sgl-model-gateway with its committed
# Cargo.lock (--locked). The tests need protoc (build.rs) and redis-server
# (the reliability tests' history backend), installed from the same
# snapshot.debian.org timestamp as the image's build stage.
# SOURCE_DATE_EPOCH is the commit time, as in the image build.
# WORK_DIR (default: a new temporary directory) holds the Cargo registry and
# target directory; reuse it to make a second run fast.
# Env: CARGO_JOBS (default: all cores).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="${ROOT}/confidential/router/Dockerfile"
WORK="${1:-$(mktemp -d)}"
mkdir -p "${WORK}/cargo-registry" "${WORK}/target"

image="$(sed -n 's/^FROM \(docker\.io\/library\/rust@sha256:[0-9a-f]\{64\}\) AS router-wheel$/\1/p' "${DOCKERFILE}")"
snapshot="$(sed -n 's/^ARG DEBIAN_SNAPSHOT=\([0-9]\{8\}T[0-9]\{6\}Z\)$/\1/p' "${DOCKERFILE}")"
[ -n "${image}" ] || { echo "no pinned Rust build image in ${DOCKERFILE}" >&2; exit 1; }
[ -n "${snapshot}" ] || { echo "no DEBIAN_SNAPSHOT in ${DOCKERFILE}" >&2; exit 1; }
epoch="${SOURCE_DATE_EPOCH:-$(git -C "${ROOT}" log -1 --format=%ct)}"
jobs="${CARGO_JOBS:-$(nproc)}"

echo "==> cargo test in ${image} (snapshot ${snapshot}, SOURCE_DATE_EPOCH=${epoch})"
docker run --rm \
  -e SOURCE_DATE_EPOCH="${epoch}" \
  -e CARGO_TARGET_DIR=/target \
  -e CARGO_BUILD_JOBS="${jobs}" \
  -e SNAPSHOT="${snapshot}" \
  -v "${ROOT}/sgl-model-gateway:/src/sgl-model-gateway:ro" \
  -v "${WORK}/cargo-registry:/usr/local/cargo/registry" \
  -v "${WORK}/target:/target" \
  -w /src/sgl-model-gateway \
  "${image}" \
  bash -euo pipefail -c '
    printf "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/%s bookworm main\n" "$SNAPSHOT" > /etc/apt/sources.list
    printf "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n" "$SNAPSHOT" >> /etc/apt/sources.list
    rm -f /etc/apt/sources.list.d/*
    apt-get update -qq >/dev/null
    apt-get install -y -qq --no-install-recommends protobuf-compiler libprotobuf-dev redis-server >/dev/null
    cargo test --locked --features vendored-openssl --lib
    cargo test --locked --features vendored-openssl --test reliability_tests
  '
