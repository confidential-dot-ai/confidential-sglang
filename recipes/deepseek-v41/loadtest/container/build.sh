#!/usr/bin/env bash
# Build the cluster stress image.
#
#   container/build.sh DATASET TOKENIZER_DIR [IMAGE]
#
# DATASET is the realmix Oracle dataset (oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json),
# TOKENIZER_DIR the MiniMax-M3 tokenizer directory (tokenizer.json). The image
# defaults to cluster-stress:<git revision of this directory>. Set BUILD_NETWORK=host
# when the Docker bridge has no outbound route (as on the confos VMs).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tools=$(dirname "$here")
dataset=${1:?dataset json}
tokenizer=${2:?tokenizer directory}
revision=${REVISION:-$(git -C "$tools" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}
image=${3:-cluster-stress:$revision}

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/tools" "$ctx/data/minimax-m3"
cp "$here/Dockerfile" "$here/requirements.txt" "$here/fetch_kubectl.py" "$ctx/"
cp "$tools"/*.py "$tools"/CLUSTER-STRESS.md "$ctx/tools/"
cp "$dataset" "$ctx/data/oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json"
cp "$tokenizer"/tokenizer.json "$tokenizer"/tokenizer_config.json "$ctx/data/minimax-m3/"
(cd "$ctx/data" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
cat "$ctx/data/SHA256SUMS"

base=${PYTHON_IMAGE:-python:3.12-slim}
docker pull -q "$base" >/dev/null || true
base_digest=$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')
docker build --network "${BUILD_NETWORK:-default}" \
  --build-arg PYTHON_IMAGE="$base_digest" \
  --build-arg REVISION="$revision" \
  --label org.opencontainers.image.title=cluster-stress \
  --label org.opencontainers.image.source=https://github.com/confidential-dot-ai/confidential-sglang \
  --label org.opencontainers.image.revision="$revision" \
  --label org.opencontainers.image.base.name="$base_digest" \
  --label ai.confidential.stress.dataset.sha256="$(sha256sum "$dataset" | cut -d' ' -f1)" \
  -t "$image" "$ctx"
echo "built $image (base $base_digest)"
