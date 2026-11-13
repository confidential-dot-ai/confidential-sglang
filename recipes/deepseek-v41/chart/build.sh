#!/usr/bin/env bash
# Build the DeepSeek-V4.1-Flash confidential-inference worker image from a
# clean checkout.
#   IMAGE=ghcr.io/confidential-dot-ai/sglang-dsv41-flash BASE_IMAGE=... build.sh
# Tags: $IMAGE:<commit[:12]> and $IMAGE:latest. Then check-image.sh $IMAGE:<tag>.
set -euo pipefail

chart_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$chart_dir/../../.." && pwd)
image=${IMAGE:-sglang-dsv41-flash}
cd "$repo_root"

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "the checkout has changes; commit them or set ALLOW_DIRTY=1" >&2
  exit 1
fi
revision=$(git rev-parse HEAD)
diff_sha256=$(git diff HEAD | sha256sum | cut -d' ' -f1)
status=$([[ -z "$(git status --porcelain)" ]] && echo clean || echo dirty)
record=$(printf '{"revision":"%s","status":"%s","diff_sha256":"%s","recipe":"deepseek-v41-chart"}' \
  "$revision" "$status" "$diff_sha256")

base=()
[[ -n "${BASE_IMAGE:-}" ]] && base=(--build-arg "BASE_IMAGE=$BASE_IMAGE")

DOCKER_BUILDKIT=1 docker build \
  --file "$chart_dir/Dockerfile" \
  --build-arg "SOURCE_RECORD=$record" \
  "${base[@]}" \
  --tag "$image:${revision:0:12}" \
  --tag "$image:latest" \
  "$repo_root"

echo "built $image:${revision:0:12}"
