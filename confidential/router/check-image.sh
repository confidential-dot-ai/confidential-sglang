#!/usr/bin/env bash
# Check a router image against the chart's router contract, without network:
#   confidential/router/check-image.sh IMAGE
# Runs as the chart does: uid 1001, read-only root filesystem, /tmp tmpfs.
set -euo pipefail
image=${1:?usage: check-image.sh IMAGE}
max_bytes=${ROUTER_MAX_IMAGE_BYTES:-600000000}

size=$(docker image inspect --format '{{.Size}}' "$image")
if (( size > max_bytes )); then
  echo "router image is $size bytes, above $max_bytes: it should hold no CUDA or engine" >&2
  exit 1
fi
test "$(docker image inspect --format '{{.Config.User}}' "$image")" = 1001:1001

docker run --rm --network=none --read-only --tmpfs /tmp --user 1001:1001 \
  --entrypoint /bin/sh "$image" -euc '
test "$(id -u):$(id -g)" = 1001:1001
test "$HOME" = /tmp
test -r /usr/local/bin/ci-sglang-router
python3 -B -m sglang_router.launch_router --help >/dev/null
python3 -B - <<PY
import importlib.util, json, os, re
import sglang_router.sglang_router_rs
for name in ("torch", "sglang", "flashinfer", "triton"):
    assert importlib.util.find_spec(name) is None, f"{name} is in the router image"
for root in ("/usr/local/cuda", "/opt/dsv41", "/opt/sglang"):
    assert not os.path.exists(root), root
# The launcher passes --json-log only when the router has it; this one must.
import sglang_router.router_args as ra
assert "json-log" in open(ra.__file__).read()
revision = os.environ.get("CI_DIAGNOSTIC_SOURCE_REVISION", "")
assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
print("router", sglang_router.__file__, "revision", revision)
PY
# Nothing outside /tmp is writable by the runtime user.
test -z "$(find /usr /opt /etc -xdev -writable -print -quit 2>/dev/null)"
echo "router contract OK"
'
