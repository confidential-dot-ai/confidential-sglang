#!/usr/bin/env bash
# Regenerate requirements.lock from requirements.in (exact PyPI pins, hashes of
# every published file) and flashinfer-wheels.txt (URL + sha256 pins).
#   confidential/engine/lock.sh        (needs uv)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
uv pip compile --no-deps --generate-hashes --no-header --no-annotate \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index-url https://pypi.org/simple "$here/requirements.in" -o "$tmp" >/dev/null
{
  sed -n '1,/^# Each PyPI pin/p' "$here/requirements.lock"
  cat "$tmp" "$here/flashinfer-wheels.txt"
} > "$here/requirements.lock.new"
mv "$here/requirements.lock.new" "$here/requirements.lock"
