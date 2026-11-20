#!/usr/bin/env bash
# The CPU test suites: helpers, layouts, release record, DSV4.1 launcher and
# load tests. Python 3.12 in a fresh venv from test-requirements.lock (hashes).
#   confidential/ci/python-tests.sh
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
venv=${TEST_VENV:-$(mktemp -d)/venv}
if command -v uv >/dev/null; then
  uv venv -q --python 3.12 "$venv"
  VIRTUAL_ENV=$venv uv pip install -q --no-deps --require-hashes -r confidential/ci/test-requirements.lock
else
  python3.12 -m venv "$venv"
  "$venv/bin/pip" install --quiet --disable-pip-version-check --no-deps --require-hashes \
    -r confidential/ci/test-requirements.lock
fi
py="$venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
"$py" -m pytest -q -p no:cacheprovider confidential/tests
"$py" -m pytest -q -p no:cacheprovider recipes/deepseek-v41/launcher
"$py" -m pytest -q -p no:cacheprovider recipes/deepseek-v41/loadtest
# The release record round trip.
tmp=$(mktemp -d)
d="sha256:$(printf '%064d' 0)"
"$py" -I confidential/ci/release_record.py write --commit "$(printf '%040d' 0)" --epoch 1 \
  --run-repository o/r --run-id 1 --run-attempt 1 --workflow w \
  --image "engine=ghcr.io/confidential-dot-ai/confidential-sglang/engine@$d,$d,$d" \
  --image "router=ghcr.io/confidential-dot-ai/confidential-sglang/router@$d,$d,$d" \
  --output "$tmp/r.json" >/dev/null
"$py" -I confidential/ci/release_record.py check "$tmp/r.json"
