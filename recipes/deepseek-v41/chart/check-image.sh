#!/usr/bin/env bash
# Check an image against the chart's worker contracts without a GPU:
#   check-image.sh IMAGE
# Runs as the chart does (uid 1001, read-only root filesystem, no network).
set -euo pipefail
image=${1:?usage: check-image.sh IMAGE}

docker run --rm --network=none --read-only --tmpfs /tmp --user 1001:1001 \
  --entrypoint /bin/sh "$image" -euc '
test "$(id -u):$(id -g)" = 1001:1001
test "$HOME" = /tmp
for f in /usr/local/bin/wait-for-model /usr/local/bin/fetch-model /usr/local/bin/sglang-pair \
         /usr/local/bin/dcgm-exporter /opt/dsv41/bin/python; do
  test -x "$f" || { echo "missing executable $f" >&2; exit 1; }
done
test -r /opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py
test -r /etc/dcgm-exporter/counters.csv
command -v curl >/dev/null
command -v python3 >/dev/null
test "$(readlink -f "$(command -v python3)")" = "$(readlink -f /opt/dsv41/bin/python)"
/usr/local/bin/wait-for-model --help >/dev/null
/usr/local/bin/fetch-model --help >/dev/null
/opt/dsv41/bin/python /usr/local/bin/sglang-pair --help >/dev/null
/opt/dsv41/bin/python /opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py --help >/dev/null
/opt/dsv41/bin/python -I -B /usr/local/bin/pair_diagnostics.py
# sglang-pair marks a worker ready only from these per-rank progress records.
/opt/dsv41/bin/python -B -c "from sglang.srt.utils import pair_progress; pair_progress.initialize; pair_progress.scheduler_ready; pair_progress.stage"
/opt/dsv41/bin/python -B - <<PY
import json, pathlib
import sglang.srt.entrypoints.http_server as http
# The hardened route table: serving routes only, no docs.
paths = sorted({p for r in http.app.routes if (p := getattr(r, "path", None))})
for banned in ("/flush_cache", "/update_weights_from_disk", "/start_profile", "/openapi.json",
               "/docs", "/redoc", "/abort_request", "/set_internal_state", "/configure_logging"):
    assert banned not in paths, banned
for needed in ("/health", "/v1/chat/completions", "/v1/completions", "/v1/models", "/generate"):
    assert needed in paths, needed
src = pathlib.Path("/opt/sglang/python/sglang/launch_server.py").read_text()
assert "only the hardened Python HTTP frontend" in src
record = json.load(open("/opt/sglang/SOURCE_RECORD.json"))
print("routes:", " ".join(paths))
print("source:", record)
PY
# Nothing under /opt is writable by the runtime user.
test -z "$(find /opt/sglang /opt/dsv41 -writable -print -quit 2>/dev/null)"
echo "contract OK"
'
