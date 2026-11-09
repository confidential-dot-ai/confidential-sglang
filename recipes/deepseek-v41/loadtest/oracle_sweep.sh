#!/bin/bash
# Sweep Oracle session counts against a running pair; stop at the first level that
# leaves a new worker traceback/OOM or fails a chat check afterwards.
#
#   DATASET=/data/dsv41-bringup/datasets/oracle/oracle-coding-together-256-saturated-v1.json \
#   LOGS=/data/dsv41-bringup/recipes/container-pd-<ts> ./oracle_sweep.sh
#
# DATASET  Oracle dataset JSON (required)
# URL      router (http://127.0.0.1:8040); MODEL (dsv41)
# SWEEP    session counts ("16 32 64 128 256")
# MIN      minutes per level (12; 0 = replay every step)
# LOGS     the pair's run directory (prefill.log, decode.log): enables the
#          traceback check and the prefill prefix-cache hit per level
# OUT      results directory (oracle-<ts>); PY: python with aiohttp (python3;
#          the dsv41-pd image's /opt/dsv41/bin/python has it)
# SKIP_LONG=1 skips the ~200K-token prompt check (needs --context-length >= 210K)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${DATASET:?set DATASET to an Oracle dataset JSON}"
URL=${URL:-http://127.0.0.1:8040}
MODEL=${MODEL:-dsv41}
PY=${PY:-python3}
OUT=${OUT:-oracle-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"

for _ in $(seq 1 120); do curl -sf -m 5 -o /dev/null "$URL/readiness" && break; sleep 15; done
curl -sf -m 5 -o /dev/null "$URL/readiness" || { echo "not ready: $URL"; exit 1; }
echo "ready $(date -u +%T)"

tb() {
  [ -n "${LOGS:-}" ] || { echo 0; return; }
  cat "$LOGS/prefill.log" "$LOGS/decode.log" 2>/dev/null | grep -c "Traceback\|out of memory"
}
chat() {
  curl -s -m 120 "$URL/v1/chat/completions" -H content-type:application/json \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}" |
    grep -q '"content"'
}
T0=$(tb)

if [ -z "${SKIP_LONG:-}" ]; then
  # "code " is ~1 DeepSeek token, so this is a ~200K-token prompt: the first eager
  # prefill chunks past the 65K graph cap are where 12K chunks OOMed (2026-11-09).
  "$PY" -c "import json; print(json.dumps({'model': '$MODEL', 'max_tokens': 8, 'messages': [
    {'role': 'system', 'content': 'code ' * 200000}, {'role': 'user', 'content': 'Reply OK'}]}))" > "$OUT/long.json"
  curl -s -m 600 -o /dev/null -w "long prompt: HTTP %{http_code} in %{time_total}s\n" \
    "$URL/v1/chat/completions" -H content-type:application/json -d @"$OUT/long.json"
  if [ "$(tb)" != "$T0" ] || ! chat; then echo "FAILED long-prompt check"; exit 1; fi
fi

for s in ${SWEEP:-16 32 64 128 256}; do
  echo "== $(basename "$DATASET"): $s sessions, ${MIN:-12} min"
  L0=0
  [ -n "${LOGS:-}" ] && L0=$(wc -l < "$LOGS/prefill.log")
  "$PY" "$HERE/oracle_replay.py" "$DATASET" --url "$URL" --model "$MODEL" --sessions "$s" \
    --time-limit $(( ${MIN:-12} * 60 )) --output "$OUT/oracle-s$s.jsonl" --report-every 60 |
    tee "$OUT/oracle-s$s.log"
  if [ -n "${LOGS:-}" ]; then
    tail -n +$((L0 + 1)) "$LOGS/prefill.log" | grep -o '#new-token: [0-9]*, #cached-token: [0-9]*' |
      awk '{n += $2; c += $4} END {if (n + c) printf "prefill cache hit: %.1f%% (%d new, %d cached tokens)\n", 100 * c / (n + c), n, c}'
  fi
  if [ "$(tb)" != "$T0" ] || ! chat; then echo "FAILED after $s sessions"; exit 1; fi
  echo "LEVEL OK $s"
done
echo "SWEEP DONE $(date -u +%T)"
