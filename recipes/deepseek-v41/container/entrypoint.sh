#!/usr/bin/env bash
# `checkpoint` assembles $MODEL from pinned Hub sources when CHECKPOINT_SPEC names
# one, `engram` holds the Engram pages, `pd` runs one PD pair through tp2-pd.sh
# with the holder's PID; anything else runs as given (e.g. a probe).
set -euo pipefail
cd /opt/sglang
export MODEL="${MODEL:-/data/DeepSeek-V4.1-Flash}"
READY_FILE="${ENGRAM_READY_FILE:-/run/dsv41/engram.ready}"

case "${1:-pd}" in
checkpoint)
    shift
    if [[ -z "${CHECKPOINT_SPEC:-}" ]]; then
        echo "No CHECKPOINT_SPEC; using $MODEL as is"
        exit 0
    fi
    exec python recipes/deepseek-v41/launcher/prepare_checkpoint.py \
        --spec "$CHECKPOINT_SPEC" --local-dir "$MODEL" \
        ${CHECKPOINT_SEED:+--seed-dir "$CHECKPOINT_SEED"} "$@"
    ;;
engram)
    shift
    exec python recipes/deepseek-v41/launcher/hold_engram_cache.py \
        --model "$MODEL" --ready-file "$READY_FILE" "$@"
    ;;
pd)
    shift
    if [[ $# -eq 0 ]]; then
        # No launch flags: run a profile baked into the image, so the image alone
        # carries a complete configuration. Its GPUS and ports apply unless set.
        profile="recipes/deepseek-v41/container/${DSV41_PROFILE:-dsv41-b200-1m}.env"
        if [[ ! -f "$profile" ]]; then
            echo "No launch flags and no profile $profile" >&2
            exit 1
        fi
        for var in GPUS PORT BOOTSTRAP_PORT NCCL_PORT ROUTER_METRICS_PORT; do
            value="$(sed -n "s/^$var=//p" "$profile")"
            if [[ -z "${!var:-}" && -n "$value" ]]; then
                export "$var=$value"
            fi
        done
        read -r -a profile_args <<< "$(sed -n 's/^DSV41_LAUNCH_ARGS=//p' "$profile")"
        set -- "${profile_args[@]}"
        echo "Launch profile $profile (GPUS=$GPUS)" >&2
    fi
    pid="$(cat "$READY_FILE")"
    # Both containers share the host PID namespace, so this is the holder's host PID.
    if ! grep -qa hold_engram_cache "/proc/$pid/cmdline" 2>/dev/null; then
        echo "Engram holder PID $pid from $READY_FILE is not running" >&2
        exit 1
    fi
    export PYTHON=python
    export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-/data/dsv41-bringup/recipes}/container-pd-$(date -u +%Y%m%dT%H%M%SZ)}"
    # For the compose healthcheck, which finds smoke.json there.
    echo "$OUTPUT_DIR" > /tmp/output-dir
    exec recipes/deepseek-v41/tp2-pd.sh --resident-engram-pid "$pid" "$@"
    ;;
*)
    exec "$@"
    ;;
esac
