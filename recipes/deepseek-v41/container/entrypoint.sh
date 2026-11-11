#!/usr/bin/env bash
# `checkpoint` assembles $MODEL from pinned Hub sources when CHECKPOINT_SPEC names
# one, `engram` holds the Engram pages, `pd` runs one PD pair through tp2-pd.sh
# with the holder's PID, `metrics` serves the pair's derived metrics to
# Prometheus; anything else runs as given (e.g. a probe).
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
metrics)
    # Prometheus exporter for the pair's derived metrics (loadtest/live_metrics.py
    # --serve): a sidecar beside `pd`, with the same PORT, GPUS and
    # DSV41_LAUNCH_ARGS, from which it finds the workers. The launcher puts
    # prefill i on PORT+1+i and decode j on PORT+1+num_prefill+j. It waits for
    # the workers, so it can start before the model is up.
    shift
    read -r -a launch <<< "${DSV41_LAUNCH_ARGS:-}"
    num_prefill=1 num_decode=1
    for ((i = 0; i < ${#launch[@]}; i++)); do
        case "${launch[i]}" in
            --num-prefill) num_prefill="${launch[i + 1]}" ;;
            --num-prefill=*) num_prefill="${launch[i]#*=}" ;;
            --num-decode) num_decode="${launch[i + 1]}" ;;
            --num-decode=*) num_decode="${launch[i]#*=}" ;;
        esac
    done
    base="${PORT:-8040}"
    args=()
    for ((i = 0; i < num_prefill; i++)); do args+=(--prefill "http://127.0.0.1:$((base + 1 + i))"); done
    for ((j = 0; j < num_decode; j++)); do args+=(--decode "http://127.0.0.1:$((base + 1 + num_prefill + j))"); done
    IFS=, read -r -a gpus <<< "${GPUS:-}"
    for label in ${METRICS_LABELS:-}; do args+=(--label "$label"); done
    exec python recipes/deepseek-v41/loadtest/live_metrics.py "${args[@]}" \
        --router "http://127.0.0.1:${ROUTER_METRICS_PORT:-29001}" --gpus "${#gpus[@]}" \
        --interval "${METRICS_INTERVAL:-10}" \
        --serve "${METRICS_PORT:-29100}" --serve-host "${METRICS_HOST:-127.0.0.1}" "$@"
    ;;
*)
    exec "$@"
    ;;
esac
