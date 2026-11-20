#!/usr/bin/env bash
# Check the engine image against the chart's worker contracts without a GPU:
#   confidential/engine/check-image.sh IMAGE
# Runs the image as the chart does: uid 1001, read-only root filesystem, no
# network, no entrypoint (the image has none; the chart sets every command).
# Exits non-zero on the first failed check.
set -euo pipefail
image=${1:?usage: check-image.sh IMAGE}

# Image configuration: no entrypoint or command, the chart's user and HOME.
config=$(docker image inspect --format '{{json .Config}}' "$image")
python3 - "$config" <<'PY'
import json, sys
config = json.loads(sys.argv[1])
assert not config.get("Entrypoint"), f"image has an entrypoint: {config.get('Entrypoint')}"
assert not config.get("Cmd"), f"image has a command: {config.get('Cmd')}"
assert config.get("User") == "1001:1001", f"user is {config.get('User')!r}, not 1001:1001"
env = dict(item.split("=", 1) for item in config.get("Env") or [])
assert env.get("HOME") == "/tmp", f"HOME is {env.get('HOME')!r}"
print("config: no entrypoint/cmd, user 1001:1001, HOME=/tmp")
PY

docker run --rm --network=none --read-only --tmpfs /tmp --user 1001:1001 \
  --entrypoint /bin/sh "$image" -euc '
test "$(id -u):$(id -g)" = 1001:1001
test "$HOME" = /tmp

# Helpers at the paths the chart runs them by.
for f in /usr/local/bin/wait-for-model /usr/local/bin/fetch-model /usr/local/bin/sglang-pair \
         /usr/local/bin/dcgm-exporter /opt/dsv41/bin/python; do
  test -x "$f" || { echo "missing executable $f" >&2; exit 1; }
done
for f in /usr/local/bin/pair_diagnostics.py /usr/local/bin/private_log_sink.py \
         /etc/dcgm-exporter/counters.csv /opt/sglang/SOURCE_RECORD.json; do
  test -r "$f" || { echo "missing file $f" >&2; exit 1; }
done
command -v curl >/dev/null
command -v python3 >/dev/null
test "$(readlink -f "$(command -v python3)")" = "$(readlink -f /opt/dsv41/bin/python)" \
  || { echo "python3 on PATH is not /opt/dsv41/bin/python" >&2; exit 1; }

# Both recipes.
for f in /opt/sglang/recipes/deepseek-v41/launcher/launch.py \
         /opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py \
         /opt/sglang/recipes/deepseek-v41/loadtest/cluster_stress.py \
         /opt/sglang/recipes/qwen35-0.8b/chart-values.yaml \
         /opt/sglang/recipes/qwen35-0.8b/container/model-manifest.json; do
  test -r "$f" || { echo "missing recipe file $f" >&2; exit 1; }
done

/usr/local/bin/wait-for-model --help >/dev/null
/usr/local/bin/fetch-model --help >/dev/null
python3 /usr/local/bin/sglang-pair --help >/dev/null
python3 /opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py --help >/dev/null
python3 /opt/sglang/recipes/deepseek-v41/launcher/launch.py --help >/dev/null
python3 -I -B /usr/local/bin/pair_diagnostics.py
# sglang-pair marks a worker ready only from these per-rank progress records.
python3 -B -c "from sglang.srt.utils import pair_progress; pair_progress.initialize; pair_progress.scheduler_ready; pair_progress.stage"

python3 -B - <<PY
import importlib.util, json, pathlib

# HTTP policy (harden_http.py --variant pd): serving routes only, no docs.
import sglang.srt.entrypoints.http_server as http
paths = sorted({p for r in http.app.routes if (p := getattr(r, "path", None))})
for banned in ("/flush_cache", "/update_weights_from_disk", "/start_profile", "/openapi.json",
               "/docs", "/redoc", "/abort_request", "/set_internal_state", "/configure_logging",
               "/scale_elastic_ep", "/plugin_weight_update", "/get_hicache_l3_cache_layout"):
    assert banned not in paths, banned
for needed in ("/health", "/v1/chat/completions", "/v1/completions", "/v1/models", "/generate"):
    assert needed in paths, needed
assert http.app.openapi_url is None and http.app.docs_url is None and http.app.redoc_url is None
sglang = pathlib.Path(http.__file__).resolve().parents[2]
src = lambda rel: (sglang / rel).read_text()
assert "only the hardened Python HTTP frontend" in src("launch_server.py")
assert "This image does not log or dump request data" in src("srt/entrypoints/http_server.py")
# Media policy (harden_media.py): inline media only.
assert "def inline_media_container" in src("srt/multimodal/media_io.py")
assert "def download_remote_media(url: str, timeout: float) -> bytes:\n    raise ValueError" in src("srt/multimodal/media_io.py")
# Listener policy (harden_dist.py): loopback internal listeners, single node.
assert "_pin_listeners_to_loopback(dist_port)" in src("srt/distributed/bootstrap.py")
assert "_refuse_non_local_listeners(resolving_view(server_args))" in src("srt/entrypoints/engine.py")
print("routes:", len(paths), " ".join(paths))

# FlashInfer prebuilt kernels for B200 (sm100a) and B300 (sm103a).
import flashinfer, flashinfer_cubin, flashinfer_jit_cache
root = pathlib.Path(flashinfer_jit_cache.__file__).parent / "providers"
missing = [a for a in ("sm100a", "sm103a") if not (root / a / "jit_cache/sampling/sampling.so").is_file()]
assert not missing, f"jit-cache providers missing sampling.so: {missing}"
print("flashinfer", flashinfer.__version__, "jit-cache", flashinfer_jit_cache.__version__,
      "cubin", flashinfer_cubin.__version__)

# Baked DeepSeek-V4.1 launch plans, one per recipes/deepseek-v41/container/*.env.
layouts = pathlib.Path("/opt/sglang/layouts")
names = sorted(p.stem for p in layouts.glob("*.json"))
for name in ("dsv41", "dsv41-262k", "dsv41-b200-1m", "dsv41-b300-1m", "dsv41-b300-1m-ced"):
    assert name in names, f"missing layout {name}: {names}"
for path in layouts.glob("*.json"):
    plan = json.loads(path.read_text())
    assert "/data0" not in path.read_text(), f"{path} names a host path"
print("layouts:", " ".join(names))

# CPU simulator (chart inference.mode: simulator), outside the default import path.
import sys
sys.path.insert(0, "/opt/confidential-inference/sglang-simulator/src")
assert importlib.util.find_spec("sglang_simulator.simulation.sglang.launch_server")

print("source:", json.load(open("/opt/sglang/SOURCE_RECORD.json")))
PY

# Nothing under /opt is writable by the runtime user.
w=$(find /opt -xdev -writable -print -quit 2>/dev/null || true)
test -z "$w" || { echo "writable under /opt: $w" >&2; exit 1; }
echo "contract OK"
'
