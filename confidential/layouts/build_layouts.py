#!/usr/bin/env python3
"""Bake the DeepSeek-V4.1 launch plans of the engine image.

For each docker-recipe profile recipes/deepseek-v41/container/<layout>.env this
runs recipes/deepseek-v41/launcher/launch.py --dry-run with the arguments the
recipe passes (entrypoint.sh `pd` -> tp2-pd.sh -> _common.sh run_launch), with
every host path replaced by the engine image's container paths, and writes
<out>/<layout>.json. README.md in this directory documents the schema.

Standard library only. It never imports SGLang, reads no GPU and no host RAM,
and the output depends only on the source tree and the revision:

    python3 -I -B confidential/layouts/build_layouts.py --out /opt/sglang/layouts

The revision is --revision, else $SOURCE_REVISION, else SOURCE_RECORD.json at
the repository root, else `git rev-parse HEAD`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "recipes/deepseek-v41"
ENV_DIR = RECIPE / "container"
LAUNCH = RECIPE / "launcher/launch.py"

SCHEMA = "confidential-sglang/dsv41-layout/v1"

# Container paths of the engine image (README.md, "Container paths").
INSTALL_ROOT = "/opt/sglang"
PYTHON = "/opt/dsv41/bin/python"
# The chart mounts each worker's model at inference.model.volumeDirectory
# (/mnt/c8s-data/models) / <volume name>. The plans use this volume name;
# substitute plan "paths.model" when a deployment names it differently.
MODEL = "/mnt/c8s-data/models/deepseek-v41-flash"
# Writable at run time under the chart (HOME=/tmp, read-only root otherwise).
RUN_ROOT = "/tmp/dsv41"
KERNEL_CACHE_DIR = f"{RUN_ROOT}/kernel-cache"
OUTPUT_DIR = f"{RUN_ROOT}/run"
# The recipe always runs with the Engram holder (hold_engram_cache.py) and
# passes its PID; the PID only selects the "resident holder" branch of the plan
# (no Engram tables counted against host RAM) and is not written to the plan.
RESIDENT_ENGRAM_PID = "1"

# Host RAM --prefill-hicache-ratio auto may use, per layout, instead of reading
# /proc/meminfo at build time. The B200 target host is 8x B200 with 1 TiB, where
# the holder pins the one shared ~198 GiB copy of the Engram tables: 768 GiB
# remain, enough that auto lands on its cap (dsv41.env). The B300 layouts size
# HiCache in GiB (--prefill-hicache-size-gib) and never use auto.
HOST_AVAILABLE_GIB = {
    "dsv41": 768,
    "dsv41-262k": 768,
    "dsv41-b200-1m": 768,
}

# The recipe's fixed launch flags (tp2-pd.sh).
RECIPE_FLAGS = ["--cpu-engram-staging", "--split-cuda-graphs"]
# Keys the recipe reads from the env file into _common.sh's flags.
PORT_FLAGS = (
    ("PORT", "--port", "8040"),
    ("BOOTSTRAP_PORT", "--bootstrap-port", "8999"),
    ("NCCL_PORT", "--nccl-port", "29600"),
    ("ROUTER_METRICS_PORT", "--router-metrics-port", "29001"),
)


def read_env(path: Path) -> dict[str, str]:
    """compose --env-file syntax: KEY=VALUE to the end of the line, no expansion."""
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"{path.name}: not KEY=VALUE: {line!r}")
        values[key] = value
    return values


def flag_value(argv: list[str], flag: str) -> str | None:
    """The last value of `flag` in argv (argparse keeps the last one)."""
    value = None
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif item.startswith(flag + "="):
            value = item.split("=", 1)[1]
    return value


# Flags whose profile values are host paths or host GPU indices; the plan sets
# its own container values instead.
CONTAINER_FLAGS = (
    "--model", "--python", "--gpus", "--output-dir", "--install-root",
    "--kernel-cache-dir", "--source-revision", "--host-available-gib",
)


def without_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """argv without `flags` and their values (both `--f v` and `--f=v`)."""
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
        elif item in flags:
            skip = True
        elif not item.startswith(tuple(f + "=" for f in flags)):
            out.append(item)
    return out


def revision_of(explicit: str | None) -> str:
    revision = explicit or os.environ.get("SOURCE_REVISION") or None
    record = REPO / "SOURCE_RECORD.json"
    if not revision and record.exists():
        revision = json.loads(record.read_text()).get("revision")
    if not revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(
            "no source revision: pass --revision, set SOURCE_REVISION, or build "
            "from a git checkout"
        )
    return revision


def layout_argv(name: str, env: dict[str, str], revision: str) -> list[str]:
    """launch.py arguments for one layout, in the recipe's order, container paths."""
    argv = ["--model", MODEL]
    for key, flag, default in PORT_FLAGS:
        argv += [flag, env.get(key) or default]
    argv += [*RECIPE_FLAGS, "--output-dir", OUTPUT_DIR]
    argv += ["--resident-engram-pid", RESIDENT_ENGRAM_PID]
    argv += without_flags(
        env.get("DSV41_LAUNCH_ARGS", "").split() + env.get("DSV41_EXTRA_ARGS", "").split(),
        CONTAINER_FLAGS,
    )
    # Container values replace the profile's host values (removed above). GPUs
    # are not passed: in a pod the visible GPUs are 0..N-1, launch.py's default.
    argv += [
        "--python", PYTHON,
        "--install-root", INSTALL_ROOT,
        "--kernel-cache-dir", KERNEL_CACHE_DIR,
        "--source-revision", revision,
    ]
    if flag_value(argv, "--prefill-hicache-ratio") == "auto":
        if name not in HOST_AVAILABLE_GIB:
            raise SystemExit(f"{name}: HiCache auto needs a HOST_AVAILABLE_GIB entry")
        argv += ["--host-available-gib", str(HOST_AVAILABLE_GIB[name])]
    return argv


def build_one(name: str, revision: str) -> dict:
    env_path = ENV_DIR / f"{name}.env"
    env = read_env(env_path)
    argv = layout_argv(name, env, revision)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(LAUNCH), *argv, "--dry-run"],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{name}: launch.py --dry-run failed:\n{completed.stderr}")
    plan = json.loads(completed.stdout)
    num_gpus = len(plan["physical_gpus"])
    gpu_models = flag_value(argv, "--gpu-models") or "B300,GB300"
    return {
        "schema": SCHEMA,
        "layout": name,
        "source_env": str(env_path.relative_to(REPO)),
        "revision": revision,
        "target": {
            "gpu_count": num_gpus,
            "gpu_models": gpu_models.split(","),
            "memory_profile": flag_value(argv, "--memory-profile"),
            "host_available_gib": HOST_AVAILABLE_GIB.get(name)
            if flag_value(argv, "--prefill-hicache-ratio") == "auto"
            else None,
            "host_gpus": [int(g) for g in env["GPUS"].split(",")] if env.get("GPUS") else None,
        },
        "paths": {
            "install_root": INSTALL_ROOT,
            "python": PYTHON,
            "launcher": f"{INSTALL_ROOT}/recipes/deepseek-v41/launcher/launch.py",
            "engram_holder": f"{INSTALL_ROOT}/recipes/deepseek-v41/launcher/hold_engram_cache.py",
            "model": MODEL,
            "kernel_cache_dir": KERNEL_CACHE_DIR,
            "output_dir": OUTPUT_DIR,
        },
        "launch_argv": argv,
        "plan": plan,
    }


def layouts() -> list[str]:
    return sorted(p.stem for p in ENV_DIR.glob("*.env"))


def render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True, help="Directory for <layout>.json")
    p.add_argument("--revision", help="Source commit (40 hex)")
    p.add_argument("--layout", action="append", help="Only these layouts (default all)")
    args = p.parse_args(argv)
    revision = revision_of(args.revision)
    names = args.layout or layouts()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        text = render(build_one(name, revision))
        if "/data" in text:
            raise SystemExit(f"{name}: plan still holds a /data host path")
        (args.out / f"{name}.json").write_text(text)
        print(f"{args.out / (name + '.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
