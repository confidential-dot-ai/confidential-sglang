"""confidential/layouts/build_layouts.py: every recipe profile becomes a
deterministic container-path launch plan (python3 -m pytest confidential/tests/layouts)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "confidential/layouts/build_layouts.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
LAYOUTS = sorted(p.stem for p in (REPO / "recipes/deepseek-v41/container").glob("*.env"))

TOP_KEYS = {
    "schema", "layout", "source_env", "revision", "target", "paths", "launch_argv", "plan",
}
TARGET_KEYS = {"gpu_count", "gpu_models", "memory_profile", "host_available_gib", "host_gpus"}
PATH_KEYS = {
    "install_root", "python", "launcher", "engram_holder", "model", "kernel_cache_dir",
    "output_dir",
}
PLAN_KEYS = {"revision", "status", "physical_gpus", "topology", "collectives", "workers", "router"}


def build(out: Path, *extra: str) -> None:
    subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), "--out", str(out), "--revision", REVISION,
         *extra],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    first = tmp_path_factory.mktemp("first")
    second = tmp_path_factory.mktemp("second")
    build(first)
    build(second)
    return first, second


def test_every_env_profile_has_a_layout():
    assert LAYOUTS == [
        "dsv41", "dsv41-1m", "dsv41-262k", "dsv41-b200-1m", "dsv41-b300-1m",
        "dsv41-b300-1m-ced",
    ]


def test_build_is_byte_identical(built):
    first, second = built
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(f"{name}.json" for name in LAYOUTS)
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


@pytest.mark.parametrize("name", LAYOUTS)
def test_schema_and_container_paths(built, name):
    text = (built[0] / f"{name}.json").read_text()
    assert "/data" not in text
    doc = json.loads(text)
    assert set(doc) == TOP_KEYS
    assert doc["schema"] == "confidential-sglang/dsv41-layout/v1"
    assert doc["layout"] == name
    assert doc["source_env"] == f"recipes/deepseek-v41/container/{name}.env"
    assert doc["revision"] == REVISION
    assert set(doc["target"]) == TARGET_KEYS
    assert set(doc["paths"]) == PATH_KEYS
    plan = doc["plan"]
    assert PLAN_KEYS <= set(plan)
    assert plan["revision"] == REVISION
    assert plan["physical_gpus"] == list(range(doc["target"]["gpu_count"]))
    for item in (*plan["workers"], plan["router"]):
        assert item["command"][0] == "/opt/dsv41/bin/python"
        assert item["env"]["PYTHONPATH"] == "/opt/sglang/python"
        assert item["url"].startswith("http://127.0.0.1:")
    for worker in plan["workers"]:
        command = worker["command"]
        assert command[command.index("--model-path") + 1] == doc["paths"]["model"]
        assert worker["env"]["SGLANG_CACHE_DIR"] == doc["paths"]["kernel_cache_dir"]
        assert worker["env"]["TRITON_CACHE_DIR"].startswith("/tmp/")
        # A kernel-cache seed is the kernel_cache_dir tree: every compiler cache
        # whose default is outside SGLANG_CACHE_DIR must be inside it.
        for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                    "CUDA_CACHE_PATH", "SGLANG_JIT_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUTE_DSL_CACHE_DIR"):
            assert worker["env"][key].startswith(doc["paths"]["kernel_cache_dir"] + "/"), key
    # The baked argv reproduces the plan with the image's launcher.
    assert "--dry-run" not in doc["launch_argv"]


def test_hicache_auto_is_planned_from_the_recorded_host_ram(built):
    doc = json.loads((built[0] / "dsv41-b200-1m.json").read_text())
    assert doc["target"]["host_available_gib"] == 768
    assert doc["plan"]["hicache_auto"]["host_available_gib"] == 768.0
    one_m = json.loads((built[0] / "dsv41-1m.json").read_text())
    assert one_m["target"]["host_available_gib"] == 768
    assert one_m["target"]["gpu_count"] == 4
    assert one_m["plan"]["hicache_auto"]["ratio"] == 8.0
    assert one_m["plan"]["memory"]["per_rank"]["prefill"]["pool_tokens"] == 1572864
    assert one_m["plan"]["memory"]["per_rank"]["decode"]["contexts_resident"] == 6
    b300 = json.loads((built[0] / "dsv41-b300-1m-ced.json").read_text())
    assert b300["target"]["host_available_gib"] is None
    assert b300["target"]["memory_profile"] == "B300"
    assert "memory" in b300["plan"]


def test_revision_is_required_without_git(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), "--out", str(tmp_path), "--revision", "nope"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "revision" in completed.stderr
