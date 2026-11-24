"""CPU-only: a JIT module that ninja builds is reported to a pair supervisor
(confidential-sglang), and a cache hit is not.

load_jit runs for real down to the compiler: ninja.build, the tvm-ffi load and
the dependency scan are stubbed, so no nvcc or GPU is needed.
"""

from __future__ import annotations

import json
import pathlib
import socket

import pytest

from sglang.kernels.jit.utils.compile import cache, loader, ninja
from sglang.srt.utils import pair_progress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """A pair run directory with the supervisor's datagram socket."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(str(run_dir / "events.sock"))
    receiver.settimeout(0.2)
    monkeypatch.setenv("SGLANG_PAIR_ROLE", "decode")
    monkeypatch.setenv("SGLANG_PAIR_RUN_DIR", str(run_dir))
    monkeypatch.setattr(pair_progress, "_reporter", None)
    yield receiver
    receiver.close()


@pytest.fixture
def stub_compiler(tmp_path, monkeypatch):
    """ninja.build writes a fake .so; the load and dependency scan are stubbed."""
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(tmp_path / "jit"))
    cache.clear_digest_cache()
    builds = []

    def build(*, spec, build_dir, build_file):
        builds.append(spec.module_name)
        build_dir.mkdir(parents=True, exist_ok=True)
        library = build_dir / f"{spec.module_name}.so"
        library.write_bytes(b"\x7fELF fake")
        return library

    monkeypatch.setattr(ninja, "build", build)
    monkeypatch.setattr(loader, "_load", lambda library: ("module", pathlib.Path(library).name))
    source = pathlib.Path(loader.__file__)
    monkeypatch.setattr(ninja, "scan_dependencies", lambda staging: [source])
    yield builds
    cache.clear_digest_cache()


def _receive(receiver):
    try:
        return json.loads(receiver.recv(4096))
    except socket.timeout:
        return None


def _load():
    return loader.load_jit(
        "rmsnorm", "bf16_t", "4096",
        cpp_wrappers=[("run", "Kernel::run")],
    )


def test_a_build_is_reported_by_family_only(pair, stub_compiler):
    _load()
    assert stub_compiler == ["sgl_kernel_jit_rmsnorm_bf16_t_4096"]
    event = _receive(pair)
    assert event is not None
    assert event["event"] == "jit_compile"
    assert event["role"] == "decode"
    # The template args (here a hidden size) never leave the process.
    assert event["kernel"] == "sgl_jit_rmsnorm"
    assert isinstance(event["duration_ms"], int) and event["duration_ms"] >= 0


def test_a_cache_hit_is_not_reported(pair, stub_compiler):
    _load()
    assert _receive(pair) is not None
    _load()
    assert stub_compiler == ["sgl_kernel_jit_rmsnorm_bf16_t_4096"]
    assert _receive(pair) is None


def test_event_names_stay_identifiers():
    # The supervisor drops a kernel that is not a 64-char identifier.
    assert loader.build_event_name(("qk_norm_rope", "128")) == "sgl_jit_qk_norm_rope"
    assert loader.build_event_name(("a b", "x")) == "sgl_jit_module"
    assert loader.build_event_name(()) == "sgl_jit_module"
    assert len(loader.build_event_name(("k" * 200,))) == 64
