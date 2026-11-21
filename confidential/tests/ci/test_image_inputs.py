"""confidential/ci/image_inputs.py: the input list covers every source of both
Dockerfiles, and the input commit skips commits that change no input
(python3 -m pytest confidential/tests/ci)."""

import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCKERFILES = sorted(REPO.glob("confidential/*/Dockerfile"))

spec = importlib.util.spec_from_file_location("image_inputs", REPO / "confidential/ci/image_inputs.py")
image_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_inputs)

INPUTS = image_inputs.read_inputs()


def _instructions(text):
    lines, current = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        lines.append(current + stripped)
        current = ""
    assert not current, "Dockerfile ends inside a continuation"
    return lines


def _build_context_sources(text):
    """COPY/ADD sources and RUN bind-mount sources read from the build context."""
    sources = []
    for line in _instructions(text):
        words = shlex.split(line, posix=True)
        keyword = words[0].upper()
        if keyword in ("COPY", "ADD"):
            flags = [w for w in words[1:] if w.startswith("--")]
            args = [w for w in words[1:] if not w.startswith("--")]
            if not any(f.startswith("--from=") for f in flags):
                sources += args[:-1]
        elif keyword == "RUN":
            for word in words[1:]:
                if not word.startswith("--mount="):
                    continue
                options = dict(kv.split("=", 1) for kv in word[len("--mount="):].split(",") if "=" in kv)
                if options.get("type", "bind") == "bind" and "from" not in options:
                    sources.append(options.get("source", options.get("src", ".")))
    return [s.strip("/") or "." for s in sources]


def _covered(path):
    return any(path == p or path.startswith(p + "/") for p in INPUTS)


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.parent.name)
def test_every_build_context_source_is_an_input(dockerfile):
    sources = _build_context_sources(dockerfile.read_text())
    assert sources, f"no build-context source parsed from {dockerfile}"
    missing = [s for s in sources if not _covered(s)]
    assert not missing, f"add to confidential/ci/image-inputs.txt: {missing}"


def test_the_parser_sees_copies_and_bind_mounts():
    sources = _build_context_sources((REPO / "confidential/engine/Dockerfile").read_text())
    assert "python" in sources
    assert "confidential/engine/requirements.lock" in sources
    assert "confidential/helpers/dcgm-counters.csv" in sources


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.parent.name)
def test_the_dockerfile_and_its_dockerignore_are_inputs(dockerfile):
    relative = dockerfile.relative_to(REPO).as_posix()
    assert relative in INPUTS
    assert relative + ".dockerignore" in INPUTS


def test_the_workflow_and_the_list_are_inputs():
    assert ".github/workflows/images.yml" in INPUTS
    assert "confidential/ci/image-inputs.txt" in INPUTS


def test_every_input_exists():
    missing = [p for p in INPUTS if not (REPO / p).exists()]
    assert not missing, f"stale entries in confidential/ci/image-inputs.txt: {missing}"


def _git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def _commit(repo, path, content, message):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


PATHS = ["python", "confidential/engine/Dockerfile"]


def test_a_commit_outside_the_inputs_keeps_the_input_commit(repo):
    built = _commit(repo, "python/sglang/x.py", "1", "input")
    _commit(repo, "confidential/ci/free-disk.sh", "1", "ci only")
    _commit(repo, "python2/y.py", "1", "a prefix of an input, not an input")
    assert image_inputs.input_commit(repo=repo, rev="HEAD", paths=PATHS) == built


def test_a_change_under_an_input_directory_moves_the_input_commit(repo):
    _commit(repo, "python/sglang/x.py", "1", "input")
    _commit(repo, "confidential/ci/free-disk.sh", "1", "ci only")
    changed = _commit(repo, "python/sglang/deep/z.py", "1", "nested input")
    assert image_inputs.input_commit(repo=repo, rev="HEAD", paths=PATHS) == changed


def test_a_merged_input_change_gives_the_merge_commit(repo):
    """A record names a main commit; never a commit of the merged branch."""
    _commit(repo, "python/sglang/x.py", "1", "input")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "confidential/engine/Dockerfile", "FROM x", "branch input")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "README.md", "1", "main, not an input")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    merge = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "docs/x.md", "1", "docs only")
    assert image_inputs.input_commit(repo=repo, rev="HEAD", paths=PATHS) == merge


def test_no_input_commit_is_an_error(repo):
    _commit(repo, "README.md", "1", "not an input")
    with pytest.raises(ValueError):
        image_inputs.input_commit(repo=repo, rev="HEAD", paths=PATHS)
