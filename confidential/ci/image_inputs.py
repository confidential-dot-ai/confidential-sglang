#!/usr/bin/env python3
"""The input commit of the engine and router images.

The input commit of a revision is the newest commit on its first-parent line
that changed an image input (image-inputs.txt). Every input is byte-identical
at the revision and at its input commit, so the images built from both are
identical; the images workflow builds, tags and records the input commit.

  image_inputs.py commit [--repo DIR] [REV]   print the input commit of REV
  image_inputs.py paths                       print the input paths
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

INPUTS_FILE = Path(__file__).resolve().with_name("image-inputs.txt")
COMMIT = re.compile(r"[0-9a-f]{40}")


def read_inputs(path: Path = INPUTS_FILE) -> list[str]:
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if line.startswith("/") or line.endswith("/") or ".." in line.split("/"):
                raise ValueError(f"{path}: not a plain repository path: {line!r}")
            paths.append(line)
    if not paths:
        raise ValueError(f"{path}: no input paths")
    return paths


def input_commit(*, repo: Path, rev: str, paths: list[str]) -> str:
    # --first-parent: on main, the merge commit, never a commit of the merged branch.
    # ":(literal)": a path is never read as a glob.
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--first-parent", "--format=%H", "--end-of-options", rev, "--",
         *(f":(literal){p}" for p in paths)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not COMMIT.fullmatch(out):
        raise ValueError(f"no commit reachable from {rev} changes an image input")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    commit = sub.add_parser("commit")
    commit.add_argument("--repo", type=Path, default=Path("."))
    commit.add_argument("rev", nargs="?", default="HEAD")
    sub.add_parser("paths")
    args = parser.parse_args(argv)
    paths = read_inputs()
    if args.command == "paths":
        print("\n".join(paths))
    else:
        print(input_commit(repo=args.repo, rev=args.rev, paths=paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
