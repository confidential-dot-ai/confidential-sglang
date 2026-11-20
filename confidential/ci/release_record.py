#!/usr/bin/env python3
"""Write and check the image release record of one confidential-sglang commit.

The images workflow writes it after both images are published, and attaches it
to the GitHub release ``images-<first 12 hex of the commit>`` as
``images-<12 hex>.json``. A consumer (confidential-inference) pins images from
it: it checks the record with ``check`` and uses ``images[].reference``.

Record (schema confidential.ai/confidential-sglang-images/v1)::

  {
    "schema": "confidential.ai/confidential-sglang-images/v1",
    "repository": "https://github.com/confidential-dot-ai/confidential-sglang",
    "sourceCommit": "<40 hex>",
    "sourceDateEpoch": <int, the commit time>,
    "run": {"repository": "<owner/name>", "id": <int>, "attempt": <int>,
            "url": "https://github.com/<owner/name>/actions/runs/<id>/attempts/<n>",
            "workflow": "<workflow ref>"},
    "images": [            # sorted by name
      {"name": "engine",
       "image": "ghcr.io/confidential-dot-ai/confidential-sglang/engine",
       "digest": "sha256:<64 hex>",            # the pushed manifest digest
       "reference": "<image>@<digest>",
       "tag": "sha-<40 hex>",
       "builds": ["sha256:...", "sha256:..."]}  # the two clean builds; equal
    ]
  }

  release_record.py write --commit C --epoch E --run-repository R --run-id N \
      --run-attempt A --workflow W --image NAME=IMAGE@DIGEST,BUILD1,BUILD2 ... --output F
  release_record.py check F [--commit C]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "confidential.ai/confidential-sglang-images/v1"
REPOSITORY = "https://github.com/confidential-dot-ai/confidential-sglang"
REGISTRY = "ghcr.io/confidential-dot-ai/confidential-sglang/"
NAMES = ("engine", "router")
COMMIT = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class RecordError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RecordError(message)


def check(record: dict, commit: str | None = None) -> dict:
    require(isinstance(record, dict), "record is not an object")
    require(set(record) == {"schema", "repository", "sourceCommit", "sourceDateEpoch", "run", "images"},
            f"record keys: {sorted(record)}")
    require(record["schema"] == SCHEMA, f"schema {record['schema']!r}")
    require(record["repository"] == REPOSITORY, f"repository {record['repository']!r}")
    source = record["sourceCommit"]
    require(isinstance(source, str) and COMMIT.fullmatch(source) is not None, "sourceCommit")
    require(commit is None or source == commit, f"sourceCommit {source} is not {commit}")
    require(isinstance(record["sourceDateEpoch"], int) and record["sourceDateEpoch"] > 0, "sourceDateEpoch")
    run = record["run"]
    require(set(run) == {"repository", "id", "attempt", "url", "workflow"}, f"run keys: {sorted(run)}")
    require(isinstance(run["id"], int) and isinstance(run["attempt"], int), "run id/attempt")
    require(run["url"] == f"https://github.com/{run['repository']}/actions/runs/{run['id']}/attempts/{run['attempt']}",
            "run url")
    images = record["images"]
    require([i.get("name") for i in images] == list(NAMES), f"images must be {list(NAMES)}, sorted")
    for entry in images:
        require(set(entry) == {"name", "image", "digest", "reference", "tag", "builds"}, f"image keys: {sorted(entry)}")
        require(entry["image"] == REGISTRY + entry["name"], f"image {entry['image']!r}")
        require(DIGEST.fullmatch(entry["digest"]) is not None, f"{entry['name']} digest")
        require(entry["reference"] == f"{entry['image']}@{entry['digest']}", f"{entry['name']} reference")
        require(entry["tag"] == f"sha-{source}", f"{entry['name']} tag")
        builds = entry["builds"]
        require(isinstance(builds, list) and len(builds) == 2, f"{entry['name']} needs two builds")
        require(all(b == entry["digest"] for b in builds), f"{entry['name']} builds differ from the pushed digest")
    return record


def parse_image(value: str) -> dict:
    match = re.fullmatch(r"([a-z]+)=([^@,]+)@(sha256:[0-9a-f]{64}),(sha256:[0-9a-f]{64}),(sha256:[0-9a-f]{64})", value)
    if not match:
        raise argparse.ArgumentTypeError(f"--image {value!r}: NAME=IMAGE@DIGEST,BUILD1,BUILD2")
    name, image, digest, one, two = match.groups()
    return {"name": name, "image": image, "digest": digest, "reference": f"{image}@{digest}", "builds": [one, two]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--commit", required=True)
    write.add_argument("--epoch", required=True, type=int)
    write.add_argument("--run-repository", required=True)
    write.add_argument("--run-id", required=True, type=int)
    write.add_argument("--run-attempt", required=True, type=int)
    write.add_argument("--workflow", required=True)
    write.add_argument("--image", required=True, action="append", type=parse_image)
    write.add_argument("--output", required=True, type=Path)
    read = sub.add_parser("check")
    read.add_argument("record", type=Path)
    read.add_argument("--commit")
    args = parser.parse_args(argv)
    try:
        if args.command == "write":
            images = sorted(({**i, "tag": f"sha-{args.commit}"} for i in args.image), key=lambda i: i["name"])
            record = {
                "schema": SCHEMA,
                "repository": REPOSITORY,
                "sourceCommit": args.commit,
                "sourceDateEpoch": args.epoch,
                "run": {
                    "repository": args.run_repository,
                    "id": args.run_id,
                    "attempt": args.run_attempt,
                    "url": f"https://github.com/{args.run_repository}/actions/runs/{args.run_id}/attempts/{args.run_attempt}",
                    "workflow": args.workflow,
                },
                "images": images,
            }
            check(record, args.commit)
            args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({i["name"]: i["reference"] for i in images}, indent=2))
        else:
            check(json.loads(args.record.read_text(encoding="utf-8")), args.commit)
            print(f"{args.record}: OK")
    except RecordError as error:
        print(f"release record: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
