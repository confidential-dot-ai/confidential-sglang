#!/usr/bin/env python3
"""List every file of every layer of an OCI image archive, with its SHA-256.

  oci_listing.py IMAGE.tar > listing.jsonl

One JSON object per line: the manifest and config digests first, then one
line per layer entry (layer index, path, type, mode, uid, gid, mtime, size,
sha256 or link target). Two clean builds that should be identical give equal
listings; when their digests differ, `diff` of the two listings shows the
files that differ. Standard library only; streams the layers.
"""
import hashlib
import json
import sys
import tarfile


def blob(outer, digest):
    return outer.extractfile("blobs/" + digest.replace(":", "/"))


def main(path):
    out = sys.stdout
    with tarfile.open(path) as outer:
        index = json.load(outer.extractfile("index.json"))
        manifest_digest = index["manifests"][0]["digest"]
        manifest = json.load(blob(outer, manifest_digest))
        config = json.load(blob(outer, manifest["config"]["digest"]))
        out.write(json.dumps({"manifest": manifest_digest, "config": manifest["config"]["digest"]}) + "\n")
        out.write(json.dumps({"config_json": config}, sort_keys=True) + "\n")
        for number, layer in enumerate(manifest["layers"]):
            out.write(json.dumps({"layer": number, "digest": layer["digest"], "size": layer["size"],
                                  "diff_id": config["rootfs"]["diff_ids"][number]}) + "\n")
            mode = "r|gz" if layer["mediaType"].endswith("gzip") else "r|"
            with tarfile.open(fileobj=blob(outer, layer["digest"]), mode=mode) as inner:
                for entry in inner:
                    row = {"l": number, "p": entry.name, "t": entry.type.decode(), "m": oct(entry.mode),
                           "u": entry.uid, "g": entry.gid, "mt": entry.mtime, "s": entry.size}
                    if entry.isfile():
                        digest = hashlib.sha256()
                        handle = inner.extractfile(entry)
                        for chunk in iter(lambda: handle.read(1 << 20), b""):
                            digest.update(chunk)
                        row["h"] = digest.hexdigest()
                    elif entry.issym() or entry.islnk():
                        row["ln"] = entry.linkname
                    if entry.pax_headers:
                        row["x"] = {k: v for k, v in sorted(entry.pax_headers.items())}
                    out.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
