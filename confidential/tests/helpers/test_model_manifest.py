"""The model identity does not depend on its provider or weight format."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


HELPERS = Path(__file__).resolve().parents[2] / "helpers"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HELPERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("wait_for_model")
fetch = load("fetch_model")


class GenericManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.gguf").write_bytes(b"provider-independent model")
        (self.model / "empty-config").touch()
        raw = fetch.local_manifest(self.model)
        self.digest = hashlib.sha256(raw).hexdigest()
        (self.model / "manifest.json").write_bytes(raw)
        self.mountinfo = self.root / "mountinfo"
        self.mountinfo.write_text(f"44 33 0:77 / {self.model} ro - erofs /dev/mapper/c8s-verity-model ro\n")
        self.args = SimpleNamespace(path=self.model, mountinfo=self.mountinfo,
                                    plain_cache=False, expected_file=[("manifest.json", self.digest)],
                                    revision_metadata="manifest.json", expected_repository="", expected_revision="")

    def verify(self):
        # Mount write access has its own existing tests. This test isolates byte identity.
        from unittest.mock import patch
        with patch.object(gate, "verify_no_write"):
            gate.verify_once(self.args)

    def test_non_hugging_face_non_safetensors_model(self):
        self.verify()

    def test_manifest_is_reproducible(self):
        (self.model / "manifest.json").unlink()
        self.assertEqual(self.digest, hashlib.sha256(fetch.local_manifest(self.model)).hexdigest())

    def test_wrong_manifest_pin(self):
        self.args.expected_file = [("manifest.json", "0" * 64)]
        with self.assertRaisesRegex(gate.ModelMountError, "wrong digest"):
            self.verify()

    def test_explicit_manifest_argument_needs_no_provider_metadata(self):
        from unittest.mock import patch
        argv = ["wait-for-model", "--path", str(self.model), "--timeout-seconds", "1",
                "--revision-metadata", "manifest.json", "--model-manifest-sha256", self.digest,
                "--", "python3", "-m", "sglang.launch_server"]
        with patch("sys.argv", argv):
            args = gate.parse_args()
        self.assertEqual([("manifest.json", self.digest)], args.expected_file)
        self.assertEqual("", args.expected_repository)

    def test_conflicting_manifest_arguments_are_refused(self):
        from unittest.mock import patch
        argv = ["wait-for-model", "--path", str(self.model), "--timeout-seconds", "1",
                "--revision-metadata", "manifest.json", "--model-manifest-sha256", self.digest,
                "--expected-file", f"manifest.json={'0' * 64}", "--", "python3"]
        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            gate.parse_args()

    def test_modified_same_size_weights_are_refused(self):
        weights = self.model / "weights.gguf"
        weights.write_bytes(b"x" * weights.stat().st_size)
        with self.assertRaisesRegex(gate.ModelMountError, "wrong digest"):
            self.verify()

    def test_extra_file_is_refused(self):
        (self.model / "alternate.gguf").write_bytes(b"other weights")
        with self.assertRaisesRegex(gate.ModelMountError, "does not list"):
            self.verify()

    def test_symlink_is_refused(self):
        (self.model / "alias").symlink_to(self.model / "weights.gguf")
        with self.assertRaisesRegex(gate.ModelMountError, "non-regular"):
            self.verify()

    def test_duplicate_and_unsafe_inventory_records_are_refused(self):
        for path in ("weights.gguf", "../weights.gguf", "nested/../weights.gguf", "/weights.gguf", "./weights.gguf"):
            with self.subTest(path=path):
                document = json.loads((self.model / "manifest.json").read_bytes())
                record = dict(document["files"][0], path=path)
                document["files"] = [record, record]
                (self.model / "manifest.json").write_text(json.dumps(document))
                self.args.expected_file = [("manifest.json", gate.sha256_file(self.model / "manifest.json"))]
                with self.assertRaisesRegex(gate.ModelMountError, "invalid file record"):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
