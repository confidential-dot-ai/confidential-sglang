"""CPU-only checks: python3 -m unittest discover -s recipes/deepseek-v41/launcher."""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_checkpoint", Path(__file__).with_name("prepare_checkpoint.py")
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

BASE = prep.Source("org/nvfp4", "a" * 40)
ENGRAM = prep.Source("org/official", "b" * 40)


def remote(source, path, data: bytes, lfs: bool = True):
    return prep.RemoteFile(
        source=source,
        path=path,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest() if lfs else None,
        blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    )


def index(weight_map):
    return {"metadata": {"total_size": 0}, "weight_map": weight_map}


def safetensors(tensors) -> bytes:
    header, offset = {}, 0
    for name, (dtype, nbytes) in tensors.items():
        header[name] = dict(dtype=dtype, shape=[nbytes], data_offsets=[offset, offset + nbytes])
        offset += nbytes
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * offset


class TestPlan(unittest.TestCase):
    def test_engram_shards_come_from_the_engram_source(self):
        weights = {"layers.0.ffn.w1": "s1", "layers.1.engram.embed.weight": "s2"}
        files = prep.plan_files(
            base_files=[
                remote(BASE, "s1", b"base"),
                remote(BASE, "s2", b"mxfp4"),
                remote(BASE, prep.CONFIG, b"{}", lfs=False),
                remote(BASE, prep.INDEX, b"{}", lfs=False),
            ],
            engram_files=[remote(ENGRAM, "s1", b"fp8 experts"), remote(ENGRAM, "s2", b"fp8")],
            base_index=index(weights),
            engram_index=index(weights),
        )
        self.assertEqual(sorted(files), ["s1", "s2"])
        self.assertEqual(files["s1"].source, BASE)
        self.assertEqual(files["s2"].source, ENGRAM)

    def test_rejects_an_engram_shard_holding_other_tensors(self):
        with self.assertRaisesRegex(ValueError, "also hold"):
            prep.engram_shards({"layers.1.engram.q_weight": "s2", "layers.1.attn.wq": "s2"})

    def test_rejects_sources_with_different_engram_tensors(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            prep.plan_files(
                base_files=[],
                engram_files=[],
                base_index=index({"layers.1.engram.embed.weight": "s2"}),
                engram_index=index({"layers.1.engram.embed.scale": "s2"}),
            )

    MXFP4_FIELDS = {
        "engram_embed_quant": "mxfp4",
        "engram_embed_block_size": 32,
        "engram_embed_scale_dtype": "ue8m0",
        "engram_embed_layers": [1, 14],
    }

    def test_config_drops_the_base_engram_fields_for_fp8_tables(self):
        # 1017-nvfp4-fp8-engram: an MXFP4-table base, FP8 tables from the official repo.
        base = {"a": 1, "quantization_config": {"quant_method": "fp8", **self.MXFP4_FIELDS}}
        official = {"quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"}}
        out = prep.derived_config(base, official)
        self.assertEqual(out, {"a": 1, "quantization_config": {"quant_method": "fp8"}})
        self.assertIn("engram_embed_quant", base["quantization_config"])

    def test_config_takes_the_engram_fields_of_mxfp4_tables(self):
        # 1017-mxfp4-engram: the official base, MXFP4 tables from NVIDIA's repo.
        base = {"a": 1, "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"}}
        nvidia = {"quantization_config": {"quant_method": "fp8", "moe_quant_algo": "NVFP4", **self.MXFP4_FIELDS}}
        out = prep.derived_config(base, nvidia)
        self.assertEqual(
            out,
            {"a": 1, "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4", **self.MXFP4_FIELDS}},
        )
        self.assertNotIn("engram_embed_quant", base["quantization_config"])

    def test_engram_table_dtype_must_match_the_config(self):
        def header(dtype):
            return {"s47": {"layers.1.engram.embed.weight": {"dtype": dtype},
                            "layers.1.engram.embed.scale": {"dtype": "F8_E8M0"}},
                    "s48": {"layers.14.engram.embed.weight": {"dtype": dtype}}}
        fp8 = {"quantization_config": {"quant_method": "fp8"}}
        mxfp4 = {"quantization_config": {"engram_embed_quant": "mxfp4"}}
        prep.check_engram_tables(fp8, header("F8_E4M3"))
        prep.check_engram_tables(mxfp4, header("U8"))
        with self.assertRaisesRegex(ValueError, "declares mxfp4"):
            prep.check_engram_tables(mxfp4, header("F8_E4M3"))
        with self.assertRaisesRegex(ValueError, "declares fp8"):
            prep.check_engram_tables(fp8, header("U8"))
        with self.assertRaisesRegex(ValueError, "no engram.embed.weight"):
            prep.check_engram_tables(fp8, {"s47": {"layers.1.engram.q_weight": {"dtype": "BF16"}}})

    def test_1017_mxfp4_engram_pins_the_official_weights_and_nvidia_tables(self):
        spec_ = prep.SPECS["1017-mxfp4-engram"]
        self.assertEqual(
            spec_.base,
            ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"),
        )
        self.assertEqual(
            spec_.engram,
            ("nvidia/DeepSeek-V4.1-Flash-1017-NVFP4", "3ae9c0d3fbbea5370bd06e679d22044db15883f1"),
        )
        for source in prep.SPECS.values():
            for repo, revision in source:
                self.assertRegex(revision, "^[0-9a-f]{40}$", repo)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.local, self.seed = self.tmp / "out", self.tmp / "seed"
        self.local.mkdir()
        self.seed.mkdir()
        self.downloads = []

    def download(self, payloads):
        def fetch(remote_file, local_dir):
            self.downloads.append(remote_file.path)
            (local_dir / remote_file.path).write_bytes(payloads[remote_file.path])

        return fetch

    def test_seeds_by_hard_link_and_downloads_the_rest(self):
        (self.seed / "s1").write_bytes(b"base")
        (self.seed / "s2").write_bytes(b"mxfp4")
        payloads = {"s1": b"base", "s2": b"fp8!"}
        outcomes = {
            path: prep.ensure_file(
                remote(BASE if path == "s1" else ENGRAM, path, data),
                local_dir=self.local,
                seed_dir=self.seed,
                download=self.download(payloads),
            )
            for path, data in payloads.items()
        }
        self.assertEqual(outcomes, {"s1": "seeded", "s2": "downloaded"})
        self.assertEqual(self.downloads, ["s2"])
        self.assertEqual((self.local / "s1").stat().st_ino, (self.seed / "s1").stat().st_ino)

    def test_replaces_a_local_file_whose_hash_does_not_match(self):
        (self.local / "s2").write_bytes(b"mxfp")  # same size, wrong bytes
        outcome = prep.ensure_file(
            remote(ENGRAM, "s2", b"fp8!"),
            local_dir=self.local,
            seed_dir=None,
            download=self.download({"s2": b"fp8!"}),
        )
        self.assertEqual(outcome, "downloaded")
        self.assertEqual((self.local / "s2").read_bytes(), b"fp8!")

    def test_small_files_are_checked_by_git_blob_id(self):
        (self.local / "README.md").write_bytes(b"hello")
        self.assertTrue(prep.file_matches(self.local / "README.md", remote(BASE, "README.md", b"hello", lfs=False)))
        self.assertFalse(prep.file_matches(self.local / "README.md", remote(BASE, "README.md", b"hellO", lfs=False)))

    def test_index_total_size_is_recounted_from_local_shards(self):
        (self.local / "s1").write_bytes(safetensors({"w": ("F8_E4M3", 10), "s": ("F32", 4)}))
        (self.local / "s2").write_bytes(safetensors({"e": ("F8_E4M3", 6)}))
        out = prep.derived_index(index({"w": "s1", "s": "s1", "e": "s2"}), self.local)
        self.assertEqual(out["metadata"]["total_size"], 20)

    def test_manifest_fast_path_notices_a_changed_file(self):
        name, spec_ = "1017-nvfp4-fp8-engram", prep.SPECS["1017-nvfp4-fp8-engram"]
        (self.local / "s1").write_bytes(b"base")
        stat = (self.local / "s1").stat()
        manifest = dict(
            spec=name,
            status="complete",
            sources={"base": list(spec_.base), "engram": list(spec_.engram)},
            files={"s1": dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)},
        )
        (self.local / prep.MANIFEST).write_text(json.dumps(manifest))
        self.assertTrue(prep.manifest_is_current(self.local, name, spec_))
        (self.local / "s1").write_bytes(b"BASE!")
        self.assertFalse(prep.manifest_is_current(self.local, name, spec_))


if __name__ == "__main__":
    unittest.main()
