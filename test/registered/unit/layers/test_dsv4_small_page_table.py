import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


class TestSmallPageTable(unittest.TestCase):
    @unittest.skipIf(_INTERPRET, "the interpreter has no specialization key")
    def test_page_count_is_not_part_of_the_compile_key(self):
        """An eager decode or verify step passes the batch's live page count;
        a page count in the key would compile a new kernel at every length after
        the worker was ready (up to 4096 variants at 1M context)."""
        from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
            _small_page_table,
        )

        params = {p.name: p for p in _small_page_table.params}
        self.assertFalse(params["NUM_PAGES"].is_constexpr)
        self.assertTrue(params["NUM_PAGES"].do_not_specialize)

    @unittest.skipUnless(torch.cuda.is_available() or _INTERPRET, "needs CUDA or TRITON_INTERPRET=1")
    def test_pages_positions_and_window(self):
        from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
            build_page_table_positions_small,
        )

        dev = "cuda" if torch.cuda.is_available() and not _INTERPRET else "cpu"
        page, slots = 256, 4096
        req_to_token = (torch.arange(4 * slots, dtype=torch.int32, device=dev) * 3).reshape(4, slots)
        reqs = torch.tensor([2, 0, 3], device=dev)
        lens = torch.tensor([1000, 300, 1], dtype=torch.int32, device=dev)
        out = build_page_table_positions_small(
            req_to_token=req_to_token,
            req_pool_indices_repeated=reqs,
            seq_lens_casual=lens,
            max_seq_len=1000,
            page_size=page,
            swa_window=128,
        )
        pages = torch.arange(4, device=dev) * page
        expected = req_to_token[reqs][:, pages] // page
        self.assertTrue(torch.equal(out.page_table, expected))
        self.assertEqual(out.positions_casual.tolist(), [999, 299, 0])
        self.assertEqual(out.swa_topk_lengths.tolist(), [128, 128, 1])
        self.assertEqual(out.seq_lens_casual.tolist(), [1000, 300, 1])


if __name__ == "__main__":
    unittest.main()
