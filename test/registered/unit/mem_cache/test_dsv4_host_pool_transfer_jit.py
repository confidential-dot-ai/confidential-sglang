import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import memory_pool_host as host_module
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4HostPoolBuildsTransferAtInit(unittest.TestCase):
    def test_token_granular_pool_builds_its_copy_kernel_before_serving(self):
        """The token-granular copy is a JIT module; built on its first call, the
        first HiCache eviction would compile it after the server is ready."""
        for page_aligned_only, builds in ((False, 1), (True, 0)):
            prepare = MagicMock()
            with (
                self.subTest(page_aligned_only=page_aligned_only),
                patch.object(host_module, "_is_cuda", True),
                patch.object(host_module, "prepare_transfer_cache_dsv4_mla", prepare),
            ):
                DeepSeekV4PagedHostPool(
                    pool_name="c1",
                    device_buffers=[torch.zeros((4, 64), dtype=torch.uint8)],
                    item_bytes=64,
                    num_host_pages=4,
                    slot_page_size=256,
                    pin_memory=False,
                    page_aligned_only=page_aligned_only,
                )
            self.assertEqual(prepare.call_count, builds)


if __name__ == "__main__":
    unittest.main()
