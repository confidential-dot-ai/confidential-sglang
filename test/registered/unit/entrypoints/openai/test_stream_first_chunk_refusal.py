"""A stream refused before any output is answered with its HTTP status (no model)."""

import json
import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DSPARK_MIN_P = ('data: {"error": {"object": "error", "message": "min_p is not supported with DSPARK '
                'speculative decoding; disable min_p or speculative decoding.", "type": "BAD_REQUEST", '
                '"param": null, "code": 400}}\n\n')


class TestStreamFirstChunkRefusal(unittest.TestCase):
    def test_error_event_first_becomes_http_error(self):
        # The helper is on the shared base; chat's error body also needs its
        # wire state, which only a real server sets up (checked live on GPU).
        serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
        response = serving.error_response_for_first_chunk(DSPARK_MIN_P)
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["code"], 400)
        self.assertIn("min_p is not supported with DSPARK", body["message"])

    def test_other_first_chunks_stream(self):
        serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
        for chunk in ('data: {"id": "x", "choices": [{"delta": {"role": "assistant"}}]}\n\n',
                      'data: {"error": "not json-shaped\n\n', 'data: {"error": {"code": 200}}\n\n',
                      b'data: {"error": {"code": 400}}', None):
            self.assertIsNone(serving.error_response_for_first_chunk(chunk))


if __name__ == "__main__":
    unittest.main()
