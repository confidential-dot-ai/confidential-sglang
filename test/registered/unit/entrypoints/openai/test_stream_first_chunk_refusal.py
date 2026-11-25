"""How a stream starts (no model): a stream refused before any output is answered
with its HTTP status; one whose first chunk is late answers 200 at once and sends
SSE keep-alive comments until output starts."""

import asyncio
import json
import unittest

from starlette.requests import ClientDisconnect

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import (
    SSE_KEEPALIVE,
    KeepaliveStreamingResponse,
    prefetch_first_chunk,
)
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

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


class _Stream:
    """A fake engine stream: waits ``delay`` before its first chunk."""

    def __init__(self, delay, chunks=("data: a\n\n", "data: [DONE]\n\n"), error=None):
        self.delay, self.chunks, self.error = delay, chunks, error
        self.closed = False
        self.cancelled = False

    async def gen(self):
        try:
            await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            for c in self.chunks:
                yield c
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.closed = True


def _error_chunk(message):
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    return serving.create_streaming_error_response(message)


def _response(first, gen, interval=0.05):
    return KeepaliveStreamingResponse(
        first, gen, interval_s=interval, error_chunk=_error_chunk,
        media_type="text/event-stream",
    )


async def _drain(body):
    return [c async for c in body]


class TestStreamKeepalive(unittest.IsolatedAsyncioTestCase):
    async def test_fast_first_chunk_is_ready(self):
        first = await prefetch_first_chunk(_Stream(0).gen(), 1.0)
        self.assertTrue(first.done())
        self.assertEqual(first.result(), "data: a\n\n")

    async def test_zero_wait_waits_for_the_chunk(self):
        first = await prefetch_first_chunk(_Stream(0.1).gen(), 0)
        self.assertTrue(first.done())

    async def test_validation_error_in_time_is_raised(self):
        first = await prefetch_first_chunk(_Stream(0, error=ValueError("too long")).gen(), 1.0)
        with self.assertRaisesRegex(ValueError, "too long"):
            first.result()

    async def test_late_chunk_streams_keepalives_then_output(self):
        gen = _Stream(0.3).gen()
        first = await prefetch_first_chunk(gen, 0.02)
        self.assertFalse(first.done())
        out = await _drain(_response(first, gen).body_iterator)
        n = out.index("data: a\n\n")
        self.assertGreaterEqual(n, 3)
        self.assertEqual(set(out[:n]), {SSE_KEEPALIVE})
        self.assertEqual(out[n:], ["data: a\n\n", "data: [DONE]\n\n"])

    async def test_late_validation_error_goes_in_stream(self):
        gen = _Stream(0.15, error=ValueError("prompt too long")).gen()
        first = await prefetch_first_chunk(gen, 0.02)
        out = await _drain(_response(first, gen).body_iterator)
        events = [c for c in out if c != SSE_KEEPALIVE]
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        error = json.loads(events[0][len("data: "):])
        self.assertEqual(error["error"]["code"], 400)
        self.assertIn("prompt too long", error["error"]["message"])

    async def test_client_gone_before_output_cancels_the_request(self):
        stream = _Stream(10)
        gen = stream.gen()
        first = await prefetch_first_chunk(gen, 0.02)
        body = _response(first, gen).body_iterator
        self.assertEqual(await body.__anext__(), SSE_KEEPALIVE)
        await body.aclose()
        self.assertTrue(first.cancelled())
        self.assertTrue(stream.cancelled and stream.closed)

    async def test_headers_go_out_before_the_first_chunk(self):
        gen = _Stream(0.3).gen()
        first = await prefetch_first_chunk(gen, 0.02)
        sent = []
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        async def send(message):
            sent.append((loop.time() - t0, message))

        async def receive():
            await asyncio.sleep(3600)

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST",
                 "path": "/v1/chat/completions", "headers": []}
        await _response(first, gen)(scope, receive, send)
        start = [t for t, m in sent if m["type"] == "http.response.start"]
        body = [(t, m.get("body", b"")) for t, m in sent if m["type"] == "http.response.body"]
        self.assertEqual(sent[0][1]["status"], 200)
        first_data = next(t for t, b in body if b.startswith(b"data: a"))
        self.assertLess(start[0], 0.1)
        self.assertGreater(first_data, 0.25)
        self.assertTrue(any(b == SSE_KEEPALIVE.encode() for _, b in body))

    async def test_send_failure_before_output_still_cancels(self):
        stream = _Stream(10)
        gen = stream.gen()
        first = await prefetch_first_chunk(gen, 0.02)

        async def send(message):
            raise OSError("client gone")

        async def receive():
            await asyncio.sleep(3600)

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST",
                 "path": "/v1/completions", "headers": []}
        # Starlette reports a failed send as ClientDisconnect.
        with self.assertRaises(ClientDisconnect):
            await _response(first, gen)(scope, receive, send)
        self.assertTrue(first.cancelled())
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
