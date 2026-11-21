from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


# confidential-sglang: run every kernel path a request can take before the
# server listens, so no compiler (Triton, SGLang JIT, inductor) writes a
# kernel after the worker is ready. Staging staging-39086210fdc6 (Qwen3.5-0.8B,
# B200) compiled 20 times after ready, in bursts of up to 77 files; each burst
# stalled the worker, the router's /health_generate check timed out, and
# clients got 503. Custom warmups run in the server lifespan, before it
# listens, so the probes and the router see the worker only afterwards. The
# shapes follow confidential-inference-internal acceptance.cold_path_shapes.
COLD_PATH_SAMPLING = (
    {"temperature": 0.0},                                    # greedy
    {},                                                      # the model's defaults
    {"temperature": 0.7, "top_k": 40},
    {"temperature": 0.7, "top_p": 0.9, "min_p": 0.05},
    {"temperature": 0.7, "repetition_penalty": 1.1, "frequency_penalty": 0.5, "presence_penalty": 0.5},
    {"temperature": 0.7, "stop": ["\n\n", "END"]},
    {"temperature": 0.0, "json_schema": '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}'},
)
COLD_PATH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


@warmup("confidential_cold_path")
async def confidential_cold_path(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Prefill lengths from 1 token to 32K, every sampling path (alone and
    mixed in one batch, with and without logprobs), and decode batch sizes up
    to 64."""
    import asyncio

    def request(size: int, sampling: dict, max_new_tokens: int, logprobs: bool = False) -> GenerateReqInput:
        req = GenerateReqInput(
            input_ids=np.random.randint(2**16, size=[size]).tolist(),
            sampling_params={"max_new_tokens": max_new_tokens, **sampling},
            return_logprob=logprobs,
            top_logprobs_num=2 if logprobs else 0,
        )
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        return req

    async def run(req: GenerateReqInput) -> None:
        async for _ in tokenizer_manager.generate_request(req, None):
            pass

    await prefill_shapes(disaggregation_mode, tokenizer_manager)
    # Short prompts, which prefill_shapes (64 and up) does not reach. Below 64
    # tokens some Triton kernels compile other variants: staging-d0d1b05e01e8
    # compiled 32 Triton files on the first /health_generate (a 1-token
    # prompt, input_ids=[0]) and 8 on ~30-token chat prompts. Every length
    # from 1 to 64, greedy and one token, is cheap.
    for size in range(1, 65):
        await run(request(size, {"temperature": 0.0}, 1))
    for size in (1, 8, 16, 32):
        for sampling in COLD_PATH_SAMPLING:
            await run(request(size, sampling, 16, size == 32))
    for sampling in COLD_PATH_SAMPLING:
        for logprobs in (False, True):
            await run(request(64, sampling, 16, logprobs))
    # One batch that mixes every sampling path, like the gateway's real traffic.
    await asyncio.gather(*(run(request(64, s, 16, lp)) for s in COLD_PATH_SAMPLING for lp in (False, True)))
    for batch in COLD_PATH_BATCH_SIZES:
        await asyncio.gather(*(run(request(64 + i, COLD_PATH_SAMPLING[i % len(COLD_PATH_SAMPLING)], 16))
                               for i in range(batch)))
    logger.info("confidential_cold_path warmup ended")
