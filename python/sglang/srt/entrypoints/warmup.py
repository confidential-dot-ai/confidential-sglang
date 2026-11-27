from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

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


def _warmup_lock_dir() -> str:
    import tempfile

    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()


def _multi_tokenizer_server_key() -> Optional[str]:
    from sglang.srt.runtime_context import get_serving

    if (get_serving().tokenizer_worker_num or 1) <= 1:
        return None
    from sglang.srt.managers.multi_tokenizer_mixin import get_main_process_id

    return str(get_main_process_id())


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
    server_key: Optional[str] = "auto",
) -> bool:
    """confidential-sglang: run the custom warmups once per server.

    With --tokenizer-worker-num N, every tokenizer process runs the lifespan, so
    each ran the whole warmup: N times the requests (dsv41-262k, N=4: +3.5 min
    on a 4x B200). The processes of one server (server_key: the main process
    ID) elect one with flock. It runs the warmups and marks them done; the
    others wait on the lock, so none of them finishes startup (and answers
    /health) before the warmup ends, then skip. A warmup that raises fails its
    process, and the server exits, as before. Returns whether this process ran
    them. server_key "auto": the main process ID with --tokenizer-worker-num > 1,
    else None (one tokenizer process), which runs them directly. This lives
    here, not in http_server.py, whose source hash harden_http.py pins."""
    if server_key == "auto":
        server_key = _multi_tokenizer_server_key()
    if server_key is None:
        await _run_warmups(disaggregation_mode, warmup_names, tokenizer_manager)
        return True
    import asyncio
    import fcntl

    # The PID alone repeats after a container restart, and /dev/shm (a pod
    # emptyDir) can outlive the container: add the process start time.
    try:
        started = open(f"/proc/{server_key}/stat").read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        started = "0"
    base = os.path.join(_warmup_lock_dir(), f"sglang-warmup-{server_key}-{started}")
    fd = os.open(base + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        if os.path.exists(base + ".done"):
            logger.info("Warmups already ran in another tokenizer process of this server")
            return False
        await _run_warmups(disaggregation_mode, warmup_names, tokenizer_manager)
        with open(base + ".done", "w") as marker:
            marker.write(f"{os.getpid()}\n")
        return True
    finally:
        os.close(fd)  # releases the lock


async def _run_warmups(
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
    # top_k and top_p together: DSpark verify then rescales top-p in a
    # torch.compile'd function (dflash_utils._adjust_top_p_for_top_k), which
    # compiles apart for a top-k of 1.
    {"temperature": 0.7, "top_k": 40, "top_p": 0.9},
    {"temperature": 0.7, "top_k": 1, "top_p": 0.9},
    {"temperature": 0.7, "repetition_penalty": 1.1, "frequency_penalty": 0.5, "presence_penalty": 0.5},
    {"temperature": 0.7, "stop": ["\n\n", "END"]},
    {"temperature": 0.0, "json_schema": '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}'},
)
COLD_PATH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
# Ten 8K chunks, past the dsv41 plans' prefill graph cap (65536; every layer eager
# above it) and the 64K prefix where a short extend takes the paged mega prefill,
# then an eager 200-token tail: the 128 M bucket of the FlashInfer MXFP8 GEMM
# tactics, and an M below 1024 that is not a multiple of 16.
LONG_PROMPT_TOKENS = 10 * 8192 + 200
# The longest prompt ends in a 5003-token eager chunk: an M from 1024 to 8191
# that is not a multiple of 16 (Triton integer classes, MHC split counts).
LONG_PROMPT_TAIL = 5003
# Room for the follow-up's extra tokens, 16 new tokens and a DSpark verify block.
LONG_PROMPT_MARGIN = 1024
LONG_PROMPT_FOLLOW_UP = 64
_GRAMMAR_KEYS = ("json_schema", "regex", "ebnf", "structural_tag")
_PENALTY_KEYS = ("repetition_penalty", "frequency_penalty", "presence_penalty", "min_new_tokens")


def cold_path_plan(spec_algorithm, disaggregation_mode: str = "null") -> tuple[tuple[dict, ...], bool]:
    """The cold-path sampling paths and whether to ask for logprobs, minus what
    the scheduler refuses under the server's speculative algorithm. A refused
    request aborts, the warmup raises, and the server never listens: every
    speculative algorithm but DSpark aborts min_p (scheduler.py);
    UNO aborts return_logprob (validate_uno_request). The paths follow the
    scheduler's admission checks; a stripped path still runs its other
    parameters."""
    drop = set()
    if disaggregation_mode == "decode":
        # A PD decode warmup request takes the fake bootstrap: no prefill
        # samples its first token under the grammar, so the grammar refuses it
        # ("Grammar accept_token failed", HTTP 500) and the warmup raises
        # (4x B200 dsv41-262k, 2026-11-23). Real PD requests get that token
        # from prefill, which runs the grammar paths in its own warmup.
        drop.update(_GRAMMAR_KEYS)
    if not spec_algorithm.is_some() and not drop:
        return COLD_PATH_SAMPLING, True
    if spec_algorithm.is_some() and not spec_algorithm.is_dspark():
        drop.add("min_p")
    if spec_algorithm.is_uno():
        drop.update(_PENALTY_KEYS)
    if spec_algorithm.is_uno() or (spec_algorithm.is_dflash_family() and not spec_algorithm.is_dspark()):
        drop.update(_GRAMMAR_KEYS)
    sampling = []
    for path in COLD_PATH_SAMPLING:
        kept = {k: v for k, v in path.items() if k not in drop}
        if (kept or not path) and kept not in sampling:
            sampling.append(kept)
    logprobs = not spec_algorithm.is_uno()
    return tuple(sampling), logprobs


def long_prompt_lengths(max_input_len: Optional[int], chunk: int = 8192) -> tuple[int, ...]:
    """Prompt lengths for the long-context step of the cold path: about 80K
    tokens, and nearly the longest prompt the worker admits, so kernels keyed on
    sequence or page counts, the eager prefill path past the prefill graph cap,
    8K chunks on a near-full prefix and decode at that length all run before the
    server listens. Empty when the worker admits no prompt that long."""
    if not max_input_len:
        return ()
    room = max_input_len - LONG_PROMPT_MARGIN
    longest = (room - LONG_PROMPT_TAIL) // chunk * chunk + LONG_PROMPT_TAIL
    if longest <= LONG_PROMPT_TOKENS:
        return ()
    return (LONG_PROMPT_TOKENS, longest)


@warmup("confidential_cold_path")
async def confidential_cold_path(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Prefill lengths from 1 token to 32K, every sampling path (alone and
    mixed in one batch, with and without logprobs), and decode batch sizes up
    to 64."""
    import asyncio

    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    sampling_paths, with_logprobs = cold_path_plan(spec_algorithm, disaggregation_mode)
    logger.info("confidential_cold_path: %d sampling paths, logprobs %s (speculative %s)",
                len(sampling_paths), "on" if with_logprobs else "off", spec_algorithm.name)

    def request(size: int, sampling: dict, max_new_tokens: int, logprobs: bool = False,
                input_ids: Optional[List[int]] = None) -> GenerateReqInput:
        req = GenerateReqInput(
            input_ids=input_ids if input_ids is not None else np.random.randint(2**16, size=[size]).tolist(),
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
        for sampling in sampling_paths:
            await run(request(size, sampling, 16, with_logprobs and size == 32))
    for sampling in sampling_paths:
        for logprobs in ((False, True) if with_logprobs else (False,)):
            await run(request(64, sampling, 16, logprobs))
    # One batch that mixes every sampling path, like the gateway's real traffic.
    await asyncio.gather(*(run(request(64, s, 16, lp)) for s in sampling_paths for lp in ((False, True) if with_logprobs else (False,))))
    for batch in COLD_PATH_BATCH_SIZES:
        await asyncio.gather(*(run(request(64 + i, sampling_paths[i % len(sampling_paths)], 16))
                               for i in range(batch)))
    # Batches of short prompts, prefilled together: staging-09361fae4780 still compiled
    # Triton kernels on 16 parallel ~30-token chat requests.
    for batch in COLD_PATH_BATCH_SIZES:
        await asyncio.gather(*(run(request(1 + (7 * i) % 63, sampling_paths[i % len(sampling_paths)], 16))
                               for i in range(batch)))
    # Long prompts, then the same prompt plus a few tokens: a short extend on a
    # long cached prefix (the turn after a long one). Alone, so the prefill pool
    # holds the whole prompt.
    max_input_len = tokenizer_manager.max_req_input_len or tokenizer_manager.context_len
    for size in long_prompt_lengths(max_input_len):
        logger.info("confidential_cold_path: %d-token prompt", size)
        ids = np.random.randint(2**16, size=[size]).tolist()
        await run(request(size, {"temperature": 0.0}, 16, input_ids=ids))
        follow_up = ids + np.random.randint(2**16, size=[LONG_PROMPT_FOLLOW_UP]).tolist()
        await run(request(len(follow_up), {"temperature": 0.7, "top_k": 40}, 16, input_ids=follow_up))
    logger.info("confidential_cold_path warmup ended")
