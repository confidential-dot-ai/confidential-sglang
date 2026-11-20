# Router changes (for docs/confidential-changes.md)

Compared with confidential-inference's `images/sglang-router` (upstream
a7fd2e1 + 7 patches), the fork's `main` already had worker-withdrawal,
admission-caps and pd-router (the DeepSeek-V4.1 stack's own router commits).
It lacked four:

- `http-no-tokenizer`: ported. An HTTP worker gets no router tokenizer, so the
  router (no egress) never tries to fetch the worker's model from the Hub.
- `client-errors` (public #328): ported with its tests. A worker's 4xx (not
  408/429) does not count against the regular router's circuit breaker, so a
  few bad requests cannot eject every worker.
- `private-errors`: ported. PD request errors log fixed `ci_event` names,
  roles and status numbers; no error bodies, body previews, URLs or error
  text. The fork's extra decode-gate preparation path got the same treatment.
  `confidential/tests/router/test_recipe.py` checks that every `ci_event` in
  the Rust source is one `private_router.py` forwards.
- `pd-pairs`: replaced. The public `CI_ROUTER_PD_PAIRS` env (exactly two
  fixed URL pairs) is gone. In PD mode with `--service-discovery`, a pod with
  the `sglang.ai/prefill-port` and `sglang.ai/decode-port` annotations is one
  pair: discovery registers both workers on the pod IP with the label
  `pd_pair=<pod name>`, and the PD router never crosses pairs (a paired
  prefill is selectable only while its own decode is available; decode is
  always the prefill's peer, also on the decode-gate path). Half or invalid
  annotations refuse the pod. Pods without them, and regular-mode
  `--service-discovery --selector`, behave as upstream. Unit tests in
  `pd_router.rs` (`pair_selection_tests`) and `service_discovery.rs`.

Build inputs:

- `sgl-model-gateway/Cargo.lock` is now committed (upstream ignores it):
  seeded from the public image's lock, plus the crates this fork added
  (`notify`), with `notify-types` held at 2.2.0 because 2.3.x needs Rust 1.91
  and the toolchain is 1.90 (`rust-toolchain.toml`).
- `confidential/router/`: the image (Rust 1.90.0 build image and Python
  runtime by digest, Debian snapshot apt, maturin by hash, `--locked`,
  canonicalized wheel, hashed `requirements.lock`, no bytecode, uid 1001,
  `ci-sglang-router` entrypoint), its `check-image.sh`, and
  `run_router_tests.sh` (Rust lib + reliability tests in the build image with
  protoc and redis-server from the same snapshot).
- Moved from the public repo: `private_router.py`, `canonicalize_wheel.py`,
  `requirements.{in,lock}` (unchanged), and
  `tests/images/sglang-router/test_private_router.py` (now
  `confidential/tests/router/`). Not moved: `smoke_alias.py` (drives
  `CI_ROUTER_PD_PAIRS`, which no longer exists) and the public `test_recipe.py`
  (checks patch files, replaced by the new `test_recipe.py`).
