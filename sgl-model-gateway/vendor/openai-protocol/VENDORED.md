# openai-protocol 1.0.0 (vendored)

Source: https://crates.io/crates/openai-protocol/1.0.0 (Apache-2.0,
https://github.com/lightseekorg/smg). Used through `[patch.crates-io]` in
`sgl-model-gateway/Cargo.toml`.

One change: `ChatCompletionRequest.forced_text: Option<String>`
(`src/chat.rs`), so the router forwards Oracle-decoding requests to the
workers. Drop this copy when the upstream crate carries the field or a
catch-all for unknown fields.
