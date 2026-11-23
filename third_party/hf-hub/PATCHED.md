# hf-hub 0.5.0 (patched)

Unmodified copy of the crates.io release except `src/api/sync.rs` `builder()`
(native-tls): `.root_certs(RootCerts::PlatformVerifier)`, so downloads use the
OS certificate store. Used via `[patch.crates-io]` in the workspace
`Cargo.toml`. Drop this directory when hf-hub exposes the TLS roots setting.
