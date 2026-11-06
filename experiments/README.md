# Unqualified reproducers

`q4_readonly_api.mojo` preserves an AOT code-generation failure observed with the pinned SDK. Its emitted PTX failed NVIDIA assembler validation. It is not a runnable or performance-qualified kernel. See [the investigation](../docs/q4-readonly.md) for exact scope and reproduction.
