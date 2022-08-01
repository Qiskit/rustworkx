# rustworkx-core

[![License](https://img.shields.io/github/license/Qiskit/rustworkx.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![Minimum rustc 1.41.1](https://img.shields.io/badge/rustc-1.41.1+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

> :warning: The retworkx-core project has been renamed to **rustworkx-core**.
> If you're using retworkx-core 0.11.0 you will need to change your requirement
> and use statements to use the new crate name

This crate contains the rust library rustworkx-core. This library is part of the
rustworkx project. However while the larger rustworkx project is a Python library
that offers a general purpose high performance graph library written in Rust,
rustworkx-core is a pure rust library that offers a stable rust API for any
downstream crate that need it.

At it's core rustworkx is built on top of the
[petgraph](https://github.com/petgraph/petgraph) library and wraps it in a
Python layer. However, many of the algorithms (and to a lesser extent data
structures) needed for rustworkx are not available in petgraph. For places
where these algorithms are implemented in a generic way the rustworkx-core
crate exposes it for rust users.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustworkx-core = "0.11"
```

Then:

```rust
use rustworkx_core;
```

Note there is a strong version dependency between `petgraph` and `rustworkx-core`
as the functionality exposed by rustworkx-core is built on top of petgraph. For
convenience we re-export `petgraph` in the root of this crate so you can use
petgraph without explicitly needing it at the same exact version in your crate.

## License

Just as with the rest of the rustworkx project rustworkx-core is licensed under
the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
