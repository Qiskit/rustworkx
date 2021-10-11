# retworkx-lib

[![License](https://img.shields.io/github/license/Qiskit/retworkx.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![Minimum rustc 1.41.1](https://img.shields.io/badge/rustc-1.41.1+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

This crate contains the rust library retworkx-lib. This library is part of the
retworkx project. However while the larger retworkx project is a Python library
that offers a general purpose high performance graph library written in Rust,
retworkx-lib is a pure rust library that offers a stable rust API for any
downstream crate that need it.

At it's core retworkx is built on top of the
[petgraph](https://github.com/petgraph/petgraph) library and wraps it in a
Python layer. However, many of the algorithms (and to a lesser extent data
structures) needed for retworkx are not available in petgraph. For places
where these algorithms are implemented in a generic way the retworkx-lib
crate exposes it for rust users.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
retworkx-lib = "0.11"
```

Then:

```rust
use retworkx_lib;
```

Note there is a strong version dependency between `petgraph` and `retworkx-lib`
as the functionality exposed by retworkx-lib is built on top of petgraph. For
convenience we re-export `petgraph` in the root of this crate so you can use
petgraph without explicitly needing it at the same exact version in your crate.

## License

Just as with the rest of the retworkx project retworkx-lib is licensed under
the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
