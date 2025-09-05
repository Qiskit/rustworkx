"""sparse6 format helpers (placeholder).

The sparse6 format is related to graph6/digraph6 but optimized for sparse
graphs. Parsing is currently not implemented in the Rust core; the Rust
layer returns an UnsupportedFormat error when an explicit sparse6 header is
encountered.

This module centralizes the placeholder so future implementation can add
real parsing while giving users a discoverable namespace today.
"""
from __future__ import annotations

from typing import NoReturn

__all__ = ["read_sparse6_str"]


class Sparse6Unsupported(RuntimeError):
    pass


def read_sparse6_str(repr: str) -> NoReturn:
    """Attempt to read a sparse6 string (always unsupported for now)."""
    raise Sparse6Unsupported(
        "sparse6 parsing not yet implemented; contributions welcome."
    )
