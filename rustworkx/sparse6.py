"""sparse6 format helpers (placeholder).

The sparse6 format is related to graph6/digraph6 but optimized for sparse
graphs. Parsing is currently not implemented in the Rust core; the Rust
layer returns an UnsupportedFormat error when an explicit sparse6 header is
encountered.

This module centralizes the placeholder so future implementation can add
real parsing while giving users a discoverable namespace today.
"""
from __future__ import annotations

from . import read_sparse6_str as _read_sparse6_str
from . import write_sparse6_from_pygraph as _write_sparse6_from_pygraph

__all__ = ["read_sparse6_str", "write_sparse6_from_pygraph"]


def read_sparse6_str(repr: str):
    return _read_sparse6_str(repr)


def write_sparse6_from_pygraph(pygraph, header: bool = True):
    return _write_sparse6_from_pygraph(pygraph, header)
