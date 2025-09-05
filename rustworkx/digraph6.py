"""digraph6 format helpers.

Directed variant of graph6 per the documented format. The core dispatch
routine already auto-detects directed strings (leading '&') or header form
(>>digraph6<<:). This namespace provides clarity and future room for
specialized helpers without breaking existing API.
"""
from __future__ import annotations

from . import read_graph6_str as _read_graph6_str
from . import write_graph6_from_pydigraph as _write_graph6_from_pydigraph

__all__ = [
    "read_graph6_str",
    "write_graph6_from_pydigraph",
    "read",
    "write",
]


def read_graph6_str(repr: str):  # noqa: D401 - thin wrapper
    return _read_graph6_str(repr)

read = read_graph6_str


def write_graph6_from_pydigraph(digraph):  # noqa: D401 - thin wrapper
    return _write_graph6_from_pydigraph(digraph)

write = write_graph6_from_pydigraph
