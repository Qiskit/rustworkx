"""graph6 format helpers.

This module provides a namespace for working with undirected graph6 strings
as described in: https://users.cecs.anu.edu.au/~bdm/data/formats.txt

It wraps the low-level functions exported from the compiled extension
(`read_graph6_str`, `write_graph6_from_pygraph`) and offers convenience
helpers. Backwards compatibility: existing top-level functions in
`rustworkx` remain valid; this is a thin façade only.
"""
from __future__ import annotations

from . import read_graph6_str as _read_graph6_str
from . import write_graph6_from_pygraph as _write_graph6_from_pygraph

__all__ = [
    "read_graph6_str",
    "write_graph6_from_pygraph",
    "read",
    "write",
]


def read_graph6_str(repr: str):
    """Parse a graph6 representation into a PyGraph.

    Accepts either raw graph6, header form (>>graph6<<:), or directed strings.
    For clarity, use digraph6.read_graph6_str for directed graphs. This wrapper
    leaves behavior unchanged (delegates to the core function) but documents
    intent that this namespace targets undirected graphs.
    """
    g = _read_graph6_str(repr)
    return g

# Short aliases
read = read_graph6_str


def write_graph6_from_pygraph(graph):
    """Serialize a PyGraph to a graph6 string."""
    return _write_graph6_from_pygraph(graph)

write = write_graph6_from_pygraph
