//! Combined module: conversion, error, utils, write, undirected and directed
//! This file is intended as a drop-in single-module alternative to
//! the separate files in `src/` so callers can `mod all; use all::...` and
//! avoid many `use super` / `use crate` imports inside the library.

/// Conversion trait for graphs into various text graph formats
pub trait GraphConversion {
    /// Returns the bitvector representation of the graph
    fn bit_vec(&self) -> &[usize];

    /// Returns the number of vertices in the graph
    fn size(&self) -> usize;

    /// Returns true if the graph is directed
    fn is_directed(&self) -> bool;

    /// Returns the graph in the DOT format
    fn to_dot(&self, id: Option<usize>) -> String {
        let n = self.size();
        let bit_vec = self.bit_vec();

        let mut dot = String::new();

        // include graph type
        if self.is_directed() {
            dot.push_str("digraph ");
        } else {
            dot.push_str("graph ");
        }

        // include graph id
        if let Some(id) = id {
            dot.push_str(&format!("graph_{} {{", id));
        } else {
            dot.push('{');
        }

        // include edges
        if self.is_directed() {
            self.to_directed_dot(&mut dot, bit_vec, n);
        } else {
            self.to_undirected_dot(&mut dot, bit_vec, n);
        }

        // close graph
        dot.push_str("\n}");

        dot
    }

    fn to_undirected_dot(&self, dot: &mut String, bit_vec: &[usize], n: usize) {
        for i in 0..n {
            for j in i..n {
                if bit_vec[i * n + j] == 1 {
                    dot.push_str(&format!("\n{} -- {};", i, j));
                }
            }
        }
    }

    fn to_directed_dot(&self, dot: &mut String, bit_vec: &[usize], n: usize) {
        for i in 0..n {
            for j in 0..n {
                if bit_vec[i * n + j] == 1 {
                    dot.push_str(&format!("\n{} -> {};", i, j));
                }
            }
        }
    }

    /// Returns the graph as an adjacency matrix
    fn to_adjmat(&self) -> String {
        let n = self.size();
        let bit_vec = self.bit_vec();

        let mut adj = String::new();
        for i in 0..n {
            for j in 0..n {
                adj.push_str(&format!("{}", bit_vec[i * n + j]));
                if j < n - 1 {
                    adj.push(' ');
                }
            }
            adj.push('\n');
        }
        adj
    }

    /// Returns the graph in a flat adjacency matrix
    fn to_flat(&self) -> String {
        let n = self.size();
        let bit_vec = self.bit_vec();

        let mut flat = String::new();
        for i in 0..n {
            for j in 0..n {
                flat.push_str(&format!("{}", bit_vec[i * n + j]));
            }
        }
        flat
    }

    /// Returns the graph in the Pajek NET format
    fn to_net(&self) -> String {
        let n = self.size();
        let bit_vec = self.bit_vec();

        let mut net = String::new();
        net.push_str(&format!("*Vertices {}\n", n));
        for i in 0..n {
            net.push_str(&format!("{} \"{}\"\n", i + 1, i));
        }
        net.push_str("*Arcs\n");
        for i in 0..n {
            for j in 0..n {
                if bit_vec[i * n + j] == 1 {
                    net.push_str(&format!("{} {}\n", i + 1, j + 1));
                }
            }
        }
        net
    }
}

/// IO / parsing errors
#[derive(Debug, PartialEq, Eq)]
pub enum IOError {
    InvalidDigraphHeader,
    InvalidSizeChar,
    GraphTooLarge,
    InvalidAdjacencyMatrix,
    NonCanonicalEncoding,
}

// ---------------------------------------------------------------------------
// Shared size (N(n)) encoding/decoding and bit-width helper used by graph6,
// digraph6, and sparse6. Centralizing here avoids divergence in canonical
// encoding rules and bound checks. The formats share identical size rules.
// ---------------------------------------------------------------------------

/// Trait encapsulating graph size field (N(n)) codec.
pub trait SizeCodec {
    /// Encode a vertex count `n` into its canonical representation as 63-offset bytes.
    fn encode_size(n: usize) -> Result<Vec<u8>, IOError>;
    /// Decode size field at position `pos`, returning (n, bytes_consumed).
    fn decode_size(bytes: &[u8], pos: usize) -> Result<(usize, usize), IOError>;
    /// Compute number of bits needed to represent integers in [0, n-1]. (R(x) in spec)
    fn needed_bits(n: usize) -> usize {
        if n <= 1 { 0 } else { (usize::BITS - (n - 1).leading_zeros()) as usize }
    }
}

/// Concrete codec implementation shared across formats.
pub struct GraphNumberCodec;

impl GraphNumberCodec {
    #[inline]
    fn validate(n: usize) -> Result<(), IOError> {
        if n >= (1usize << 36) { return Err(IOError::GraphTooLarge); }
        Ok(())
    }
}

impl SizeCodec for GraphNumberCodec {
    fn encode_size(n: usize) -> Result<Vec<u8>, IOError> {
        Self::validate(n)?;
        let mut out = Vec::with_capacity(8);
        if n < 63 {
            out.push((n as u8) + 63);
        } else if n < (1 << 18) {
            out.push(b'~');
            let mut v = n as u32;
            let mut parts = [0u8; 3];
            for i in (0..3).rev() { parts[i] = (v & 0x3F) as u8; v >>= 6; }
            out.extend(parts.iter().map(|p| p + 63));
        } else {
            out.push(b'~'); out.push(b'~');
            let mut v = n as u64;
            let mut parts = [0u8; 6];
            for i in (0..6).rev() { parts[i] = (v & 0x3F) as u8; v >>= 6; }
            out.extend(parts.iter().map(|p| p + 63));
        }
        Ok(out)
    }

    fn decode_size(bytes: &[u8], pos: usize) -> Result<(usize, usize), IOError> {
        let first = *bytes.get(pos).ok_or(IOError::InvalidSizeChar)?;
        if first == b'~' {
            let second = *bytes.get(pos + 1).ok_or(IOError::InvalidSizeChar)?;
            if second == b'~' {
                // long form: '~~' + 6 chars
                let mut val: u64 = 0;
                for i in 0..6 {
                    let c = *bytes.get(pos + 2 + i).ok_or(IOError::InvalidSizeChar)?;
                    if c < 63 { return Err(IOError::InvalidSizeChar); }
                    val = (val << 6) | ((c - 63) as u64);
                }
                if val >= (1u64 << 36) { return Err(IOError::GraphTooLarge); }
                if val < (1 << 18) { return Err(IOError::NonCanonicalEncoding); }
                Ok((val as usize, 8))
            } else {
                // medium form: '~' + 3 chars
                let mut val: u32 = 0;
                for i in 0..3 {
                    let c = *bytes.get(pos + 1 + i).ok_or(IOError::InvalidSizeChar)?;
                    if c < 63 { return Err(IOError::InvalidSizeChar); }
                    val = (val << 6) | ((c - 63) as u32);
                }
                if val < 63 { return Err(IOError::NonCanonicalEncoding); }
                Ok((val as usize, 4))
            }
        } else {
            if first < 63 { return Err(IOError::InvalidSizeChar); }
            let n = (first - 63) as usize;
            if n >= 63 { return Err(IOError::NonCanonicalEncoding); }
            Ok((n, 1))
        }
    }
}

impl From<IOError> for PyErr {
    fn from(e: IOError) -> PyErr {
        match e {
            IOError::InvalidDigraphHeader => Graph6ParseError::new_err("Invalid digraph header"),
            IOError::InvalidSizeChar => {
                Graph6ParseError::new_err("Invalid size character in header")
            }
            IOError::GraphTooLarge => {
                Graph6OverflowError::new_err("Graph too large for graph6 encoding")
            }
            IOError::InvalidAdjacencyMatrix => {
                Graph6ParseError::new_err("Invalid adjacency matrix")
            }
            IOError::NonCanonicalEncoding => {
                Graph6ParseError::new_err("Non-canonical graph6 encoding")
            }
        }
    }
}

/// Utility functions used by parsers and writers
pub mod utils {
    use super::IOError;
    use super::{GraphNumberCodec, SizeCodec};

    /// Iterates through the bytes of a graph and fills a bitvector representing
    /// the adjacency matrix of the graph
    pub fn fill_bitvector(bytes: &[u8], size: usize, offset: usize) -> Option<Vec<usize>> {
        let mut bit_vec = Vec::with_capacity(size);
        let mut pos = 0;
        for b in bytes.iter().skip(offset) {
            let b = b.checked_sub(63)?;
            for i in 0..6 {
                let bit = (b >> (5 - i)) & 1;
                bit_vec.push(bit as usize);
                pos += 1;
                if pos == size {
                    break;
                }
            }
        }
        Some(bit_vec)
    }

    /// Parse the size field (n) from a graph6/digraph6 string starting at `pos`.
    /// Returns (n, bytes_consumed_for_size_field).
    /// Supports the standard forms:
    ///  - single char: n < 63, encoded as n + 63
    ///  - '~' + 3 chars: 63 <= n < 2^18 (except values whose top 6 bits are all 1, to avoid ambiguity with long form)
    ///  - '~~' + 6 chars: remaining values up to < 2^36
    pub fn parse_size(bytes: &[u8], pos: usize) -> Result<(usize, usize), IOError> {
        GraphNumberCodec::decode_size(bytes, pos)
    }

    /// Returns the upper triangle of a bitvector
    pub fn upper_triangle(bit_vec: &[usize], n: usize) -> Vec<usize> {
        let mut tri = Vec::with_capacity(n * (n - 1) / 2);
        for i in 1..n {
            for j in 0..i {
                let idx = i * n + j;
                tri.push(bit_vec[idx])
            }
        }
        tri
    }
}

/// Graph6 writer utilities
pub mod write {
    use super::utils::upper_triangle;
    use super::GraphConversion;
    use super::IOError;
    use super::{GraphNumberCodec, SizeCodec};

    /// Trait to write graphs into graph 6 formatted strings
    pub trait WriteGraph: GraphConversion {
        fn write_graph(&self) -> Result<String, IOError> {
            to_file(self.bit_vec().to_vec(), self.size(), self.is_directed())
        }
    }

    fn write_header(repr: &mut String, is_directed: bool) {
        if is_directed {
            repr.push('&');
        }
    }

    fn write_size(repr: &mut String, size: usize) -> Result<(), IOError> {
        let enc = GraphNumberCodec::encode_size(size)?;
        for b in enc { repr.push(b as char); }
        Ok(())
    }

    fn pad_bitvector(bit_vec: &mut Vec<usize>) {
        if bit_vec.len() % 6 != 0 {
            (0..6 - (bit_vec.len() % 6)).for_each(|_| bit_vec.push(0));
        }
    }

    fn parse_bitvector(bit_vec: &[usize], repr: &mut String) -> Result<(), IOError> {
        for chunk in bit_vec.chunks(6) {
            let mut sum = 0;
            for (i, bit) in chunk.iter().rev().enumerate() {
                sum += bit * 2usize.pow(i as u32);
            }
            let raw = sum as u32 + 63;
            let c = char::from_u32(raw).ok_or(IOError::InvalidSizeChar)?;
            repr.push(c);
        }
        Ok(())
    }

    pub fn to_file(bit_vec: Vec<usize>, n: usize, is_directed: bool) -> Result<String, IOError> {
        // enforce graph6 maximum (2^36 - 1) like sparse6
        if n >= (1usize << 36) {
            return Err(IOError::GraphTooLarge);
        }
        let mut repr = String::new();
        let mut bit_vec = if !is_directed {
            if n < 2 {
                // For n=0 or n=1, upper triangle is empty.
                // This avoids an underflow in upper_triangle.
                Vec::new()
            } else {
                upper_triangle(&bit_vec, n)
            }
        } else {
            bit_vec
        };
        write_header(&mut repr, is_directed);
        write_size(&mut repr, n)?;
        pad_bitvector(&mut bit_vec);
        parse_bitvector(&bit_vec, &mut repr)?;
        Ok(repr)
    }
}

// WriteGraph is only used in tests via the tests module's imports

use crate::get_edge_iter_with_weights;
use crate::{graph::PyGraph, StablePyGraph};
use crate::{Graph6OverflowError, Graph6ParseError};
use flate2::write::GzEncoder;
use flate2::Compression;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use pyo3::prelude::*;
use pyo3::PyErr;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Undirected graph implementation
#[derive(Debug)]
pub struct Graph6 {
    pub bit_vec: Vec<usize>,
    pub n: usize,
}
impl Graph6 {
    /// Creates a new undirected graph from a graph6 representation
    pub fn from_g6(repr: &str) -> Result<Self, IOError> {
        let bytes = repr.as_bytes();
        let (n, n_len) = utils::parse_size(bytes, 0)?;
        let bit_vec = Self::build_bitvector(bytes, n, n_len)?;
        Ok(Self { bit_vec, n })
    }

    /// Creates a new undirected graph from a flattened adjacency matrix.
    /// The adjacency matrix must be square.
    /// The adjacency matrix will be forced into a symmetric matrix.
    pub fn from_adj(adj: &[usize]) -> Result<Self, IOError> {
        let n2 = adj.len();
        let n = (n2 as f64).sqrt() as usize;
        if n * n != n2 {
            return Err(IOError::InvalidAdjacencyMatrix);
        }
        let mut bit_vec = vec![0; n * n];
        for i in 0..n {
            for j in 0..n {
                if adj[i * n + j] == 1 {
                    let idx = i * n + j;
                    let jdx = j * n + i;
                    bit_vec[idx] = 1;
                    bit_vec[jdx] = 1;
                }
            }
        }
        Ok(Self { bit_vec, n })
    }

    /// Builds the bitvector from the graph6 representation
    fn build_bitvector(bytes: &[u8], n: usize, n_len: usize) -> Result<Vec<usize>, IOError> {
        // For n < 2 we still materialize an n*n bitvector (0-length for n=0, length 1 for n=1)
        // to avoid downstream index calculations (i * n + j) from panicking in utility
        // functions (DOT conversion, PyGraph conversion, etc.).
        if n < 2 {
            return Ok(vec![0; n * n]);
        }
        let bv_len = n * (n - 1) / 2;
        let Some(bit_vec) = utils::fill_bitvector(bytes, bv_len, n_len) else {
            return Err(IOError::NonCanonicalEncoding);
        };
        Self::fill_from_triangle(&bit_vec, n)
    }

    /// Fills the adjacency bitvector from an upper triangle
    fn fill_from_triangle(tri: &[usize], n: usize) -> Result<Vec<usize>, IOError> {
        let mut bit_vec = vec![0; n * n];
        let mut tri_iter = tri.iter();
        for i in 1..n {
            for j in 0..i {
                let idx = i * n + j;
                let jdx = j * n + i;
                let Some(&val) = tri_iter.next() else {
                    return Err(IOError::NonCanonicalEncoding);
                };
                bit_vec[idx] = val;
                bit_vec[jdx] = val;
            }
        }
        Ok(bit_vec)
    }
}
impl GraphConversion for Graph6 {
    fn bit_vec(&self) -> &[usize] {
        &self.bit_vec
    }

    fn size(&self) -> usize {
        self.n
    }

    fn is_directed(&self) -> bool {
        false
    }
}
impl write::WriteGraph for Graph6 {}

use crate::digraph6::{DiGraph6, digraph6_to_pydigraph};

// End of combined module

/// Convert internal Graph (undirected) to PyGraph
fn graph6_to_pygraph<'py>(py: Python<'py>, g: &Graph6) -> PyResult<Bound<'py, PyAny>> {
    let mut graph = StablePyGraph::<Undirected>::with_capacity(g.size(), 0);
    if g.bit_vec.len() < g.size().saturating_mul(g.size()) {
        return Err(Graph6ParseError::new_err("Bitvector shorter than n*n; invalid internal state"));
    }
    // add nodes
    for _ in 0..g.size() {
        graph.add_node(py.None());
    }
    // add edges
    for i in 0..g.size() {
        for j in i..g.size() {
            if g.bit_vec[i * g.size() + j] == 1 {
                let u = NodeIndex::new(i);
                let v = NodeIndex::new(j);
                graph.add_edge(u, v, py.None());
            }
        }
    }
    let out = PyGraph {
        graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(out.into_pyobject(py)?.into_any())
}

// digraph_to_pydigraph provided by crate::digraph6

/// Write a graph6 string to a file path. Supports gzip if the extension is `.gz`.
pub(crate) fn to_file(path: impl AsRef<Path>, content: &str) -> std::io::Result<()> {
    let extension = path
        .as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if extension == "gz" {
        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        let mut encoder = GzEncoder::new(buf_writer, Compression::default());
        encoder.write_all(content.as_bytes())?;
        encoder.finish()?;
    } else {
        std::fs::write(path, content)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(repr))]
pub fn read_graph6_str<'py>(py: Python<'py>, repr: &str) -> PyResult<Bound<'py, PyAny>> {

    enum ParserResult{
        Graph6(Graph6),
        DiGraph6(DiGraph6),
    }

    let result = if let Ok(g) = Graph6::from_g6(repr) {
        Ok(ParserResult::Graph6(g))
    } else if let Ok(dg) = DiGraph6::from_d6(repr) {
        Ok(ParserResult::DiGraph6(dg))
    } else {
        Err(IOError::NonCanonicalEncoding)
    };

    match result {
        Ok(ParserResult::Graph6(g)) => graph6_to_pygraph(py, &g),
        Ok(ParserResult::DiGraph6(dg)) => digraph6_to_pydigraph(py, &dg),
        Err(io_err) => Err(PyErr::from(io_err)),
    }

}

#[pyfunction]
#[pyo3(signature=(pygraph))]
pub fn graph_write_graph6_to_str<'py>(py: Python<'py>, pygraph: Py<PyGraph>) -> PyResult<String> {
    let g = pygraph.borrow(py);
    let n = g.graph.node_count();
    if n >= (1usize << 36) {
        return Err(Graph6OverflowError::new_err("Graph too large for graph6 encoding"));
    }
    // build bit_vec
    let mut bit_vec = vec![0usize; n * n];
    for (i, j, _w) in get_edge_iter_with_weights(&g.graph) {
        bit_vec[i * n + j] = 1;
        bit_vec[j * n + i] = 1;
    }
    let graph6 = write::to_file(bit_vec, n, false)?;
    Ok(graph6)
}

/// Parse the size header of a graph6 or digraph6 string.
///
/// Returns a tuple (n, size_field_length). For a directed (digraph6) string
/// starting with '&', pass offset=1. This function enforces canonical
/// encoding (shortest valid length) per the specification in:
/// https://users.cecs.anu.edu.au/~bdm/data/formats.txt
#[pyfunction]
#[pyo3(signature=(data, offset=0))]
pub fn parse_graph6_size(data: &str, offset: usize) -> PyResult<(usize, usize)> {
    let bytes = data.as_bytes();
    let (n, consumed) = utils::parse_size(bytes, offset)?;
    Ok((n, consumed))
}


/// Read a graph6 file from disk and return a PyGraph or PyDiGraph
#[pyfunction]
#[pyo3(signature=(path))]
pub fn read_graph6<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    use std::fs;
    let data = fs::read_to_string(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("IO error: {}", e)))?;
    // graph6 files may contain newlines; take first non-empty line
    let line = data.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    read_graph6_str(py, line)
}

/// Write a PyGraph to a graph6 file
#[pyfunction]
#[pyo3(signature=(graph, path))]
pub fn graph_write_graph6(py: Python<'_>, graph: Py<PyGraph>, path: &str) -> PyResult<()> {
    let s = graph_write_graph6_to_str(py, graph)?;
    to_file(path, &s)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("IO error: {}", e)))?;
    Ok(())
}
