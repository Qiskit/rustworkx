//! Combined module: conversion, error, utils, write, undirected and directed
//! This file is intended as a drop-in single-module alternative to
//! the separate files in `src/` so callers can `mod all; use all::...` and
//! avoid many `use super` / `use crate` imports inside the library.

#[allow(dead_code)]
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
    #[allow(dead_code)]
    InvalidAdjacencyMatrix,
    NonCanonicalEncoding,
}

/// Utility functions used by parsers and writers
pub mod utils {
    use super::IOError;

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

    /// Returns the size of the graph
    pub fn get_size(bytes: &[u8], pos: usize) -> Result<usize, IOError> {
        let size = bytes[pos];
        if size == 126 {
            Err(IOError::GraphTooLarge)
        } else if size < 63 {
            Err(IOError::InvalidSizeChar)
        } else {
            Ok((size - 63) as usize)
        }
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

    /// Trait to write graphs into graph 6 formatted strings
    #[allow(dead_code)]
    pub trait WriteGraph: GraphConversion {
        fn write_graph(&self) -> String {
            write_graph6(self.bit_vec().to_vec(), self.size(), self.is_directed())
        }
    }

    fn write_header(repr: &mut String, is_directed: bool) {
        if is_directed {
            repr.push('&');
        }
    }

    fn write_size(repr: &mut String, size: usize) {
        let size_char = char::from_u32(size as u32 + 63).unwrap();
        repr.push(size_char);
    }

    fn pad_bitvector(bit_vec: &mut Vec<usize>) {
        if bit_vec.len() % 6 != 0 {
            (0..6 - (bit_vec.len() % 6)).for_each(|_| bit_vec.push(0));
        }
    }

    fn parse_bitvector(bit_vec: &[usize], repr: &mut String) {
        for chunk in bit_vec.chunks(6) {
            let mut sum = 0;
            for (i, bit) in chunk.iter().rev().enumerate() {
                sum += bit * 2usize.pow(i as u32);
            }
            let char = char::from_u32(sum as u32 + 63).unwrap();
            repr.push(char);
        }
    }

    pub fn write_graph6(bit_vec: Vec<usize>, n: usize, is_directed: bool) -> String {
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
        write_size(&mut repr, n);
        pad_bitvector(&mut bit_vec);
        parse_bitvector(&bit_vec, &mut repr);
        repr
    }
}

// WriteGraph is only used in tests via the tests module's imports

use crate::get_edge_iter_with_weights;
use crate::{digraph::PyDiGraph, graph::PyGraph, StablePyGraph};
use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use flate2::write::GzEncoder;
use flate2::Compression;

/// Undirected graph implementation
#[derive(Debug)]
pub struct Graph {
    pub bit_vec: Vec<usize>,
    pub n: usize,
}
impl Graph {
    /// Creates a new undirected graph from a graph6 representation
    pub fn from_g6(repr: &str) -> Result<Self, IOError> {
        let bytes = repr.as_bytes();
        let n = utils::get_size(bytes, 0)?;
        let bit_vec = Self::build_bitvector(bytes, n)?;
        Ok(Self { bit_vec, n })
    }

    /// Creates a new undirected graph from a flattened adjacency matrix.
    /// The adjacency matrix must be square.
    /// The adjacency matrix will be forced into a symmetric matrix.
    #[cfg(test)]
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
    fn build_bitvector(bytes: &[u8], n: usize) -> Result<Vec<usize>, IOError> {
        if n < 2 {
            return Ok(Vec::new());
        }
        let bv_len = n * (n - 1) / 2;
        let Some(bit_vec) = utils::fill_bitvector(bytes, bv_len, 1) else {
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
#[allow(dead_code)]
impl GraphConversion for Graph {
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
#[cfg(test)]
impl write::WriteGraph for Graph {}

/// Directed graph implementation
#[derive(Debug)]
pub struct DiGraph {
    pub bit_vec: Vec<usize>,
    pub n: usize,
}
impl DiGraph {
    /// Creates a new DiGraph from a graph6 representation string
    pub fn from_d6(repr: &str) -> Result<Self, IOError> {
        let bytes = repr.as_bytes();
        Self::valid_digraph(bytes)?;
        let n = utils::get_size(bytes, 1)?;
        let Some(bit_vec) = Self::build_bitvector(bytes, n) else {
            return Err(IOError::NonCanonicalEncoding);
        };
        Ok(Self { bit_vec, n })
    }

    /// Creates a new DiGraph from a flattened adjacency matrix
    #[cfg(test)]
    pub fn from_adj(adj: &[usize]) -> Result<Self, IOError> {
        let n2 = adj.len();
        let n = (n2 as f64).sqrt() as usize;
        if n * n != n2 {
            return Err(IOError::InvalidAdjacencyMatrix);
        }
        let bit_vec = adj.to_vec();
        Ok(Self { bit_vec, n })
    }

    /// Validates graph6 directed representation
    fn valid_digraph(repr: &[u8]) -> Result<bool, IOError> {
        if repr[0] == b'&' {
            Ok(true)
        } else {
            Err(IOError::InvalidDigraphHeader)
        }
    }

    /// Iteratores through the bytes and builds a bitvector
    /// representing the adjaceny matrix of the graph
    fn build_bitvector(bytes: &[u8], n: usize) -> Option<Vec<usize>> {
        let bv_len = n * n;
        utils::fill_bitvector(bytes, bv_len, 2)
    }
}
#[allow(dead_code)]
impl GraphConversion for DiGraph {
    fn bit_vec(&self) -> &[usize] {
        &self.bit_vec
    }

    fn size(&self) -> usize {
        self.n
    }

    fn is_directed(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[cfg(test)]
impl write::WriteGraph for DiGraph {}

// End of combined module

/// Convert internal Graph (undirected) to PyGraph
fn graph_to_pygraph<'py>(py: Python<'py>, g: &Graph) -> PyResult<Bound<'py, PyAny>> {
    let mut graph = StablePyGraph::<Undirected>::with_capacity(g.size(), 0);
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

/// Convert internal DiGraph to PyDiGraph
fn digraph_to_pydigraph<'py>(py: Python<'py>, g: &DiGraph) -> PyResult<Bound<'py, PyAny>> {
    let mut graph = StablePyGraph::<Directed>::with_capacity(g.size(), 0);
    for _ in 0..g.size() {
        graph.add_node(py.None());
    }
    for i in 0..g.size() {
        for j in 0..g.size() {
            if g.bit_vec[i * g.size() + j] == 1 {
                let u = NodeIndex::new(i);
                let v = NodeIndex::new(j);
                graph.add_edge(u, v, py.None());
            }
        }
    }
    let out = PyDiGraph {
        graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(out.into_pyobject(py)?.into_any())
}

/// Write a graph6 string to a file path. Supports gzip if the extension is `.gz`.
fn to_file(path: impl AsRef<Path>, content: &str) -> std::io::Result<()> {
    let extension = path.as_ref().extension().and_then(|e| e.to_str()).unwrap_or("");
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
    // try undirected first
    if let Ok(g) = Graph::from_g6(repr) {
        return graph_to_pygraph(py, &g);
    }
    // try directed
    if let Ok(dg) = DiGraph::from_d6(repr) {
        return digraph_to_pydigraph(py, &dg);
    }
    Err(PyException::new_err("Failed to parse graph6 string"))
}

#[pyfunction]
#[pyo3(signature=(pygraph))]
pub fn write_graph6_from_pygraph(pygraph: Py<PyGraph>) -> PyResult<String> {
    Python::with_gil(|py| {
        let g = pygraph.borrow(py);
        let n = g.graph.node_count();
        // build bit_vec
        let mut bit_vec = vec![0usize; n * n];
        for (i, j, _w) in get_edge_iter_with_weights(&g.graph) {
            bit_vec[i * n + j] = 1;
            bit_vec[j * n + i] = 1;
        }
        let graph6 = write::write_graph6(bit_vec, n, false);
        Ok(graph6)
    })
}

#[pyfunction]
#[pyo3(signature=(pydigraph))]
pub fn write_graph6_from_pydigraph(pydigraph: Py<PyDiGraph>) -> PyResult<String> {
    Python::with_gil(|py| {
        let g = pydigraph.borrow(py);
        let n = g.graph.node_count();
        let mut bit_vec = vec![0usize; n * n];
        for (i, j, _w) in get_edge_iter_with_weights(&g.graph) {
            bit_vec[i * n + j] = 1;
        }
        let graph6 = write::write_graph6(bit_vec, n, true);
        Ok(graph6)
    })
}

/// Read a graph6 file from disk and return a PyGraph or PyDiGraph
#[pyfunction]
#[pyo3(signature=(path))]
pub fn read_graph6_file<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    use std::fs;
    let data =
        fs::read_to_string(path).map_err(|e| PyException::new_err(format!("IO error: {}", e)))?;
    // graph6 files may contain newlines; take first non-empty line
    let line = data.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    read_graph6_str(py, line)
}

/// Write a PyGraph to a graph6 file
#[pyfunction]
#[pyo3(signature=(graph, path))]
pub fn graph_write_graph6_file(graph: Py<PyGraph>, path: &str) -> PyResult<()> {
    let s = write_graph6_from_pygraph(graph)?;
    to_file(path, &s).map_err(|e| PyException::new_err(format!("IO error: {}", e)))?;
    Ok(())
}

/// Write a PyDiGraph to a graph6 file
#[pyfunction]
#[pyo3(signature=(digraph, path))]
pub fn digraph_write_graph6_file(digraph: Py<PyDiGraph>, path: &str) -> PyResult<()> {
    let s = write_graph6_from_pydigraph(digraph)?;
    to_file(path, &s).map_err(|e| PyException::new_err(format!("IO error: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod testing {
    use super::utils::{fill_bitvector, get_size, upper_triangle};
    use super::write::{write_graph6, WriteGraph};
    use super::{DiGraph, Graph, GraphConversion, IOError};

    // Tests from error.rs
    #[test]
    fn test_error_enum() {
        let err = IOError::InvalidDigraphHeader;
        println!("{:?}", err);
    }

    // Tests from utils.rs
    #[test]
    fn test_size_pos_0() {
        let bytes = b"AG";
        let size = get_size(bytes, 0).unwrap();
        assert_eq!(size, 2);
    }

    #[test]
    fn test_size_pos_1() {
        let bytes = b"&AG";
        let size = get_size(bytes, 1).unwrap();
        assert_eq!(size, 2);
    }

    #[test]
    fn test_size_oversize() {
        let bytes = b"~AG";
        let size = get_size(bytes, 0).unwrap_err();
        assert_eq!(size, IOError::GraphTooLarge);
    }

    #[test]
    fn test_size_invalid_size_char() {
        let bytes = b">AG";
        let size = get_size(bytes, 0).unwrap_err();
        assert_eq!(size, IOError::InvalidSizeChar);
    }

    #[test]
    fn test_bitvector() {
        let bytes = b"Bw";
        let n = 3;
        let bit_vec = fill_bitvector(bytes, n * n, 0).unwrap();
        assert_eq!(bit_vec, vec![0, 0, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_bitvector_offset() {
        let bytes = b"Bw";
        let n = 2;
        let bit_vec = fill_bitvector(bytes, n * n, 1).unwrap();
        assert_eq!(bit_vec, vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_upper_triangle_n2() {
        let bit_vec = vec![0, 1, 1, 0];
        let tri = upper_triangle(&bit_vec, 2);
        assert_eq!(tri, vec![1]);
    }

    #[test]
    fn test_upper_triangle_n3() {
        let bit_vec = vec![0, 1, 1, 1, 0, 0, 1, 0, 0];
        let tri = upper_triangle(&bit_vec, 3);
        assert_eq!(tri, vec![1, 1, 0]);
    }

    // Tests from write.rs
    #[test]
    fn test_write_undirected_n2() {
        let bit_vec = vec![0, 1, 1, 0];
        let repr = write_graph6(bit_vec, 2, false);
        assert_eq!(repr, "A_");
    }

    #[test]
    fn test_write_directed_n2_mirror() {
        let bit_vec = vec![0, 1, 1, 0];
        let repr = write_graph6(bit_vec, 2, true);
        assert_eq!(repr, "&AW");
    }

    #[test]
    fn test_write_directed_n2_unmirrored() {
        let bit_vec = vec![0, 0, 1, 0];
        let repr = write_graph6(bit_vec, 2, true);
        assert_eq!(repr, "&AG");
    }

    // Tests from undirected.rs
    #[test]
    fn test_graph_n2() {
        let graph = Graph::from_g6("A_").unwrap();
        assert_eq!(graph.size(), 2);
        assert_eq!(graph.bit_vec(), &[0, 1, 1, 0]);
    }

    #[test]
    fn test_graph_n2_empty() {
        let graph = Graph::from_g6("A?").unwrap();
        assert_eq!(graph.size(), 2);
        assert_eq!(graph.bit_vec(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_graph_n3() {
        let graph = Graph::from_g6("Bw").unwrap();
        assert_eq!(graph.size(), 3);
        assert_eq!(graph.bit_vec(), &[0, 1, 1, 1, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_graph_n4() {
        let graph = Graph::from_g6("C~").unwrap();
        assert_eq!(graph.size(), 4);
        assert_eq!(
            graph.bit_vec(),
            &[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn test_too_short_input() {
        let parsed = Graph::from_g6("a");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_invalid_char() {
        let parsed = Graph::from_g6("A1");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_to_adjacency() {
        let graph = Graph::from_g6("A_").unwrap();
        let adj = graph.to_adjmat();
        assert_eq!(adj, "0 1\n1 0\n");
    }

    #[test]
    fn test_to_dot() {
        let graph = Graph::from_g6("A_").unwrap();
        let dot = graph.to_dot(None);
        assert_eq!(dot, "graph {\n0 -- 1;\n}");
    }

    #[test]
    fn test_to_dot_with_label() {
        let graph = Graph::from_g6("A_").unwrap();
        let dot = graph.to_dot(Some(1));
        assert_eq!(dot, "graph graph_1 {\n0 -- 1;\n}");
    }

    #[test]
    fn test_to_net() {
        let repr = r"A_";
        let graph = Graph::from_g6(repr).unwrap();
        let net = graph.to_net();
        assert_eq!(net, "*Vertices 2\n1 \"0\"\n2 \"1\"\n*Arcs\n1 2\n2 1\n");
    }

    #[test]
    fn test_to_flat() {
        let repr = r"A_";
        let graph = Graph::from_g6(repr).unwrap();
        let flat = graph.to_flat();
        assert_eq!(flat, "0110");
    }

    #[test]
    fn test_write_n2() {
        let repr = r"A_";
        let graph = Graph::from_g6(repr).unwrap();
        let g6 = graph.write_graph();
        assert_eq!(g6, repr);
    }

    #[test]
    fn test_write_n3() {
        let repr = r"Bw";
        let graph = Graph::from_g6(repr).unwrap();
        let g6 = graph.write_graph();
        assert_eq!(g6, repr);
    }

    #[test]
    fn test_write_n4() {
        let repr = r"C~";
        let graph = Graph::from_g6(repr).unwrap();
        let g6 = graph.write_graph();
        assert_eq!(g6, repr);
    }

    #[test]
    fn test_from_adj() {
        let adj = &[0, 0, 1, 0];
        let graph = Graph::from_adj(adj).unwrap();
        assert_eq!(graph.size(), 2);
        assert_eq!(graph.bit_vec(), &[0, 1, 1, 0]);
        assert_eq!(graph.write_graph(), "A_");
    }

    #[test]
    fn test_from_nonsquare_adj() {
        let adj = &[0, 0, 1, 0, 1];
        let graph = Graph::from_adj(adj);
        assert!(graph.is_err());
    }

    // Tests from directed.rs
    #[test]
    fn test_header() {
        let repr = b"&AG";
        assert!(DiGraph::valid_digraph(repr).is_ok());
    }

    #[test]
    fn test_invalid_header() {
        let repr = b"AG";
        assert!(DiGraph::valid_digraph(repr).is_err());
    }

    #[test]
    fn test_from_adj_directed() {
        let adj = &[0, 0, 1, 0];
        let graph = DiGraph::from_adj(adj).unwrap();
        assert_eq!(graph.size(), 2);
        assert_eq!(graph.bit_vec(), vec![0, 0, 1, 0]);
        assert_eq!(graph.write_graph(), "&AG");
    }

    #[test]
    fn test_from_nonsquare_adj_directed() {
        let adj = &[0, 0, 1, 0, 1];
        let graph = DiGraph::from_adj(adj);
        assert!(graph.is_err());
    }

    #[test]
    fn test_bitvector_n2() {
        let repr = "&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        assert_eq!(graph.size(), 2);
        assert_eq!(graph.bit_vec(), vec![0, 0, 1, 0]);
    }

    #[test]
    fn test_bitvector_n3() {
        let repr = r"&B\o";
        let graph = DiGraph::from_d6(repr).unwrap();
        assert_eq!(graph.size(), 3);
        assert_eq!(graph.bit_vec(), vec![0, 1, 1, 1, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_bitvector_n4() {
        let repr = r"&C]|w";
        let graph = DiGraph::from_d6(repr).unwrap();
        assert_eq!(graph.size(), 4);
        assert_eq!(
            graph.bit_vec(),
            vec![0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn test_init_invalid_n2() {
        let repr = "AG";
        let graph = DiGraph::from_d6(repr);
        assert!(graph.is_err());
    }

    #[test]
    fn test_to_adjacency_directed() {
        let repr = r"&C]|w";
        let graph = DiGraph::from_d6(repr).unwrap();
        let adj = graph.to_adjmat();
        assert_eq!(adj, "0 1 1 1\n1 0 1 1\n1 1 0 1\n1 1 1 0\n");
    }

    #[test]
    fn test_to_dot_directed() {
        let repr = r"&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        let dot = graph.to_dot(None);
        assert_eq!(dot, "digraph {\n1 -> 0;\n}");
    }

    #[test]
    fn test_to_dot_with_id_directed() {
        let repr = r"&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        let dot = graph.to_dot(Some(1));
        assert_eq!(dot, "digraph graph_1 {\n1 -> 0;\n}");
    }

    #[test]
    fn test_to_net_directed() {
        let repr = r"&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        let net = graph.to_net();
        assert_eq!(net, "*Vertices 2\n1 \"0\"\n2 \"1\"\n*Arcs\n2 1\n");
    }

    #[test]
    fn test_to_flat_directed() {
        let repr = r"&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        let flat = graph.to_flat();
        assert_eq!(flat, "0010");
    }

    #[test]
    fn test_write_n2_directed() {
        let repr = r"&AG";
        let graph = DiGraph::from_d6(repr).unwrap();
        let graph6 = graph.write_graph();
        assert_eq!(graph6, repr);
    }

    #[test]
    fn test_write_n3_directed() {
        let repr = r"&B\o";
        let graph = DiGraph::from_d6(repr).unwrap();
        let graph6 = graph.write_graph();
        assert_eq!(graph6, repr);
    }

    #[test]
    fn test_write_n4_directed() {
        let repr = r"&C]|w";
        let graph = DiGraph::from_d6(repr).unwrap();
        let graph6 = graph.write_graph();
        assert_eq!(graph6, repr);
    }
}
