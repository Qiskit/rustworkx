use crate::{get_edge_iter_with_weights, StablePyGraph};
use crate::graph6::{utils, IOError, GraphConversion};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use petgraph::graph::NodeIndex;
use petgraph::algo;

/// Directed graph implementation (extracted from graph6.rs)
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
    let (n, n_len) = utils::parse_size(bytes, 1)?;
    let Some(bit_vec) = Self::build_bitvector(bytes, n, 1 + n_len) else {
            return Err(IOError::NonCanonicalEncoding);
        };
        Ok(Self { bit_vec, n })
    }

    /// Creates a new DiGraph from a flattened adjacency matrix
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
    pub(crate) fn valid_digraph(repr: &[u8]) -> Result<bool, IOError> {
        if repr.is_empty() {
            return Err(IOError::InvalidDigraphHeader);
        }
        if repr[0] == b'&' {
            Ok(true)
        } else {
            Err(IOError::InvalidDigraphHeader)
        }
    }

    /// Iteratores through the bytes and builds a bitvector
    /// representing the adjaceny matrix of the graph
    fn build_bitvector(bytes: &[u8], n: usize, offset: usize) -> Option<Vec<usize>> {
        let bv_len = n * n;
        utils::fill_bitvector(bytes, bv_len, offset)
    }
}

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

/// Convert internal DiGraph to PyDiGraph
pub fn digraph_to_pydigraph<'py>(py: Python<'py>, g: &DiGraph) -> PyResult<Bound<'py, PyAny>> {
    use crate::graph6::GraphConversion as _;
    let mut graph = StablePyGraph::<petgraph::Directed>::with_capacity(g.size(), 0);
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
    let out = crate::digraph::PyDiGraph {
        graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(out.into_pyobject(py)?.into_any())
}

#[pyfunction]
#[pyo3(signature=(pydigraph))]
pub fn write_graph6_from_pydigraph(pydigraph: Py<crate::digraph::PyDiGraph>) -> PyResult<String> {
    Python::with_gil(|py| {
        let g = pydigraph.borrow(py);
        let n = g.graph.node_count();
        let mut bit_vec = vec![0usize; n * n];
        for (i, j, _w) in get_edge_iter_with_weights(&g.graph) {
            bit_vec[i * n + j] = 1;
        }
        let graph6 = crate::graph6::write::write_graph6(bit_vec, n, true)?;
        Ok(graph6)
    })
}

#[pyfunction]
#[pyo3(signature=(digraph, path))]
pub fn digraph_write_graph6_file(digraph: Py<crate::digraph::PyDiGraph>, path: &str) -> PyResult<()> {
    let s = write_graph6_from_pydigraph(digraph)?;
    crate::graph6::to_file(path, &s)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("IO error: {}", e)))?;
    Ok(())
}

// Enable write_graph() in tests for DiGraph via the WriteGraph trait
impl crate::graph6::write::WriteGraph for DiGraph {}
