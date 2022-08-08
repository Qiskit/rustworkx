// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use std::collections::VecDeque;
use std::iter;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Undirected;

use pyo3::exceptions::{PyIndexError, PyOverflowError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use super::{digraph, graph, StablePyGraph};

pub fn pairwise<I>(right: I) -> impl Iterator<Item = (Option<I::Item>, I::Item)>
where
    I: IntoIterator + Clone,
{
    let left = iter::once(None).chain(right.clone().into_iter().map(Some));
    left.zip(right)
}

#[inline]
fn get_num_nodes(num_nodes: &Option<usize>, weights: &Option<Vec<PyObject>>) -> usize {
    if weights.is_some() {
        weights.as_ref().unwrap().len()
    } else {
        num_nodes.unwrap()
    }
}

/// Generate a cycle graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the cycle graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated cycle graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_cycle_graph(5)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)")]
pub fn directed_cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let num_edges = if bidirectional {
        2 * node_len
    } else {
        node_len
    };
    let mut graph = StablePyGraph::<Directed>::with_capacity(node_len, num_edges);
    if node_len == 0 {
        return Ok(digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }

    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };
    for a in 0..node_len - 1 {
        let node_b = NodeIndex::new(a + 1);
        let node_a = NodeIndex::new(a);
        graph.add_edge(node_a, node_b, py.None());
        if bidirectional {
            graph.add_edge(node_b, node_a, py.None());
        }
    }
    let last_node_index = NodeIndex::new(node_len - 1);
    let first_node_index = NodeIndex::new(0);
    graph.add_edge(last_node_index, first_node_index, py.None());
    if bidirectional {
        graph.add_edge(first_node_index, last_node_index, py.None());
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected cycle graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the cycle graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated cycle graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.cycle_graph(5)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, multigraph=True)")]
pub fn cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let mut graph = StablePyGraph::<Undirected>::with_capacity(node_len, node_len);
    if node_len == 0 {
        return Ok(graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }

    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };
    for node_a in 0..node_len - 1 {
        let node_b = node_a + 1;
        graph.add_edge(NodeIndex::new(node_a), NodeIndex::new(node_b), py.None());
    }
    let last_node_index = NodeIndex::new(node_len - 1);
    let first_node_index = NodeIndex::new(0);
    graph.add_edge(last_node_index, first_node_index, py.None());
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed path graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the path graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated path graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_path_graph(10)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)")]
pub fn directed_path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let num_edges = if bidirectional {
        2 * node_len
    } else {
        node_len
    };
    let mut graph = StablePyGraph::<Directed>::with_capacity(node_len, num_edges);
    if node_len == 0 {
        return Ok(digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }

    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => (0..node_len).for_each(|_| {
            graph.add_node(py.None());
        }),
    };
    for a in 0..node_len - 1 {
        let node_b = NodeIndex::new(a + 1);
        let node_a = NodeIndex::new(a);
        graph.add_edge(node_a, node_b, py.None());
        if bidirectional {
            graph.add_edge(node_b, node_a, py.None());
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected path graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the path graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated path graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.path_graph(10)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, multigraph=True)")]
pub fn path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let mut graph = StablePyGraph::<Undirected>::with_capacity(node_len, node_len);
    if node_len == 0 {
        return Ok(graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }
    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => (0..node_len).for_each(|_| {
            graph.add_node(py.None());
        }),
    };
    for node_a in 0..node_len - 1 {
        let node_b = NodeIndex::new(node_a + 1);
        graph.add_edge(NodeIndex::new(node_a), node_b, py.None());
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed star graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the star graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``.
/// :param bool inward: If set ``True`` the nodes will be directed towards the
///     center node. This parameter is ignored if ``bidirectional`` is set to
///     ``True``.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated star graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_star_graph(10)
///   mpl_draw(graph)
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_star_graph(10, inward=True)
///   mpl_draw(graph)
///
#[pyfunction(inward = "false", bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(/, num_nodes=None, weights=None, inward=False, bidirectional=False, multigraph=True)"
)]
pub fn directed_star_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    inward: bool,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    if node_len == 0 {
        return Ok(digraph::PyDiGraph {
            graph: StablePyGraph::<Directed>::default(),
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }
    let num_edges = if bidirectional {
        (2 * node_len) - 2
    } else {
        node_len - 1
    };
    let mut graph = StablePyGraph::<Directed>::with_capacity(node_len, num_edges);
    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };
    let zero_index = NodeIndex::new(0);
    for node_index in 1..node_len {
        //Add edges in both directions if bidirection is True
        let node = NodeIndex::new(node_index);
        if bidirectional {
            graph.add_edge(node, zero_index, py.None());
            graph.add_edge(zero_index, node, py.None());
        } else if inward {
            graph.add_edge(node, zero_index, py.None());
        } else {
            graph.add_edge(zero_index, node, py.None());
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected star graph
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the star graph. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated star graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.star_graph(10)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, multigraph=True)")]
pub fn star_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    if node_len == 0 {
        return Ok(graph::PyGraph {
            graph: StablePyGraph::<Undirected>::default(),
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }
    let mut graph = StablePyGraph::<Undirected>::with_capacity(node_len, node_len - 1);
    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };
    let zero_index = NodeIndex::new(0);
    for node in 1..node_len {
        graph.add_edge(zero_index, NodeIndex::new(node), py.None());
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected mesh graph where every node is connected to every other
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated mesh graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.mesh_graph(4)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, multigraph=True)")]
pub fn mesh_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    if node_len == 0 {
        return Ok(graph::PyGraph {
            graph: StablePyGraph::<Undirected>::default(),
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }
    let num_edges = (node_len * (node_len - 1)) / 2;
    let mut graph = StablePyGraph::<Undirected>::with_capacity(node_len, num_edges);
    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };

    for i in 0..node_len - 1 {
        for j in i + 1..node_len {
            let i_index = NodeIndex::new(i);
            let j_index = NodeIndex::new(j);
            graph.add_edge(i_index, j_index, py.None());
        }
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed mesh graph where every node is connected to every other
///
/// :param int num_node: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_node`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated mesh graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_mesh_graph(4)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = "true")]
#[pyo3(text_signature = "(/, num_nodes=None, weights=None, multigraph=True)")]
pub fn directed_mesh_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    if node_len == 0 {
        return Ok(digraph::PyDiGraph {
            graph: StablePyGraph::<Directed>::default(),
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }
    let num_edges = node_len * (node_len - 1);
    let mut graph = StablePyGraph::<Directed>::with_capacity(node_len, num_edges);
    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            (0..node_len).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };
    for i in 0..node_len - 1 {
        for j in i + 1..node_len {
            let i_index = NodeIndex::new(i);
            let j_index = NodeIndex::new(j);
            graph.add_edge(i_index, j_index, py.None());
            graph.add_edge(j_index, i_index, py.None());
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected grid graph.
///
/// :param int rows: The number of rows to generate the graph with.
///     If specified, cols also need to be specified
/// :param list cols: The number of rows to generate the graph with.
///     If specified, rows also need to be specified. rows*cols
///     defines the number of nodes in the graph
/// :param list weights: A list of node weights. Nodes are filled row wise.
///     If rows and cols are not specified, then a linear graph containing
///     all the values in weights list is created.
///     If number of nodes(rows*cols) is less than length of
///     weights list, the trailing weights are ignored.
///     If number of nodes(rows*cols) is greater than length of
///     weights list, extra nodes with None weight are appended.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated grid graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``rows`` or ``cols`` and ``weights`` are
///      specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.grid_graph(2, 3)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, rows=None, cols=None, weights=None, multigraph=True)")]
pub fn grid_graph(
    py: Python,
    rows: Option<usize>,
    cols: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if weights.is_none() && (rows.is_none() || cols.is_none()) {
        return Err(PyIndexError::new_err(
            "dimensions and weights list not specified",
        ));
    }

    let mut rowlen = rows.unwrap_or(0);
    let mut collen = cols.unwrap_or(0);
    let mut num_nodes = rowlen * collen;
    let mut num_edges = 0;
    if num_nodes == 0 {
        if weights.is_none() {
            return Ok(graph::PyGraph {
                graph: StablePyGraph::<Undirected>::default(),
                node_removed: false,
                multigraph,
                attrs: py.None(),
            });
        }
    } else {
        num_edges = (rowlen - 1) * collen + (collen - 1) * rowlen;
    }
    let mut graph = StablePyGraph::<Undirected>::with_capacity(num_nodes, num_edges);

    match weights {
        Some(weights) => {
            if num_nodes < weights.len() && rowlen == 0 {
                collen = weights.len();
                rowlen = 1;
                num_nodes = collen;
            }

            let mut node_cnt = num_nodes;

            for weight in weights {
                if node_cnt == 0 {
                    break;
                }
                graph.add_node(weight);
                node_cnt -= 1;
            }
            for _i in 0..node_cnt {
                graph.add_node(py.None());
            }
        }
        None => {
            (0..num_nodes).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new((i + 1) * collen + j),
                    py.None(),
                );
            }
            if j + 1 < collen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new(i * collen + j + 1),
                    py.None(),
                );
            }
        }
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed grid graph. The edges propagate towards right and
///     bottom direction if ``bidirectional`` is ``false``
///
/// :param int rows: The number of rows to generate the graph with.
///     If specified, cols also need to be specified.
/// :param list cols: The number of rows to generate the graph with.
///     If specified, rows also need to be specified. rows*cols
///     defines the number of nodes in the graph.
/// :param list weights: A list of node weights. Nodes are filled row wise.
///     If rows and cols are not specified, then a linear graph containing
///     all the values in weights list is created.
///     If number of nodes(rows*cols) is less than length of
///     weights list, the trailing weights are ignored.
///     If number of nodes(rows*cols) is greater than length of
///     weights list, extra nodes with None weight are appended.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated grid graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``rows`` or ``cols`` and ``weights`` are
///      specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_grid_graph(2, 3)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(/, rows=None, cols=None, weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_grid_graph(
    py: Python,
    rows: Option<usize>,
    cols: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if weights.is_none() && (rows.is_none() || cols.is_none()) {
        return Err(PyIndexError::new_err(
            "dimensions and weights list not specified",
        ));
    }

    let mut rowlen = rows.unwrap_or(0);
    let mut collen = cols.unwrap_or(0);
    let mut num_nodes = rowlen * collen;
    let mut num_edges = 0;
    if num_nodes == 0 {
        if weights.is_none() {
            return Ok(digraph::PyDiGraph {
                graph: StablePyGraph::<Directed>::default(),
                node_removed: false,
                check_cycle: false,
                cycle_state: algo::DfsSpace::default(),
                multigraph,
                attrs: py.None(),
            });
        }
    } else {
        num_edges = (rowlen - 1) * collen + (collen - 1) * rowlen;
    }
    if bidirectional {
        num_edges *= 2;
    }
    let mut graph = StablePyGraph::<Directed>::with_capacity(num_nodes, num_edges);

    match weights {
        Some(weights) => {
            if num_nodes < weights.len() && rowlen == 0 {
                collen = weights.len();
                rowlen = 1;
                num_nodes = collen;
            }

            let mut node_cnt = num_nodes;

            for weight in weights {
                if node_cnt == 0 {
                    break;
                }
                graph.add_node(weight);
                node_cnt -= 1;
            }
            for _i in 0..node_cnt {
                graph.add_node(py.None());
            }
        }
        None => {
            (0..num_nodes).for_each(|_| {
                graph.add_node(py.None());
            });
        }
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new((i + 1) * collen + j),
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        NodeIndex::new((i + 1) * collen + j),
                        NodeIndex::new(i * collen + j),
                        py.None(),
                    );
                }
            }

            if j + 1 < collen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new(i * collen + j + 1),
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        NodeIndex::new(i * collen + j + 1),
                        NodeIndex::new(i * collen + j),
                        py.None(),
                    );
                }
            }
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

// MAX_ORDER is determined based on the pointer width of the target platform
#[cfg(target_pointer_width = "64")]
const MAX_ORDER: u32 = 60;
#[cfg(not(target_pointer_width = "64"))]
const MAX_ORDER: u32 = 29;

/// Generate an undirected binomial tree of order n recursively.
///
/// :param int order: Order of the binomial tree. The maximum allowed value
///     for order on the platform your running on. If it's a 64bit platform
///     the max value is 59 and on 32bit systems the max value is 29. Any order
///     value above these will raise a ``OverflowError``.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order extra nodes with with None will be appended.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A binomial tree with 2^n vertices and 2^n - 1 edges.
/// :rtype: PyGraph
/// :raises IndexError: If the length of ``weights`` is greater that 2^n
/// :raises OverflowError: If the input order exceeds the maximum value for the
///     current platform.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.binomial_tree_graph(4)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(order, /, weights=None, multigraph=True)")]
pub fn binomial_tree_graph(
    py: Python,
    order: u32,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if order >= MAX_ORDER {
        return Err(PyOverflowError::new_err(format!(
            "An order of {} exceeds the max allowable size",
            order
        )));
    }
    let num_nodes = usize::pow(2, order);
    let num_edges = usize::pow(2, order) - 1;
    let mut graph = StablePyGraph::<Undirected>::with_capacity(num_nodes, num_edges);
    for i in 0..num_nodes {
        match weights {
            Some(ref weights) => {
                if weights.len() > num_nodes {
                    return Err(PyIndexError::new_err("weights should be <= 2**order"));
                }
                if i < weights.len() {
                    graph.add_node(weights[i].clone_ref(py))
                } else {
                    graph.add_node(py.None())
                }
            }
            None => graph.add_node(py.None()),
        };
    }

    let mut n = 1;
    let zero_index = NodeIndex::new(0);

    for _ in 0..order {
        let edges: Vec<(NodeIndex, NodeIndex)> = graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect();
        for (source, target) in edges {
            let source_index = NodeIndex::new(source.index() + n);
            let target_index = NodeIndex::new(target.index() + n);

            graph.add_edge(source_index, target_index, py.None());
        }

        graph.add_edge(zero_index, NodeIndex::new(n), py.None());

        n *= 2;
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Creates a full r-ary tree of `n` nodes.
/// Sometimes called a k-ary, n-ary, or m-ary tree.
///
/// :param int order: Order of the tree.
/// :param list weights: A list of node weights. If the number of weights is
///     less than n, extra nodes with with None will be appended.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A r-ary tree.
/// :rtype: PyGraph
/// :raises IndexError: If the lenght of ``weights`` is greater that n
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.full_rary_tree(5, 15)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(branching_factor, num_nodes, /, weights=None, multigraph=True)")]
pub fn full_rary_tree(
    py: Python,
    branching_factor: u32,
    num_nodes: usize,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = StablePyGraph::<Undirected>::default();

    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::with_capacity(num_nodes);
            if weights.len() > num_nodes {
                return Err(PyIndexError::new_err("weights can't be greater than nodes"));
            }
            let node_count = num_nodes - weights.len();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            for _ in 0..node_count {
                let index = graph.add_node(py.None());
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes).map(|_| graph.add_node(py.None())).collect(),
    };

    if num_nodes > 0 {
        let mut parents = VecDeque::from(vec![nodes[0].index()]);
        let mut nod_it: usize = 1;

        while !parents.is_empty() {
            let source: usize = parents.pop_front().unwrap(); //If is empty it will never try to pop
            for _ in 0..branching_factor {
                if nod_it < num_nodes {
                    let target: usize = nodes[nod_it].index();
                    parents.push_back(target);
                    nod_it += 1;
                    graph.add_edge(nodes[source], nodes[target], py.None());
                }
            }
        }
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected binomial tree of order n recursively.
/// The edges propagate towards right and bottom direction if ``bidirectional`` is ``false``
///
/// :param int order: Order of the binomial tree. The maximum allowed value
///     for order on the platform your running on. If it's a 64bit platform
///     the max value is 59 and on 32bit systems the max value is 29. Any order
///     value above these will raise a ``OverflowError``.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order extra nodes with None will be appended.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A directed binomial tree with 2^n vertices and 2^n - 1 edges.
/// :rtype: PyDiGraph
/// :raises IndexError: If the lenght of ``weights`` is greater that 2^n
/// :raises OverflowError: If the input order exceeds the maximum value for the
///     current platform.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_binomial_tree_graph(4)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(text_signature = "(order, /,  weights=None, bidirectional=False, multigraph=True)")]
pub fn directed_binomial_tree_graph(
    py: Python,
    order: u32,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    if order >= MAX_ORDER {
        return Err(PyOverflowError::new_err(format!(
            "An order of {} exceeds the max allowable size",
            order
        )));
    }
    let num_nodes = usize::pow(2, order);
    let num_edges = usize::pow(2, order) - 1;
    let mut graph = StablePyGraph::<Directed>::with_capacity(num_nodes, num_edges);

    for i in 0..num_nodes {
        match weights {
            Some(ref weights) => {
                if weights.len() > num_nodes {
                    return Err(PyIndexError::new_err("weights should be <= 2**order"));
                }
                if i < weights.len() {
                    graph.add_node(weights[i].clone_ref(py))
                } else {
                    graph.add_node(py.None())
                }
            }
            None => graph.add_node(py.None()),
        };
    }

    let mut n = 1;
    let zero_index = NodeIndex::new(0);

    for _ in 0..order {
        let edges: Vec<(NodeIndex, NodeIndex)> = graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect();

        for (source, target) in edges {
            let source_index = NodeIndex::new(source.index() + n);
            let target_index = NodeIndex::new(target.index() + n);

            if graph.find_edge(source_index, target_index).is_none() {
                graph.add_edge(source_index, target_index, py.None());
            }

            if bidirectional && graph.find_edge(target_index, source_index).is_none() {
                graph.add_edge(target_index, source_index, py.None());
            }
        }
        let n_index = NodeIndex::new(n);

        if graph.find_edge(zero_index, n_index).is_none() {
            graph.add_edge(zero_index, n_index, py.None());
        }

        if bidirectional && graph.find_edge(n_index, zero_index).is_none() {
            graph.add_edge(n_index, zero_index, py.None());
        }

        n *= 2;
    }

    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected heavy square graph. Fig. 6 of
/// https://arxiv.org/abs/1907.09528.
/// An ASCII diagram of the graph is given by:
///
/// .. code-block:: console
///
///     ...       S   ...
///        \     / \
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///            .........
///            |   |   |
///        ... D   D   D ...
///             \ /     \
///        ...   S       ...
///
/// NOTE: This function generates the four-frequency variant of the heavy square code.
/// This function implements Fig 10.b left of the `paper <https://arxiv.org/abs/1907.09528>`_.
/// This function doesn't support the variant Fig 10.b right.
///
/// Note that if ``d`` is set to ``1`` a :class:`~rustworkx.PyGraph` with a
/// single node will be returned.
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyGraph` with a single node will be returned.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy square graph
/// :rtype: PyGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import graphviz_draw
///
///   graph = rustworkx.generators.heavy_square_graph(3)
///   graphviz_draw(graph, lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(d, /, multigraph=True)")]
pub fn heavy_square_graph(py: Python, d: usize, multigraph: bool) -> PyResult<graph::PyGraph> {
    let mut graph = StablePyGraph::<Undirected>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    if d == 1 {
        graph.add_node(py.None());
        return Ok(graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }

    let num_data = d * d;
    let num_syndrome = d * (d - 1);
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> = (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> = (0..num_flag).map(|_| graph.add_node(py.None())).collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, py.None());
            graph.add_edge(*flag, nodes_data[i * d + j + 1], py.None());
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            graph.add_edge(
                nodes_data[i * d + (d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            graph.add_edge(
                syndrome_chunk[syndrome_chunk.len() - 1],
                nodes_data[i * d + (2 * d - 1)],
                py.None(),
            );
        } else if i % 2 == 1 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], py.None());
            graph.add_edge(syndrome_chunk[0], nodes_data[(i + 1) * d], py.None());
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(nodes_flag[i * (d - 1) + j], *syndrome, py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j], py.None());
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(nodes_flag[i * (d - 1) + j - 1], *syndrome, py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j - 1], py.None());
                }
            }
        }
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an directed heavy square graph. Fig. 6 of
/// https://arxiv.org/abs/1907.09528.
/// An ASCII diagram of the graph is given by:
///
/// .. code-block:: console
///
///     ...       S   ...
///        \     / \
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///            .........
///            |   |   |
///        ... D   D   D ...
///             \ /     \
///        ...   S       ...
///
/// NOTE: This function generates the four-frequency variant of the heavy square code.
/// This function implements Fig 10.b left of the `paper <https://arxiv.org/abs/1907.09528>`_.
/// This function doesn't support the variant Fig 10.b right.
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyDiGraph` with a single node will be returned.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated directed heavy square graph
/// :rtype: PyDiGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import graphviz_draw
///
///   graph = rustworkx.generators.directed_heavy_square_graph(3)
///   graphviz_draw(graph, lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///
#[pyfunction(bidirectional = false, multigraph = true)]
#[pyo3(text_signature = "(d, /, bidirectional=False, multigraph=True)")]
pub fn directed_heavy_square_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StablePyGraph::<Directed>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    if d == 1 {
        graph.add_node(py.None());
        return Ok(digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }

    let num_data = d * d;
    let num_syndrome = d * (d - 1);
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> = (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> = (0..num_flag).map(|_| graph.add_node(py.None())).collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, py.None());
            graph.add_edge(*flag, nodes_data[i * d + j + 1], py.None());
            if bidirectional {
                graph.add_edge(*flag, nodes_data[i * d + j], py.None());
                graph.add_edge(nodes_data[i * d + j + 1], *flag, py.None());
            }
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            graph.add_edge(
                nodes_data[i * d + (d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            graph.add_edge(
                nodes_data[i * d + (2 * d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (d - 1)],
                    py.None(),
                );
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (2 * d - 1)],
                    py.None(),
                );
            }
        } else if i % 2 == 1 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], py.None());
            graph.add_edge(nodes_data[(i + 1) * d], syndrome_chunk[0], py.None());
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], py.None());
                graph.add_edge(syndrome_chunk[0], nodes_data[(i + 1) * d], py.None());
            }
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(*syndrome, nodes_flag[i * (d - 1) + j], py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j], py.None());
                    if bidirectional {
                        graph.add_edge(nodes_flag[i * (d - 1) + j], *syndrome, py.None());
                        graph.add_edge(nodes_flag[(i + 1) * (d - 1) + j], *syndrome, py.None());
                    }
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(*syndrome, nodes_flag[i * (d - 1) + j - 1], py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j - 1], py.None());
                    if bidirectional {
                        graph.add_edge(nodes_flag[i * (d - 1) + j - 1], *syndrome, py.None());
                        graph.add_edge(nodes_flag[(i + 1) * (d - 1) + j - 1], *syndrome, py.None());
                    }
                }
            }
        }
    }

    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected heavy hex graph. Fig. 2 of
/// https://arxiv.org/abs/1907.09528
/// An ASCII diagram of the graph is given by:
///
/// .. code-block:: text
///
///     ... D-S-D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///         .........
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///         .........
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///     ... D   D-S-D ...
///
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyGraph` with a single node will be returned.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy hex graph
/// :rtype: PyGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import graphviz_draw
///
///   graph = rustworkx.generators.heavy_hex_graph(3)
///   graphviz_draw(graph, lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(d, /, multigraph=True)")]
pub fn heavy_hex_graph(py: Python, d: usize, multigraph: bool) -> PyResult<graph::PyGraph> {
    let mut graph = StablePyGraph::<Undirected>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    if d == 1 {
        graph.add_node(py.None());
        return Ok(graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
            attrs: py.None(),
        });
    }

    let num_data = d * d;
    let num_syndrome = (d - 1) * (d + 1) / 2;
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> = (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> = (0..num_flag).map(|_| graph.add_node(py.None())).collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, py.None());
            graph.add_edge(*flag, nodes_data[i * d + j + 1], py.None());
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], py.None());
            graph.add_edge(syndrome_chunk[0], nodes_data[(i + 1) * d], py.None());
        } else if i % 2 == 1 {
            graph.add_edge(
                nodes_data[i * d + (d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            graph.add_edge(
                syndrome_chunk[syndrome_chunk.len() - 1],
                nodes_data[i * d + (2 * d - 1)],
                py.None(),
            );
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(
                        nodes_flag[i * (d - 1) + 2 * (j - 1) + 1],
                        *syndrome,
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * (j - 1) + 1],
                        py.None(),
                    );
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(nodes_flag[i * (d - 1) + 2 * j], *syndrome, py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + 2 * j], py.None());
                }
            }
        }
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed heavy hex graph. Fig. 2 of
/// https://arxiv.org/abs/1907.09528
/// An ASCII diagram of the graph is given by:
///
/// .. code-block:: text
///
///     ... D-S-D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///         .........
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///         .........
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ...-F   F-S-F ...
///         |   |   |
///     ... D   D   D ...
///         |   |   |
///     ... F-S-F   F-...
///         |   |   |
///     ... D   D-S-D ...
///
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyDiGraph` with a single node will be returned.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy hex directed graph
/// :rtype: PyDiGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import graphviz_draw
///
///   graph = rustworkx.generators.directed_heavy_hex_graph(3)
///   graphviz_draw(graph, lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///
#[pyfunction(bidirectional = false, multigraph = true)]
#[pyo3(text_signature = "(d, /, bidirectional=False, multigraph=True)")]
pub fn directed_heavy_hex_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StablePyGraph::<Directed>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    if d == 1 {
        graph.add_node(py.None());
        return Ok(digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        });
    }

    let num_data = d * d;
    let num_syndrome = (d - 1) * (d + 1) / 2;
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> = (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> = (0..num_flag).map(|_| graph.add_node(py.None())).collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, py.None());
            graph.add_edge(nodes_data[i * d + j + 1], *flag, py.None());
            if bidirectional {
                graph.add_edge(*flag, nodes_data[i * d + j], py.None());
                graph.add_edge(*flag, nodes_data[i * d + j + 1], py.None());
            }
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], py.None());
            graph.add_edge(nodes_data[(i + 1) * d], syndrome_chunk[0], py.None());
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], py.None());
                graph.add_edge(syndrome_chunk[0], nodes_data[(i + 1) * d], py.None());
            }
        } else if i % 2 == 1 {
            graph.add_edge(
                nodes_data[i * d + (d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            graph.add_edge(
                nodes_data[i * d + (2 * d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (d - 1)],
                    py.None(),
                );
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (2 * d - 1)],
                    py.None(),
                );
            }
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + 2 * (j - 1) + 1],
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * (j - 1) + 1],
                        py.None(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + 2 * (j - 1) + 1],
                            *syndrome,
                            py.None(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + 2 * (j - 1) + 1],
                            *syndrome,
                            py.None(),
                        );
                    }
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(*syndrome, nodes_flag[i * (d - 1) + 2 * j], py.None());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + 2 * j], py.None());
                    if bidirectional {
                        graph.add_edge(nodes_flag[i * (d - 1) + 2 * j], *syndrome, py.None());
                        graph.add_edge(nodes_flag[(i + 1) * (d - 1) + 2 * j], *syndrome, py.None());
                    }
                }
            }
        }
    }

    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected hexagonal lattice graph.
///
/// :param int rows: The number of rows to generate the graph with.
/// :param int cols: The number of columns to generate the graph with.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated hexagonal lattice graph.
///
/// :rtype: PyGraph
/// :raises TypeError: If either ``rows`` or ``cols`` are
///      not specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.hexagonal_lattice_graph(2, 2)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(rows, cols, /, multigraph=True)")]
pub fn hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    multigraph: bool,
) -> graph::PyGraph {
    let mut graph = StablePyGraph::<Undirected>::default();

    if rows == 0 || cols == 0 {
        return graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
            attrs: py.None(),
        };
    }

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let nodes: Vec<NodeIndex> = (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

    // Add column edges
    // first column
    for j in 0..(rowlen - 2) {
        graph.add_edge(nodes[j], nodes[j + 1], py.None());
    }

    for i in 1..(collen - 1) {
        for j in 0..(rowlen - 1) {
            graph.add_edge(nodes[i * rowlen + j - 1], nodes[i * rowlen + j], py.None());
        }
    }

    // last column
    for j in 0..(rowlen - 2) {
        graph.add_edge(
            nodes[(collen - 1) * rowlen + j - 1],
            nodes[(collen - 1) * rowlen + j],
            py.None(),
        );
    }

    // Add row edges
    for j in (0..(rowlen - 1)).step_by(2) {
        graph.add_edge(nodes[j], nodes[j + rowlen - 1], py.None());
    }

    for i in 1..(collen - 2) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j - 1],
                    nodes[(i + 1) * rowlen + j - 1],
                    py.None(),
                );
            }
        }
    }

    if collen > 2 {
        for j in ((collen % 2)..rowlen).step_by(2) {
            graph.add_edge(
                nodes[(collen - 2) * rowlen + j - 1],
                nodes[(collen - 1) * rowlen + j - 1 - (collen % 2)],
                py.None(),
            );
        }
    }

    graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    }
}

/// Generate a directed hexagonal lattice graph. The edges propagate towards  
///     right and bottom direction if ``bidirectional`` is ``false``
///
/// :param int rows: The number of rows to generate the graph with.
/// :param int cols: The number of rows to generate the graph with.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated directed hexagonal lattice graph.
///
/// :rtype: PyDiGraph
/// :raises TypeError: If either ``rows`` or ``cols`` are
///      not specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.directed_hexagonal_lattice_graph(2, 3)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(text_signature = "(rows, cols, /, bidirectional=False, multigraph=True)")]
pub fn directed_hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    bidirectional: bool,
    multigraph: bool,
) -> digraph::PyDiGraph {
    let mut graph = StablePyGraph::<Directed>::default();

    if rows == 0 || cols == 0 {
        return digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
            attrs: py.None(),
        };
    }

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let nodes: Vec<NodeIndex> = (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

    // Add column edges
    // first column
    for j in 0..(rowlen - 2) {
        graph.add_edge(nodes[j], nodes[j + 1], py.None());
        if bidirectional {
            graph.add_edge(nodes[j + 1], nodes[j], py.None());
        }
    }

    for i in 1..(collen - 1) {
        for j in 0..(rowlen - 1) {
            graph.add_edge(nodes[i * rowlen + j - 1], nodes[i * rowlen + j], py.None());
            if bidirectional {
                graph.add_edge(nodes[i * rowlen + j], nodes[i * rowlen + j - 1], py.None());
            }
        }
    }

    // last column
    for j in 0..(rowlen - 2) {
        graph.add_edge(
            nodes[(collen - 1) * rowlen + j - 1],
            nodes[(collen - 1) * rowlen + j],
            py.None(),
        );
        if bidirectional {
            graph.add_edge(
                nodes[(collen - 1) * rowlen + j],
                nodes[(collen - 1) * rowlen + j - 1],
                py.None(),
            );
        }
    }

    // Add row edges
    for j in (0..(rowlen - 1)).step_by(2) {
        graph.add_edge(nodes[j], nodes[j + rowlen - 1], py.None());
        if bidirectional {
            graph.add_edge(nodes[j + rowlen - 1], nodes[j], py.None());
        }
    }

    for i in 1..(collen - 2) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j - 1],
                    nodes[(i + 1) * rowlen + j - 1],
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        nodes[(i + 1) * rowlen + j - 1],
                        nodes[i * rowlen + j - 1],
                        py.None(),
                    );
                }
            }
        }
    }

    if collen > 2 {
        for j in ((collen % 2)..rowlen).step_by(2) {
            graph.add_edge(
                nodes[(collen - 2) * rowlen + j - 1],
                nodes[(collen - 1) * rowlen + j - 1 - (collen % 2)],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(
                    nodes[(collen - 1) * rowlen + j - 1 - (collen % 2)],
                    nodes[(collen - 2) * rowlen + j - 1],
                    py.None(),
                );
            }
        }
    }

    digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    }
}

/// Generate an undirected lollipop graph where a mesh graph is connected to a
/// path.
///
/// If neither ``num_path_nodes`` nor ``path_weights`` (both described
/// below) are specified then this is equivalent to
/// :func:`~rustworkx.generators.mesh_graph`
///
/// :param int num_mesh_nodes: The number of nodes to generate the mesh graph
///     with. Node weights will be None if this is specified. If both
///     ``num_mesh_nodes`` and ``mesh_weights`` are set this will be ignored and
///     ``mesh_weights`` will be used.
/// :param int num_path_nodes: The number of nodes to generate the path
///     with. Node weights will be None if this is specified. If both
///     ``num_path_nodes`` and ``path_weights`` are set this will be ignored and
///     ``path_weights`` will be used.
/// :param list mesh_weights: A list of node weights for the mesh graph. If both
///     ``num_mesh_nodes`` and ``mesh_weights`` are set ``num_mesh_nodes`` will
///     be ignored and ``mesh_weights`` will be used.
/// :param list path_weights: A list of node weights for the path. If both
///     ``num_path_nodes`` and ``path_weights`` are set ``num_path_nodes`` will
///     be ignored and ``path_weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated lollipop graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_mesh_nodes`` or ``mesh_weights`` are specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.lollipop_graph(4, 2)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(
    text_signature = "(/, num_mesh_nodes=None, num_path_nodes=None, mesh_weights=None, path_weights=None, multigraph=True)"
)]
pub fn lollipop_graph(
    py: Python,
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    mesh_weights: Option<Vec<PyObject>>,
    path_weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = mesh_graph(py, num_mesh_nodes, mesh_weights, multigraph)?;
    if num_path_nodes.is_none() && path_weights.is_none() {
        return Ok(graph);
    }
    let meshlen = graph.num_nodes();

    let path_nodes: Vec<NodeIndex> = match path_weights {
        Some(path_weights) => path_weights
            .into_iter()
            .map(|node| graph.graph.add_node(node))
            .collect(),
        None => (0..num_path_nodes.unwrap())
            .map(|_| graph.graph.add_node(py.None()))
            .collect(),
    };

    let pathlen = path_nodes.len();
    if pathlen > 0 {
        graph.graph.add_edge(
            NodeIndex::new(meshlen - 1),
            NodeIndex::new(meshlen),
            py.None(),
        );
        for (node_a, node_b) in pairwise(path_nodes) {
            match node_a {
                Some(node_a) => graph.graph.add_edge(node_a, node_b, py.None()),
                None => continue,
            };
        }
    }
    Ok(graph)
}

/// Generate a generalized Petersen graph :math:`G(n, k)` with :math:`2n`
/// nodes and :math:`3n` edges. See Watkins [1]_ for more details.
///
/// .. note::
///   
///   The Petersen graph itself is denoted :math:`G(5, 2)`
///
/// :param int n: number of nodes in the internal star and external regular polygon.
/// :param int k: shift that changes the internal star graph.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated generalized Petersen graph.
///
/// :rtype: PyGraph
/// :raises IndexError: If either ``n`` or ``k`` are
///      not valid
/// :raises TypeError: If either ``n`` or ``k`` are
///      not non-negative integers
///
/// .. jupyter-execute::
///   
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///   
///   # Petersen Graph is G(5, 2)
///   graph = rustworkx.generators.generalized_petersen_graph(5, 2)
///   layout = rustworkx.shell_layout(graph, nlist=[[0, 1, 2, 3, 4],[6, 7, 8, 9, 5]])
///   mpl_draw(graph, pos=layout)
///   
/// .. jupyter-execute::
///   
///   # Möbius–Kantor Graph is G(8, 3)
///   graph = rustworkx.generators.generalized_petersen_graph(8, 3)
///   layout = rustworkx.shell_layout(
///     graph, nlist=[[0, 1, 2, 3, 4, 5, 6, 7], [10, 11, 12, 13, 14, 15, 8, 9]]
///   )
///   mpl_draw(graph, pos=layout)
///
/// .. [1] Watkins, Mark E.
///    "A theorem on tait colorings with an application to the generalized Petersen graphs"
///    Journal of Combinatorial Theory 6 (2), 152–164 (1969).
///    https://doi.org/10.1016/S0021-9800(69)80116-X
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(n, k, /, multigraph=True)")]
pub fn generalized_petersen_graph(
    py: Python,
    n: usize,
    k: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if n < 3 {
        return Err(PyIndexError::new_err("n must be at least 3"));
    }

    if k == 0 || 2 * k >= n {
        return Err(PyIndexError::new_err(
            "k is invalid: it must be positive and less than n/2",
        ));
    }

    let mut graph = StablePyGraph::<Undirected>::with_capacity(2 * n, 3 * n);

    let star_nodes: Vec<NodeIndex> = (0..n).map(|_| graph.add_node(py.None())).collect();

    let polygon_nodes: Vec<NodeIndex> = (0..n).map(|_| graph.add_node(py.None())).collect();

    for i in 0..n {
        graph.add_edge(star_nodes[i], star_nodes[(i + k) % n], py.None());
    }

    for i in 0..n {
        graph.add_edge(polygon_nodes[i], polygon_nodes[(i + 1) % n], py.None());
    }

    for i in 0..n {
        graph.add_edge(polygon_nodes[i], star_nodes[i], py.None());
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected barbell graph where two identical mesh graphs are
/// connected by a path.
///
/// If ``num_path_nodes`` (described below) is not specified then this is
/// equivalent to two mesh graphs joined together.
///
/// :param int num_mesh_nodes: The number of nodes to generate the mesh graphs
///     with. Node weights will be None if this is specified. If both
///     ``num_mesh_nodes`` and ``mesh_weights`` are set this will be ignored and
///     ``mesh_weights`` will be used.
/// :param int num_path_nodes: The number of nodes to generate the path
///     with. Node weights will be None if this is specified. If both
///     ``num_path_nodes`` and ``path_weights`` are set this will be ignored and
///     ``path_weights`` will be used.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated barbell graph
/// :rtype: PyGraph
/// :raises IndexError: If ``num_mesh_nodes`` is not specified
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph = rustworkx.generators.barbell_graph(4, 2)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(/, num_mesh_nodes=None, num_path_nodes=None, multigraph=True)")]
pub fn barbell_graph(
    py: Python,
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    if num_mesh_nodes.is_none() {
        return Err(PyIndexError::new_err("num_mesh_nodes not specified"));
    }

    let mut left_mesh = StableUnGraph::<PyObject, PyObject>::default();
    let mesh_nodes: Vec<NodeIndex> = (0..num_mesh_nodes.unwrap())
        .map(|_| left_mesh.add_node(py.None()))
        .collect();
    let mut nodelen = mesh_nodes.len();
    for i in 0..nodelen - 1 {
        for j in i + 1..nodelen {
            left_mesh.add_edge(mesh_nodes[i], mesh_nodes[j], py.None());
        }
    }

    let right_mesh = left_mesh.clone();

    if let Some(num_nodes) = num_path_nodes {
        let path_nodes: Vec<NodeIndex> = (0..num_nodes)
            .map(|_| left_mesh.add_node(py.None()))
            .collect();
        left_mesh.add_edge(
            NodeIndex::new(nodelen - 1),
            NodeIndex::new(nodelen),
            py.None(),
        );

        nodelen += path_nodes.len();

        for (node_a, node_b) in pairwise(path_nodes) {
            match node_a {
                Some(node_a) => left_mesh.add_edge(node_a, node_b, py.None()),
                None => continue,
            };
        }
    }

    for node in right_mesh.node_indices() {
        let new_node = &right_mesh[node];
        left_mesh.add_node(new_node.clone_ref(py));
    }
    left_mesh.add_edge(
        NodeIndex::new(nodelen - 1),
        NodeIndex::new(nodelen),
        py.None(),
    );
    for edge in right_mesh.edge_references() {
        let new_source = NodeIndex::new(nodelen + edge.source().index());
        let new_target = NodeIndex::new(nodelen + edge.target().index());
        let weight = edge.weight();
        left_mesh.add_edge(new_source, new_target, weight.clone_ref(py));
    }

    Ok(graph::PyGraph {
        graph: left_mesh,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

#[pymodule]
pub fn generators(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cycle_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_cycle_graph))?;
    m.add_wrapped(wrap_pyfunction!(path_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_path_graph))?;
    m.add_wrapped(wrap_pyfunction!(star_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_star_graph))?;
    m.add_wrapped(wrap_pyfunction!(mesh_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_mesh_graph))?;
    m.add_wrapped(wrap_pyfunction!(grid_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_grid_graph))?;
    m.add_wrapped(wrap_pyfunction!(heavy_square_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_heavy_square_graph))?;
    m.add_wrapped(wrap_pyfunction!(heavy_hex_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_heavy_hex_graph))?;
    m.add_wrapped(wrap_pyfunction!(binomial_tree_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_binomial_tree_graph))?;
    m.add_wrapped(wrap_pyfunction!(hexagonal_lattice_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_hexagonal_lattice_graph))?;
    m.add_wrapped(wrap_pyfunction!(lollipop_graph))?;
    m.add_wrapped(wrap_pyfunction!(full_rary_tree))?;
    m.add_wrapped(wrap_pyfunction!(generalized_petersen_graph))?;
    m.add_wrapped(wrap_pyfunction!(barbell_graph))?;
    Ok(())
}
