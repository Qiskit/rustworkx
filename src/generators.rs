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

use std::iter;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::{StableDiGraph, StableUnGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use super::digraph;
use super::graph;

pub fn pairwise<I>(right: I) -> impl Iterator<Item = (Option<I::Item>, I::Item)>
where
    I: IntoIterator + Clone,
{
    let left = iter::once(None).chain(right.clone().into_iter().map(Some));
    left.zip(right)
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated cycle graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_cycle_graph(5)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len: usize;
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            node_len = weights.len();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => {
            node_len = num_nodes.unwrap();
            (0..num_nodes.unwrap())
                .map(|_| graph.add_node(py.None()))
                .collect()
        }
    };
    for (node_a, node_b) in pairwise(nodes) {
        match node_a {
            Some(node_a) => {
                if bidirectional {
                    graph.add_edge(node_b, node_a, py.None());
                }
                graph.add_edge(node_a, node_b, py.None());
            }
            None => continue,
        };
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated cycle graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.cycle_graph(5)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let node_len: usize;
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            node_len = weights.len();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => {
            node_len = num_nodes.unwrap();
            (0..num_nodes.unwrap())
                .map(|_| graph.add_node(py.None()))
                .collect()
        }
    };
    for (node_a, node_b) in pairwise(nodes) {
        match node_a {
            Some(node_a) => graph.add_edge(node_a, node_b, py.None()),
            None => continue,
        };
    }
    let last_node_index = NodeIndex::new(node_len - 1);
    let first_node_index = NodeIndex::new(0);
    graph.add_edge(last_node_index, first_node_index, py.None());
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated path graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_path_graph(10)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };
    for (node_a, node_b) in pairwise(nodes) {
        match node_a {
            Some(node_a) => {
                graph.add_edge(node_a, node_b, py.None());
                if bidirectional {
                    graph.add_edge(node_b, node_a, py.None());
                }
            }
            None => continue,
        };
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated path graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.path_graph(10)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };
    for (node_a, node_b) in pairwise(nodes) {
        match node_a {
            Some(node_a) => graph.add_edge(node_a, node_b, py.None()),
            None => continue,
        };
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated star graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_star_graph(10)
///   mpl_draw(graph)
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_star_graph(10, inward=True)
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
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };
    for node in nodes[1..].iter() {
        //Add edges in both directions if bidirection is True
        if bidirectional {
            graph.add_edge(*node, nodes[0], py.None());
            graph.add_edge(nodes[0], *node, py.None());
        } else if inward {
            graph.add_edge(*node, nodes[0], py.None());
        } else {
            graph.add_edge(nodes[0], *node, py.None());
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated star graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.star_graph(10)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };
    for node in nodes[1..].iter() {
        graph.add_edge(nodes[0], *node, py.None());
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated mesh graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.mesh_graph(4)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };

    let nodelen = nodes.len();
    for i in 0..nodelen - 1 {
        for j in i + 1..nodelen {
            graph.add_edge(nodes[i], nodes[j], py.None());
        }
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated mesh graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_mesh_graph(4)
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
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    if weights.is_none() && num_nodes.is_none() {
        return Err(PyIndexError::new_err(
            "num_nodes and weights list not specified",
        ));
    }
    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes.unwrap())
            .map(|_| graph.add_node(py.None()))
            .collect(),
    };
    let nodelen = nodes.len();
    for i in 0..nodelen - 1 {
        for j in i + 1..nodelen {
            graph.add_edge(nodes[i], nodes[j], py.None());
            graph.add_edge(nodes[j], nodes[i], py.None());
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
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
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.grid_graph(2, 3)
///   mpl_draw(graph)
///
#[pyfunction(multigraph = true)]
#[pyo3(
    text_signature = "(/, rows=None, cols=None, weights=None, multigraph=True)"
)]
pub fn grid_graph(
    py: Python,
    rows: Option<usize>,
    cols: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    if weights.is_none() && (rows.is_none() || cols.is_none()) {
        return Err(PyIndexError::new_err(
            "dimensions and weights list not specified",
        ));
    }

    let mut rowlen = rows.unwrap_or(0);
    let mut collen = cols.unwrap_or(0);
    let mut num_nodes = rowlen * collen;

    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
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
                let index = graph.add_node(weight);
                node_list.push(index);
                node_cnt -= 1;
            }
            for _i in 0..node_cnt {
                let index = graph.add_node(py.None());
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes).map(|_| graph.add_node(py.None())).collect(),
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                graph.add_edge(
                    nodes[i * collen + j],
                    nodes[(i + 1) * collen + j],
                    py.None(),
                );
            }
            if j + 1 < collen {
                graph.add_edge(
                    nodes[i * collen + j],
                    nodes[i * collen + j + 1],
                    py.None(),
                );
            }
        }
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
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
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_grid_graph(2, 3)
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
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    if weights.is_none() && (rows.is_none() || cols.is_none()) {
        return Err(PyIndexError::new_err(
            "dimensions and weights list not specified",
        ));
    }

    let mut rowlen = rows.unwrap_or(0);
    let mut collen = cols.unwrap_or(0);
    let mut num_nodes = rowlen * collen;

    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
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
                let index = graph.add_node(weight);
                node_list.push(index);
                node_cnt -= 1;
            }
            for _i in 0..node_cnt {
                let index = graph.add_node(py.None());
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes).map(|_| graph.add_node(py.None())).collect(),
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                graph.add_edge(
                    nodes[i * collen + j],
                    nodes[(i + 1) * collen + j],
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        nodes[(i + 1) * collen + j],
                        nodes[i * collen + j],
                        py.None(),
                    );
                }
            }

            if j + 1 < collen {
                graph.add_edge(
                    nodes[i * collen + j],
                    nodes[i * collen + j + 1],
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        nodes[i * collen + j + 1],
                        nodes[i * collen + j],
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
    })
}

/// Generate an undirected binomial tree of order n recursively.
///
/// :param int order: Order of the binomial tree.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order extra nodes with with None will be appended.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A binomial tree with 2^n vertices and 2^n - 1 edges.
/// :rtype: PyGraph
/// :raises IndexError: If the lenght of ``weights`` is greater that 2^n
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.binomial_tree_graph(4)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();

    let num_nodes = usize::pow(2, order);

    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();

            let mut node_count = num_nodes;

            if weights.len() > num_nodes {
                return Err(PyIndexError::new_err(
                    "weights should be <= 2**order",
                ));
            }

            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
                node_count -= 1;
            }

            for _i in 0..node_count {
                let index = graph.add_node(py.None());
                node_list.push(index);
            }

            node_list
        }

        None => (0..num_nodes).map(|_| graph.add_node(py.None())).collect(),
    };

    let mut n = 1;

    for _ in 0..order {
        let edges: Vec<(NodeIndex, NodeIndex)> = graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect();

        for (source, target) in edges {
            let source_index = source.index();
            let target_index = target.index();

            graph.add_edge(
                nodes[source_index + n],
                nodes[target_index + n],
                py.None(),
            );
        }

        graph.add_edge(nodes[0], nodes[n], py.None());

        n *= 2;
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
    })
}

/// Generate an undirected binomial tree of order n recursively.
/// The edges propagate towards right and bottom direction if ``bidirectional`` is ``false``
///
/// :param int order: Order of the binomial tree.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order extra nodes with None will be appended.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A directed binomial tree with 2^n vertices and 2^n - 1 edges.
/// :rtype: PyDiGraph
/// :raises IndexError: If the lenght of ``weights`` is greater that 2^n
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_binomial_tree_graph(4)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(order, /,  weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_binomial_tree_graph(
    py: Python,
    order: u32,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();

    let num_nodes = usize::pow(2, order);

    let nodes: Vec<NodeIndex> = match weights {
        Some(weights) => {
            let mut node_list: Vec<NodeIndex> = Vec::new();
            let mut node_count = num_nodes;

            if weights.len() > num_nodes {
                return Err(PyIndexError::new_err(
                    "weights should be <= 2**order",
                ));
            }

            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
                node_count -= 1;
            }

            for _i in 0..node_count {
                let index = graph.add_node(py.None());
                node_list.push(index);
            }

            node_list
        }

        None => (0..num_nodes).map(|_| graph.add_node(py.None())).collect(),
    };

    let mut n = 1;

    for _ in 0..order {
        let edges: Vec<(NodeIndex, NodeIndex)> = graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect();

        for (source, target) in edges {
            let source_index = source.index();
            let target_index = target.index();

            if graph
                .find_edge(nodes[source_index + n], nodes[target_index + n])
                .is_none()
            {
                graph.add_edge(
                    nodes[source_index + n],
                    nodes[target_index + n],
                    py.None(),
                );
            }

            if bidirectional
                && graph
                    .find_edge(nodes[target_index + n], nodes[source_index + n])
                    .is_none()
            {
                graph.add_edge(
                    nodes[target_index + n],
                    nodes[source_index + n],
                    py.None(),
                );
            }
        }

        if graph.find_edge(nodes[0], nodes[n]).is_none() {
            graph.add_edge(nodes[0], nodes[n], py.None());
        }

        if bidirectional && graph.find_edge(nodes[n], nodes[0]).is_none() {
            graph.add_edge(nodes[n], nodes[0], py.None());
        }

        n *= 2;
    }

    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
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
/// This function implements Fig 10.b left of the [paper](https://arxiv.org/abs/1907.09528).
/// This function doesn't support the variant Fig 10.b right.
///
/// :param int d: distance of the code.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy square graph
/// :rtype: PyGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.heavy_square_graph(3)
///   dot_str = graph.to_dot(
///       lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///   dot = pydot.graph_from_dot_data(dot_str)[0]
///
///   with tempfile.TemporaryDirectory() as tmpdirname:
///       tmp_path = os.path.join(tmpdirname, 'dag.png')
///       dot.write_png(tmp_path)
///       image = Image.open(tmp_path)
///       os.remove(tmp_path)
///   image
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(d, /, multigraph=True)")]
pub fn heavy_square_graph(
    py: Python,
    d: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    let num_data = d * d;
    let num_syndrome = d * (d - 1);
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> =
        (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> =
        (0..num_flag).map(|_| graph.add_node(py.None())).collect();

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
            graph.add_edge(
                syndrome_chunk[0],
                nodes_data[(i + 1) * d],
                py.None(),
            );
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(
                        nodes_flag[i * (d - 1) + j],
                        *syndrome,
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + j],
                        py.None(),
                    );
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(
                        nodes_flag[i * (d - 1) + j - 1],
                        *syndrome,
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + j - 1],
                        py.None(),
                    );
                }
            }
        }
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
/// This function implements Fig 10.b left of the [paper](https://arxiv.org/abs/1907.09528).
/// This function doesn't support the variant Fig 10.b right.
///
/// :param int d: distance of the code.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated directed heavy square graph
/// :rtype: PyDiGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.heavy_square_graph(3)
///   dot_str = graph.to_dot(
///       lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///   dot = pydot.graph_from_dot_data(dot_str)[0]
///
///   with tempfile.TemporaryDirectory() as tmpdirname:
///       tmp_path = os.path.join(tmpdirname, 'dag.png')
///       dot.write_png(tmp_path)
///       image = Image.open(tmp_path)
///       os.remove(tmp_path)
///   image
///
#[pyfunction(bidirectional = false, multigraph = true)]
#[pyo3(text_signature = "(d, /, bidirectional=False, multigraph=True)")]
pub fn directed_heavy_square_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    let num_data = d * d;
    let num_syndrome = d * (d - 1);
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> =
        (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> =
        (0..num_flag).map(|_| graph.add_node(py.None())).collect();

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
            graph.add_edge(
                nodes_data[(i + 1) * d],
                syndrome_chunk[0],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], py.None());
                graph.add_edge(
                    syndrome_chunk[0],
                    nodes_data[(i + 1) * d],
                    py.None(),
                );
            }
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + j],
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + j],
                        py.None(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + j],
                            *syndrome,
                            py.None(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + j],
                            *syndrome,
                            py.None(),
                        );
                    }
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + j - 1],
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + j - 1],
                        py.None(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + j - 1],
                            *syndrome,
                            py.None(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + j - 1],
                            *syndrome,
                            py.None(),
                        );
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
/// :param int d: distance of the code.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy hex graph
/// :rtype: PyGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.heavy_hex_graph(3)
///   dot_str = graph.to_dot(
///       lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///   dot = pydot.graph_from_dot_data(dot_str)[0]
///
///   with tempfile.TemporaryDirectory() as tmpdirname:
///       tmp_path = os.path.join(tmpdirname, 'dag.png')
///       dot.write_png(tmp_path)
///       image = Image.open(tmp_path)
///       os.remove(tmp_path)
///   image
///
#[pyfunction(multigraph = true)]
#[pyo3(text_signature = "(d, /, multigraph=True)")]
pub fn heavy_hex_graph(
    py: Python,
    d: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    let num_data = d * d;
    let num_syndrome = (d - 1) * (d + 1) / 2;
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> =
        (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> =
        (0..num_flag).map(|_| graph.add_node(py.None())).collect();

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
            graph.add_edge(
                syndrome_chunk[0],
                nodes_data[(i + 1) * d],
                py.None(),
            );
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
                    graph.add_edge(
                        nodes_flag[i * (d - 1) + 2 * j],
                        *syndrome,
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * j],
                        py.None(),
                    );
                }
            }
        }
    }

    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
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
/// :param int d: distance of the code.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated heavy hex directed graph
/// :rtype: PyDiGraph
/// :raises IndexError: If d is even.
///
/// .. jupyter-execute::
///
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.heavy_hex_graph(3)
///   dot_str = graph.to_dot(
///       lambda node: dict(
///           color='black', fillcolor='lightblue', style='filled'))
///   dot = pydot.graph_from_dot_data(dot_str)[0]
///
///   with tempfile.TemporaryDirectory() as tmpdirname:
///       tmp_path = os.path.join(tmpdirname, 'dag.png')
///       dot.write_png(tmp_path)
///       image = Image.open(tmp_path)
///       os.remove(tmp_path)
///   image
///
#[pyfunction(bidirectional = false, multigraph = true)]
#[pyo3(text_signature = "(d, /, bidirectional=False, multigraph=True)")]
pub fn directed_heavy_hex_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();

    if d % 2 == 0 {
        return Err(PyIndexError::new_err("d must be odd"));
    }

    let num_data = d * d;
    let num_syndrome = (d - 1) * (d + 1) / 2;
    let num_flag = d * (d - 1);

    let nodes_data: Vec<NodeIndex> =
        (0..num_data).map(|_| graph.add_node(py.None())).collect();
    let nodes_syndrome: Vec<NodeIndex> = (0..num_syndrome)
        .map(|_| graph.add_node(py.None()))
        .collect();
    let nodes_flag: Vec<NodeIndex> =
        (0..num_flag).map(|_| graph.add_node(py.None())).collect();

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
            graph.add_edge(
                nodes_data[(i + 1) * d],
                syndrome_chunk[0],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], py.None());
                graph.add_edge(
                    syndrome_chunk[0],
                    nodes_data[(i + 1) * d],
                    py.None(),
                );
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
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + 2 * j],
                        py.None(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * j],
                        py.None(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + 2 * j],
                            *syndrome,
                            py.None(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + 2 * j],
                            *syndrome,
                            py.None(),
                        );
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
    })
}

/// Generate an undirected hexagonal lattice graph.
///
/// :param int rows: The number of rows to generate the graph with.
/// :param int cols: The number of columns to generate the graph with.
/// :param bool multigraph: When set to False the output
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
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
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.hexagonal_lattice_graph(2, 2)
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
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();

    if rows == 0 || cols == 0 {
        return graph::PyGraph {
            graph,
            node_removed: false,
            multigraph,
        };
    }

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let nodes: Vec<NodeIndex> =
        (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

    // Add column edges
    // first column
    for j in 0..(rowlen - 2) {
        graph.add_edge(nodes[j], nodes[j + 1], py.None());
    }

    for i in 1..(collen - 1) {
        for j in 0..(rowlen - 1) {
            graph.add_edge(
                nodes[i * rowlen + j - 1],
                nodes[i * rowlen + j],
                py.None(),
            );
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
///     :class:`~retworkx.PyDiGraph` object will not be not be a multigraph and
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
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph = retworkx.generators.directed_hexagonal_lattice_graph(2, 3)
///   mpl_draw(graph)
///
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[pyo3(
    text_signature = "(rows, cols, /, bidirectional=False, multigraph=True)"
)]
pub fn directed_hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    bidirectional: bool,
    multigraph: bool,
) -> digraph::PyDiGraph {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();

    if rows == 0 || cols == 0 {
        return digraph::PyDiGraph {
            graph,
            node_removed: false,
            check_cycle: false,
            cycle_state: algo::DfsSpace::default(),
            multigraph,
        };
    }

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let nodes: Vec<NodeIndex> =
        (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

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
            graph.add_edge(
                nodes[i * rowlen + j - 1],
                nodes[i * rowlen + j],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(
                    nodes[i * rowlen + j],
                    nodes[i * rowlen + j - 1],
                    py.None(),
                );
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
    }
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
    Ok(())
}
