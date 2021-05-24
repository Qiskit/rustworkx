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

fn pairwise<I>(right: I) -> impl Iterator<Item = (Option<I::Item>, I::Item)>
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
///     :class:`~retworkx.PyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated cycle graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
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
///   graph = retworkx.generators.directed_cycle_graph(5)
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
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"]
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
        multigraph: multigraph,
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.cycle_graph(5)
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
#[text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"]
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
///     :class:`~retworkx.DiPyGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated path graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
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
///   graph = retworkx.generators.directed_path_graph(10)
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
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"]
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
        multigraph: multigraph,
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.path_graph(10)
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
#[text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"]
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
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated star graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
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
///   graph = retworkx.generators.directed_star_graph(10)
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
///   graph = retworkx.generators.directed_star_graph(10, inward=True)
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
#[pyfunction(inward = "false", bidirectional = "false", multigraph = "true")]
#[text_signature = "(/, num_nodes=None, weights=None, inward=False, bidirectional=False, multigraph=True)"]
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
        multigraph: multigraph,
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.star_graph(10)
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
#[text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"]
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.mesh_graph(4)
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
#[text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"]
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
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated mesh graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
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
///   graph = retworkx.generators.directed_mesh_graph(4)
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
#[pyfunction(multigraph = "true")]
#[text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"]
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
        multigraph: multigraph,
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.grid_graph(2, 3)
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
#[text_signature = "(/, rows=None, cols=None, weights=None, multigraph=True)"]
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
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated grid graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``rows`` or ``cols`` and ``weights`` are
///      specified
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
///   graph = retworkx.generators.directed_grid_graph(2, 3)
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
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[text_signature = "(/, rows=None, cols=None, weights=None, bidirectional=False, multigraph=True)"]
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
        multigraph: multigraph,
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
///   import os
///   import tempfile
///
///   import pydot
///   from PIL import Image
///
///   import retworkx.generators
///
///   graph = retworkx.generators.binomial_tree_graph(4)
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
#[text_signature = "(order, /, weights=None, multigraph=True)"]
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
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: A directed binomial tree with 2^n vertices and 2^n - 1 edges.
/// :rtype: PyDiGraph
/// :raises IndexError: If the lenght of ``weights`` is greater that 2^n
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
///   graph = retworkx.generators.directed_binomial_tree_graph(4)
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
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[text_signature = "(order, /,  weights=None, bidirectional=False, multigraph=True)"]
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
        multigraph: multigraph,
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
/// :returns: The generated hexagonal lattice graph with the following nodes deleted:
///           ```2*row + 1``` and
///           ```(col) * (2*row + 2) + (2*row + 1) * ((col) % 2)```
///
/// :rtype: PyGraph
/// :raises IndexError: If neither ``rows`` or ``cols`` are
///      specified
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
///   graph = retworkx.generators.hexagonal_lattice_graph(2, 2)
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
#[text_signature = "(/, rows=None, cols=None, multigraph=True)"]
pub fn hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen;

    let nodes: Vec<NodeIndex> =
        (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

    // Add column edges
    for i in 0..collen {
        for j in 0..(rowlen - 1) {
            graph.add_edge(
                nodes[i * rowlen + j],
                nodes[i * rowlen + (j + 1)],
                py.None(),
            );
        }
    }

    // Add row edges
    for i in 0..(collen - 1) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j],
                    nodes[(i + 1) * rowlen + j],
                    py.None(),
                );
            }
        }
    }

    // Remove corner nodes
    graph.remove_node(nodes[rowlen - 1]);
    graph.remove_node(
        nodes[(collen - 1) * rowlen + (rowlen - 1) * ((collen - 1) % 2)],
    );

    Ok(graph::PyGraph {
        graph,
        node_removed: true,
        multigraph,
    })
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
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated directed hexagonal lattice graph with the following nodes deleted:
///           ```2*row + 1``` and
///           ```(col) * (2*row + 2) + (2*row + 1) * ((col) % 2)```
///
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``rows`` or ``cols`` are
///      specified
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
///   graph = retworkx.generators.directed_hexagonal_lattice_graph(2, 3)
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
#[pyfunction(bidirectional = "false", multigraph = "true")]
#[text_signature = "(/, rows=None, cols=None, bidirectional=False, multigraph=True)"]
pub fn directed_hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();

    let mut rowlen = rows;
    let mut collen = cols;
    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen;

    let nodes: Vec<NodeIndex> =
        (0..num_nodes).map(|_| graph.add_node(py.None())).collect();

    // Add column edges
    for i in 0..collen {
        for j in 0..(rowlen - 1) {
            graph.add_edge(
                nodes[i * rowlen + j],
                nodes[i * rowlen + (j + 1)],
                py.None(),
            );
            if bidirectional {
                graph.add_edge(
                    nodes[i * rowlen + (j + 1)],
                    nodes[i * rowlen + j],
                    py.None(),
                );
            }
        }
    }

    // Add row edges
    for i in 0..(collen - 1) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j],
                    nodes[(i + 1) * rowlen + j],
                    py.None(),
                );
                if bidirectional {
                    graph.add_edge(
                        nodes[(i + 1) * rowlen + j],
                        nodes[i * rowlen + j],
                        py.None(),
                    );
                }
            }
        }
    }

    // Remove corner nodes
    graph.remove_node(nodes[rowlen - 1]);
    graph.remove_node(
        nodes[(collen - 1) * rowlen + (rowlen - 1) * ((collen - 1) % 2)],
    );

    Ok(digraph::PyDiGraph {
        graph,
        node_removed: true,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph: multigraph,
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
    m.add_wrapped(wrap_pyfunction!(binomial_tree_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_binomial_tree_graph))?;
    m.add_wrapped(wrap_pyfunction!(hexagonal_lattice_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_hexagonal_lattice_graph))?;
    Ok(())
}
