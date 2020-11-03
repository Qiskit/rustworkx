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
#[pyfunction(bidirectional = "false")]
#[text_signature = "(/, num_nodes=None, weights=None, bidirectional=False)"]
pub fn directed_cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
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
#[pyfunction]
#[text_signature = "(/, num_nodes=None, weights=None)"]
pub fn cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
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
#[pyfunction(bidirectional = "false")]
#[text_signature = "(/, num_nodes=None, weights=None, bidirectional=False)"]
pub fn directed_path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
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
                if bidirectional {
                    graph.add_edge(node_a, node_b, py.None());
                    graph.add_edge(node_b, node_a, py.None());
                } else {
                    graph.add_edge(node_a, node_b, py.None());
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
#[pyfunction]
#[text_signature = "(/, num_nodes=None, weights=None)"]
pub fn path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
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
#[pyfunction(inward = "false", bidirectional = "false")]
#[text_signature = "(/, num_nodes=None, weights=None, inward=False, bidirectional=False)"]
pub fn directed_star_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    inward: bool,
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
        } else {
            if inward {
                graph.add_edge(*node, nodes[0], py.None());
            } else {
                graph.add_edge(nodes[0], *node, py.None());
            }
        }
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
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
#[pyfunction]
#[text_signature = "(/, num_nodes=None, weights=None)"]
pub fn star_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
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
    Ok(())
}
