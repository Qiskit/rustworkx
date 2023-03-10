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

use petgraph::algo;
use petgraph::prelude::*;
use petgraph::Undirected;

use pyo3::exceptions::{PyIndexError, PyOverflowError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use super::{digraph, graph, StablePyGraph};
use rustworkx_core::generators as core_generators;

/// Generate an undirected cycle graph
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the cycle graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::cycle_graph(num_nodes, weights, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed cycle graph
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the cycle graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, bidirectional=false, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_cycle_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::cycle_graph(
        num_nodes,
        weights,
        default_fn,
        default_fn,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "num_nodes and weights list not specified",
            ))
        }
    };
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
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the path graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::path_graph(num_nodes, weights, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed path graph
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the path graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, bidirectional=false, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, bidirectional=False, multigraph=True)"
)]
pub fn directed_path_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::path_graph(
        num_nodes,
        weights,
        default_fn,
        default_fn,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "num_nodes and weights list not specified",
            ))
        }
    };
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
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the star graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn star_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::star_graph(num_nodes, weights, default_fn, default_fn, false, false)
        {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed star graph
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights, the first element in the list
///     will be the center node of the star graph. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool inward: If set ``True`` the nodes will be directed towards the
///     center node. This parameter is ignored if ``bidirectional`` is set to
///     ``True``.
/// :param bool bidirectional: Adds edges in both directions between two nodes
///     if set to ``True``. Default value is ``False``.
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, inward=false, bidirectional=false, multigraph=true),
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
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::star_graph(
        num_nodes,
        weights,
        default_fn,
        default_fn,
        inward,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "num_nodes and weights list not specified",
            ))
        }
    };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected mesh (complete) graph where every node is connected to every other
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn mesh_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    complete_graph(py, num_nodes, weights, multigraph)
}

/// Generate a directed mesh (complete) graph where every node is connected to every other
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn directed_mesh_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    directed_complete_graph(py, num_nodes, weights, multigraph)
}

/// Generate an undirected grid graph.
///
/// :param int rows: The number of rows to generate the graph with.
///     If specified, ``cols`` also need to be specified
/// :param int cols: The number of cols to generate the graph with.
///     If specified, ``rows`` also need to be specified. rows*cols
///     defines the number of nodes in the graph
/// :param list weights: A list of node weights. Nodes are filled row wise.
///     If rows and cols are not specified, then a linear graph containing
///     all the values in weights list is created.
///     If number of nodes(rows*cols) is less than length of
///     weights list, the trailing weights are ignored.
///     If number of nodes(rows*cols) is greater than length of
///     weights list, extra nodes with None weight are appended.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(rows=None, cols=None, weights=None, multigraph=true),
    text_signature = "(/, rows=None, cols=None, weights=None, multigraph=True)"
)]
pub fn grid_graph(
    py: Python,
    rows: Option<usize>,
    cols: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::grid_graph(rows, cols, weights, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("rows and cols not specified")),
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed grid graph.
///
/// The edges propagate towards right and bottom direction if ``bidirectional`` is ``False``
///
/// :param int rows: The number of rows to generate the graph with.
///     If specified, ``cols`` also need to be specified.
/// :param int cols: The number of cols to generate the graph with.
///     If specified, ``rows`` also need to be specified. rows*cols
///     defines the number of nodes in the graph.
/// :param list weights: A list of node weights. Nodes are filled row wise.
///     If rows and cols are not specified, then a linear graph containing
///     all the values in weights list is created.
///     If number of nodes(rows*cols) is less than length of
///     weights list, the trailing weights are ignored.
///     If number of nodes(rows*cols) is greater than length of
///     weights list, extra nodes with None weight are appended.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes. Defaults to ``False``.
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(rows=None, cols=None, weights=None, bidirectional=false, multigraph=true),
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
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::grid_graph(
        rows,
        cols,
        weights,
        default_fn,
        default_fn,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => return Err(PyIndexError::new_err("rows and cols not specified")),
    };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected heavy square graph.
///
/// Fig. 6 of https://arxiv.org/abs/1907.09528.
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
///
/// .. note::
///   
///   This function generates the four-frequency variant of the heavy square code.
///   This function implements Fig 10.b left of the `paper <https://arxiv.org/abs/1907.09528>`_.
///   This function doesn't support the variant Fig 10.b right.
///
/// Note that if ``d`` is set to ``1`` a :class:`~rustworkx.PyGraph` with a
/// single node will be returned.
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyGraph` with a single node will be returned. ``d`` must be
///     an odd number.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(d, multigraph=true),
    text_signature = "(d, /, multigraph=True)"
)]
pub fn heavy_square_graph(py: Python, d: usize, multigraph: bool) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::heavy_square_graph(d, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("d must be an odd number.")),
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an directed heavy square graph.
///
/// Fig. 6 of https://arxiv.org/abs/1907.09528.
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
///
/// .. note::
///   
///   This function generates the four-frequency variant of the heavy square code.
///   This function implements Fig 10.b left of the `paper <https://arxiv.org/abs/1907.09528>`_.
///   This function doesn't support the variant Fig 10.b right.
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyDiGraph` with a single node will be returned. ``d`` must be
///     an odd number.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes. Defaults to ``False``.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(d, bidirectional=false, multigraph=true),
    text_signature = "(d, /, bidirectional=False, multigraph=True)"
)]
pub fn directed_heavy_square_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> =
        match core_generators::heavy_square_graph(d, default_fn, default_fn, bidirectional) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("d must be an odd number.")),
        };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected heavy hex graph.
///
/// Fig. 2 of https://arxiv.org/abs/1907.09528
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
///     ``d`` must be an odd number.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes. Defaults to ``False``.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(d, multigraph=true),
    text_signature = "(d, /, multigraph=True)"
)]
pub fn heavy_hex_graph(py: Python, d: usize, multigraph: bool) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::heavy_hex_graph(d, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("d must be an odd number.")),
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed heavy hex graph.
///
/// Fig. 2 of https://arxiv.org/abs/1907.09528
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
///     ``d`` must be an odd number.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(d, bidirectional=false, multigraph=true),
    text_signature = "(d, /, bidirectional=False, multigraph=True)"
)]
pub fn directed_heavy_hex_graph(
    py: Python,
    d: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> =
        match core_generators::heavy_hex_graph(d, default_fn, default_fn, bidirectional) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("d must be an odd number.")),
        };
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
///     the max value is 60 and on 32bit systems the max value is 29. Any order
///     value above these will raise an ``OverflowError``.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order, extra nodes with None will be appended.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead calls which would
///     create a parallel edge will update the existing edge.
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
#[pyfunction]
#[pyo3(
    signature=(order, weights=None, multigraph=true),
    text_signature = "(order, /,  weights=None, multigraph=True)"
)]
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
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::binomial_tree_graph(order, weights, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed binomial tree of order n recursively.
///
/// The edges propagate towards right and bottom direction if ``bidirectional`` is ``False``
///
/// :param int order: Order of the binomial tree. The maximum allowed value
///     for order on the platform your running on. If it's a 64bit platform
///     the max value is 60 and on 32bit systems the max value is 29. Any order
///     value above these will raise an ``OverflowError``.
/// :param list weights: A list of node weights. If the number of weights is
///     less than 2**order, extra nodes with None will be appended.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes. Defaults to ``False``.
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(order, weights=None, bidirectional=false, multigraph=true),
    text_signature = "(order, /,  weights=None, bidirectional=False, multigraph=True)"
)]
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
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::binomial_tree_graph(
        order,
        weights,
        default_fn,
        default_fn,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "order and weights list not specified",
            ))
        }
    };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Creates a full r-ary tree of `n` nodes.
///
/// Sometimes called a k-ary, n-ary, or m-ary tree.
///
/// :param int branching factor: The number of children at each node.
/// :param int num_nodes: The number of nodes in the graph.
/// :param list weights: A list of node weights. If the number of weights is
///     less than ``num_nodes``, extra nodes with None will be appended. The
///     number of weights cannot exceed num_nodes.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(branching_factor, num_nodes, weights=None, multigraph=true),
    text_signature = "(branching_factor, num_nodes, /, weights=None, multigraph=True)"
)]
pub fn full_rary_tree(
    py: Python,
    branching_factor: usize,
    num_nodes: usize,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> = match core_generators::full_rary_tree_graph(
        branching_factor,
        num_nodes,
        weights,
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "The number of weights cannot exceed num_nodes.",
            ))
        }
    };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected hexagonal lattice graph.
///
/// :param int rows: The number of rows to generate the graph with.
/// :param int cols: The number of columns to generate the graph with.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(rows, cols, multigraph=true),
    text_signature = "(rows, cols, /, multigraph=True)"
)]
pub fn hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::hexagonal_lattice_graph(rows, cols, default_fn, default_fn, false) {
            Ok(graph) => graph,
            Err(_) => return Err(PyIndexError::new_err("rows and cols not specified")),
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed hexagonal lattice graph.
///
/// The edges propagate towards right and bottom direction if ``bidirectional`` is ``False``
///
/// :param int rows: The number of rows to generate the graph with.
/// :param int cols: The number of rows to generate the graph with.
/// :param bidirectional: A parameter to indicate if edges should exist in
///     both directions between nodes. Defaults to ``False``.
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(rows, cols, bidirectional=false, multigraph=true),
    text_signature = "(rows, cols, /, bidirectional=False, multigraph=True)"
)]
pub fn directed_hexagonal_lattice_graph(
    py: Python,
    rows: usize,
    cols: usize,
    bidirectional: bool,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> = match core_generators::hexagonal_lattice_graph(
        rows,
        cols,
        default_fn,
        default_fn,
        bidirectional,
    ) {
        Ok(graph) => graph,
        Err(_) => return Err(PyIndexError::new_err("rows and cols not specified")),
    };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected lollipop graph where a mesh (complete) graph is connected to a
/// path.
///
/// If neither ``num_path_nodes`` nor ``path_weights`` (both described
/// below) are specified then this is equivalent to
/// :func:`~rustworkx.generators.complete_graph`
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
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
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
#[pyfunction]
#[pyo3(
    signature=(num_mesh_nodes=None, num_path_nodes=None, mesh_weights=None, path_weights=None, multigraph=true),
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
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> = match core_generators::lollipop_graph(
        num_mesh_nodes,
        num_path_nodes,
        mesh_weights,
        path_weights,
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "num_nodes and weights list not specified",
            ))
        }
    };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected barbell graph where two identical complete graphs are
/// connected by a path.
///
/// If ``num_path_nodes`` (described below) is not specified then this is
/// equivalent to two complete graphs joined together.
///
/// :param int num_mesh_nodes: The number of nodes to generate the mesh graphs
///     with. Node weights will be None if this is specified. If both
///     ``num_mesh_nodes`` and ``mesh_weights`` are set this will be ignored and
///     ``mesh_weights`` will be used.
/// :param int num_path_nodes: The number of nodes to generate the path
///     with. Node weights will be None if this is specified. If both
///     ``num_path_nodes`` and ``path_weights`` are set this will be ignored and
///     ``path_weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
/// :param list mesh_weights: A list of node weights for the mesh graph. If both
///     ``num_mesh_nodes`` and ``mesh_weights`` are set ``num_mesh_nodes`` will
///     be ignored and ``mesh_weights`` will be used.
/// :param list path_weights: A list of node weights for the path. If both
///     ``num_path_nodes`` and ``path_weights`` are set ``num_path_nodes`` will
///     be ignored and ``path_weights`` will be used.
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
#[pyfunction]
#[pyo3(
    signature=(num_mesh_nodes=None, num_path_nodes=None, multigraph=true, mesh_weights=None, path_weights=None)
)]
pub fn barbell_graph(
    py: Python,
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    multigraph: bool,
    mesh_weights: Option<Vec<PyObject>>,
    path_weights: Option<Vec<PyObject>>,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> = match core_generators::barbell_graph(
        num_mesh_nodes,
        num_path_nodes,
        mesh_weights,
        path_weights,
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyIndexError::new_err(
                "num_nodes and weights list not specified",
            ))
        }
    };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
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
/// :param bool multigraph: When set to ``False`` the output
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
#[pyfunction]
#[pyo3(
    signature=(n, k, multigraph=true),
    text_signature = "(n, k, /, multigraph=True)"
)]
pub fn generalized_petersen_graph(
    py: Python,
    n: usize,
    k: usize,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::petersen_graph(n, k, default_fn, default_fn) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "n > 2, k > 0, or 2 * k > n not satisfied.",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected empty graph with ``n`` nodes and no edges.
///
/// :param int n: The number of nodes to generate the graph with.
///
/// :returns: The generated empty graph
/// :rtype: PyGraph
///
/// .. jupyter-execute::
///
///  import rustworkx.generators
///  from rustworkx.visualization import mpl_draw
///
///  graph = rustworkx.generators.empty_graph(5)
///  mpl_draw(graph)
///
#[pyfunction]
#[pyo3(
    signature=(n, multigraph=true),
    text_signature = "(/, n, multigraph=True)"
)]
pub fn empty_graph(py: Python, n: usize, multigraph: bool) -> PyResult<graph::PyGraph> {
    let mut graph = StableUnGraph::<PyObject, PyObject>::default();
    for _ in 0..n {
        graph.add_node(py.None());
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed empty graph with ``n`` nodes and no edges.
///
/// :param int n: The number of nodes to generate the graph with.
///
/// :returns: The generated empty graph
/// :rtype: PyDiGraph
///
/// .. jupyter-execute::
///
///  import rustworkx.generators
///  from rustworkx.visualization import mpl_draw
///
///  graph = rustworkx.generators.directed_empty_graph(5)
///  mpl_draw(graph)
///
#[pyfunction]
#[pyo3(
    signature=(n, multigraph=true),
    text_signature = "(/, n, multigraph=True)"
)]
pub fn directed_empty_graph(
    py: Python,
    n: usize,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let mut graph = StableDiGraph::<PyObject, PyObject>::default();
    for _ in 0..n {
        graph.add_node(py.None());
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate an undirected complete graph with ``n`` nodes.
///
/// A complete graph is a simple graph in which each pair of distinct
/// vertices is connected by a unique edge.
/// The complete graph on ``n`` nodes is the graph with the set of nodes
/// ``{0, 1, ..., n-1}`` and the set of edges ``{(i, j) : i < j, 0 <= i < n, 0 <= j < n}``.
/// The number of edges in the complete graph is ``n*(n-1)/2``.
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated complete graph
/// :rtype: PyGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///  import rustworkx.generators
///  from rustworkx.visualization import mpl_draw
///
///  graph = rustworkx.generators.complete_graph(5)
///  mpl_draw(graph)
///
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn complete_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Undirected> =
        match core_generators::complete_graph(num_nodes, weights, default_fn, default_fn) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph,
        attrs: py.None(),
    })
}

/// Generate a directed complete graph with ``n`` nodes.
///
/// A directed complete graph is a directed graph in which each pair of distinct
/// vertices is connected by a unique pair of directed edges.
/// The directed complete graph on ``n`` nodes is the graph with the set of nodes
/// ``{0, 1, ..., n-1}`` and the set of edges ``{(i, j) : 0 <= i < n, 0 <= j < n}``.
/// The number of edges in the directed complete graph is ``n*(n-1)``.
///
/// :param int num_nodes: The number of nodes to generate the graph with. Node
///     weights will be None if this is specified. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param list weights: A list of node weights. If both ``num_nodes`` and
///     ``weights`` are set this will be ignored and ``weights`` will be used.
/// :param bool multigraph: When set to ``False`` the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// :returns: The generated directed complete graph
/// :rtype: PyDiGraph
/// :raises IndexError: If neither ``num_nodes`` or ``weights`` are specified
///
/// .. jupyter-execute::
///
///  import rustworkx.generators
///  from rustworkx.visualization import mpl_draw
///
///  graph = rustworkx.generators.directed_complete_graph(5)
///  mpl_draw(graph)
///
#[pyfunction]
#[pyo3(
    signature=(num_nodes=None, weights=None, multigraph=true),
    text_signature = "(/, num_nodes=None, weights=None, multigraph=True)"
)]
pub fn directed_complete_graph(
    py: Python,
    num_nodes: Option<usize>,
    weights: Option<Vec<PyObject>>,
    multigraph: bool,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let graph: StablePyGraph<Directed> =
        match core_generators::complete_graph(num_nodes, weights, default_fn, default_fn) {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyIndexError::new_err(
                    "num_nodes and weights list not specified",
                ))
            }
        };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
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
    m.add_wrapped(wrap_pyfunction!(full_rary_tree))?;
    m.add_wrapped(wrap_pyfunction!(hexagonal_lattice_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_hexagonal_lattice_graph))?;
    m.add_wrapped(wrap_pyfunction!(lollipop_graph))?;
    m.add_wrapped(wrap_pyfunction!(barbell_graph))?;
    m.add_wrapped(wrap_pyfunction!(generalized_petersen_graph))?;
    m.add_wrapped(wrap_pyfunction!(empty_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_empty_graph))?;
    m.add_wrapped(wrap_pyfunction!(complete_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_complete_graph))?;
    Ok(())
}
