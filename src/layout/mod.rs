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

mod bipartite;
mod circular;
mod random;
mod shell;
mod spiral;
mod spring;

use crate::{digraph, graph};
use spring::Point;

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::Python;

use crate::iterators::Pos2DMapping;

/// Position nodes using Fruchterman-Reingold force-directed algorithm.
///
/// The algorithm simulates a force-directed representation of the network
/// treating edges as springs holding nodes close, while treating nodes
/// as repelling objects, sometimes called an anti-gravity force.
/// Simulation continues until the positions are close to an equilibrium.
///
/// :param PyGraph graph: Graph to be used.
/// :param dict pos:
///     Initial node positions as a dictionary with node ids as keys and values
///     as a coordinate list. If ``None``, then use random initial positions. (``default=None``)
/// :param set fixed: Nodes to keep fixed at initial position.
///     Error raised if fixed specified and ``pos`` is not. (``default=None``)
/// :param float  k:
///     Optimal distance between nodes. If ``None`` the distance is set to
///     :math:`\frac{1}{\sqrt{n}}` where :math:`n` is the number of nodes.  Increase this value
///     to move nodes farther apart. (``default=None``)
/// :param int repulsive_exponent:
///     Repulsive force exponent. (``default=2``)
/// :param bool adaptive_cooling:
///     Use an adaptive cooling scheme. If set to ``False``,
///     a linear cooling scheme is used. (``default=True``)
/// :param int num_iter:
///     Maximum number of iterations. (``default=50``)
/// :param float tol:
///     Threshold for relative error in node position changes.
///     The iteration stops if the error is below this threshold.
///     (``default = 1e-6``)
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float
///     which will be used to represent the weight of the edge.
/// :param float (default=1) default_weight: If ``weight_fn`` isn't specified
///     this optional float value will be used for the weight/cost of each edge
/// :param float|None scale: Scale factor for positions.
///     Not used unless fixed is None. If scale is ``None``, no re-scaling is
///     performed. (``default=1.0``)
/// :param list center: Coordinate pair around which to center
///     the layout. Not used unless fixed is ``None``. (``default=None``)
/// :param int seed: An optional seed to use for the random number generator
///
/// :returns: A dictionary of positions keyed by node id.
/// :rtype: dict
#[pyfunction]
#[pyo3(
    text_signature = "(graph, pos=None, fixed=None, k=None, repulsive_exponent=2, adaptive_cooling=True,
                     num_iter=50, tol=1e-6, weight_fn=None, default_weight=1, scale=1,
                     center=None, seed=None, /)"
)]
#[allow(clippy::too_many_arguments)]
pub fn graph_spring_layout(
    py: Python,
    graph: &graph::PyGraph,
    pos: Option<HashMap<usize, Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping> {
    spring::spring_layout(
        py,
        &graph.graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )
}

/// Position nodes using Fruchterman-Reingold force-directed algorithm.
///
/// The algorithm simulates a force-directed representation of the network
/// treating edges as springs holding nodes close, while treating nodes
/// as repelling objects, sometimes called an anti-gravity force.
/// Simulation continues until the positions are close to an equilibrium.
///
/// :param PyGraph graph: Graph to be used.
/// :param dict pos:
///     Initial node positions as a dictionary with node ids as keys and values
///     as a coordinate list. If ``None``, then use random initial positions. (``default=None``)
/// :param set fixed: Nodes to keep fixed at initial position.
///     Error raised if fixed specified and ``pos`` is not. (``default=None``)
/// :param float  k:
///     Optimal distance between nodes. If ``None`` the distance is set to
///     :math:`\frac{1}{\sqrt{n}}` where :math:`n` is the number of nodes.  Increase this value
///     to move nodes farther apart. (``default=None``)
/// :param int repulsive_exponent:
///     Repulsive force exponent. (``default=2``)
/// :param bool adaptive_cooling:
///     Use an adaptive cooling scheme. If set to ``False``,
///     a linear cooling scheme is used. (``default=True``)
/// :param int num_iter:
///     Maximum number of iterations. (``default=50``)
/// :param float tol:
///     Threshold for relative error in node position changes.
///     The iteration stops if the error is below this threshold.
///     (``default = 1e-6``)
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float
///     which will be used to represent the weight of the edge.
/// :param float (default=1) default_weight: If ``weight_fn`` isn't specified
///     this optional float value will be used for the weight/cost of each edge
/// :param float|None scale: Scale factor for positions.
///     Not used unless fixed is None. If scale is ``None``, no re-scaling is
///     performed. (``default=1.0``)
/// :param list center: Coordinate pair around which to center
///     the layout. Not used unless fixed is ``None``. (``default=None``)
/// :param int seed: An optional seed to use for the random number generator
///
/// :returns: A dictionary of positions keyed by node id.
/// :rtype: dict
#[pyfunction]
#[pyo3(
    text_signature = "(graph, pos=None, fixed=None, k=None, repulsive_exponent=2, adaptive_cooling=True,
                     num_iter=50, tol=1e-6, weight_fn=None, default_weight=1, scale=1,
                     center=None, seed=None, /)"
)]
#[allow(clippy::too_many_arguments)]
pub fn digraph_spring_layout(
    py: Python,
    graph: &digraph::PyDiGraph,
    pos: Option<HashMap<usize, Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping> {
    spring::spring_layout(
        py,
        &graph.graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )
}

/// Generate a random layout
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param int seed: An optional seed to set for the random number generator.
///
/// :returns: The random layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, / center=None, seed=None)")]
pub fn graph_random_layout(
    graph: &graph::PyGraph,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    random::random_layout(&graph.graph, center, seed)
}

/// Generate a random layout
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param int seed: An optional seed to set for the random number generator.
///
/// :returns: The random layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, / center=None, seed=None)")]
pub fn digraph_random_layout(
    graph: &digraph::PyDiGraph,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    random::random_layout(&graph.graph, center, seed)
}

/// Generate a bipartite layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param set first_nodes: The set of node indices on the left (or top if
///     horitontal is true)
/// :param bool horizontal: An optional bool specifying the orientation of the
///     layout
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float aspect_ratio: An optional number for the ratio of the width to
///     the height of the layout.
///
/// :returns: The bipartite layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, first_nodes, /, horitontal=False, scale=1,
                     center=None, aspect_ratio=1.33333333333333)")]
pub fn graph_bipartite_layout(
    graph: &graph::PyGraph,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    bipartite::bipartite_layout(
        &graph.graph,
        first_nodes,
        horizontal,
        scale,
        center,
        aspect_ratio,
    )
}

/// Generate a bipartite layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param set first_nodes: The set of node indices on the left (or top if
///     horizontal is true)
/// :param bool horizontal: An optional bool specifying the orientation of the
///     layout
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float aspect_ratio: An optional number for the ratio of the width to
///     the height of the layout.
///
/// :returns: The bipartite layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, first_nodes, /, horitontal=False, scale=1,
                     center=None, aspect_ratio=1.33333333333333)")]
pub fn digraph_bipartite_layout(
    graph: &digraph::PyDiGraph,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    bipartite::bipartite_layout(
        &graph.graph,
        first_nodes,
        horizontal,
        scale,
        center,
        aspect_ratio,
    )
}

/// Generate a circular layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The circular layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None)")]
pub fn graph_circular_layout(
    graph: &graph::PyGraph,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    circular::circular_layout(&graph.graph, scale, center)
}

/// Generate a circular layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The circular layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None)")]
pub fn digraph_circular_layout(
    graph: &digraph::PyDiGraph,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    circular::circular_layout(&graph.graph, scale, center)
}

/// Generate a shell layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param list nlist: The list of lists of indices which represents each shell
/// :param float rotate: Angle (in radians) by which to rotate the starting
///     position of each shell relative to the starting position of the
///     previous shell
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The shell layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, nlist=None, rotate=None, scale=1, center=None)")]
pub fn graph_shell_layout(
    graph: &graph::PyGraph,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    shell::shell_layout(&graph.graph, nlist, rotate, scale, center)
}

/// Generate a shell layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param list nlist: The list of lists of indices which represents each shell
/// :param float rotate: Angle by which to rotate the starting position of each shell
///     relative to the starting position of the previous shell (in radians)
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The shell layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, nlist=None, rotate=None, scale=1, center=None)")]
pub fn digraph_shell_layout(
    graph: &digraph::PyDiGraph,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    shell::shell_layout(&graph.graph, nlist, rotate, scale, center)
}

/// Generate a spiral layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float resolution: The compactness of the spiral layout returned.
///     Lower values result in more compressed spiral layouts.
/// :param bool equidistant: If true, nodes will be plotted equidistant from
///     each other.
///
/// :returns: The spiral layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None, resolution=0.35,
                     equidistant=False)")]
pub fn graph_spiral_layout(
    graph: &graph::PyGraph,
    scale: Option<f64>,
    center: Option<Point>,
    resolution: Option<f64>,
    equidistant: Option<bool>,
) -> Pos2DMapping {
    spiral::spiral_layout(&graph.graph, scale, center, resolution, equidistant)
}

/// Generate a spiral layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float resolution: The compactness of the spiral layout returned.
///     Lower values result in more compressed spiral layouts.
/// :param bool equidistant: If true, nodes will be plotted equidistant from
///     each other.
///
/// :returns: The spiral layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None, resolution=0.35,
                     equidistant=False)")]
pub fn digraph_spiral_layout(
    graph: &digraph::PyDiGraph,
    scale: Option<f64>,
    center: Option<Point>,
    resolution: Option<f64>,
    equidistant: Option<bool>,
) -> Pos2DMapping {
    spiral::spiral_layout(&graph.graph, scale, center, resolution, equidistant)
}
