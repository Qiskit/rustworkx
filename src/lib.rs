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

extern crate fixedbitset;
extern crate hashbrown;
extern crate ndarray;
extern crate numpy;
extern crate petgraph;
extern crate pyo3;
extern crate rand;
extern crate rand_pcg;
extern crate rayon;

mod astar;
mod dag_isomorphism;
mod digraph;
mod dijkstra;
mod dot_utils;
mod generators;
mod graph;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use hashbrown::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::{Exception, ValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{Bfs, IntoEdgeReferences, NodeIndexable, Reversed};

use ndarray::prelude::*;
use numpy::IntoPyArray;
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use generators::PyInit_generators;

fn longest_path(graph: &digraph::PyDiGraph) -> PyResult<Vec<usize>> {
    let dag = &graph.graph;
    let mut path: Vec<usize> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    if nodes.is_empty() {
        return Ok(path);
    }
    let mut dist: HashMap<NodeIndex, (usize, NodeIndex)> = HashMap::new();
    for node in nodes {
        let parents =
            dag.neighbors_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(usize, NodeIndex)> = Vec::new();
        for p_node in parents {
            let length = dist[&p_node].0 + 1;
            us.push((length, p_node));
        }
        let maxu: (usize, NodeIndex);
        if !us.is_empty() {
            maxu = *us.iter().max_by_key(|x| x.0).unwrap();
        } else {
            maxu = (0, node);
        };
        dist.insert(node, maxu);
    }
    let first = match dist.keys().max_by_key(|index| dist[index]) {
        Some(first) => first,
        None => {
            return Err(Exception::py_err("Encountered something unexpected"))
        }
    };
    let mut v = *first;
    let mut u: Option<NodeIndex> = None;
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v.index());
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    Ok(path)
}

/// Find the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
///
/// :returns: The node indices of the longest path on the DAG
/// :rtype: list
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[text_signature = "(graph, /)"]
fn dag_longest_path(graph: &digraph::PyDiGraph) -> PyResult<Vec<usize>> {
    longest_path(graph)
}

/// Find the length of the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
///
/// :returns: The longest path length on the DAG
/// :rtype: int
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[text_signature = "(graph, /)"]
fn dag_longest_path_length(graph: &digraph::PyDiGraph) -> PyResult<usize> {
    let path = longest_path(graph)?;
    if path.is_empty() {
        return Ok(0);
    }
    let path_length: usize = path.len() - 1;
    Ok(path_length)
}

/// Find the number of weakly connected components in a DAG.
///
/// :param PyDiGraph graph: The graph to find the number of weakly connected
///     components on
///
/// :returns: The number of weakly connected components in the DAG
/// :rtype: int
#[pyfunction]
#[text_signature = "(graph, /)"]
fn number_weakly_connected_components(graph: &digraph::PyDiGraph) -> usize {
    algo::connected_components(graph)
}

/// Check that the PyDiGraph or PyDAG doesn't have a cycle
///
/// :param PyDiGraph graph: The graph to check for cycles
///
/// :returns: ``True`` if there are no cycles in the input graph, ``False``
///     if there are cycles
/// :rtype: bool
#[pyfunction]
#[text_signature = "(graph, /)"]
fn is_directed_acyclic_graph(graph: &digraph::PyDiGraph) -> bool {
    let cycle_detected = algo::is_cyclic_directed(graph);
    !cycle_detected
}

/// Determine if 2 graphs are structurally isomorphic
///
/// This checks if 2 graphs are structurally isomorphic (it doesn't match
/// the contents of the nodes or edges on the graphs).
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
///
/// :returns: ``True`` if the 2 graphs are structurally isomorphic, ``False``
///     if they are not
/// :rtype: bool
#[pyfunction]
#[text_signature = "(first, second, /)"]
fn is_isomorphic(
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> PyResult<bool> {
    let res = dag_isomorphism::is_isomorphic(first, second)?;
    Ok(res)
}

/// Determine if 2 DAGs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data using the provided matcher function. The matcher
/// function takes in 2 node data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyDAG()
///     graph_b = retworkx.PyDAG()
///     retworkx.is_isomorphic_node_match(graph_a, graph_b,
///                                       lambda x, y: x == y)
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
/// :param callable matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded as
///     matching.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction]
#[text_signature = "(first, second, matcher, /)"]
fn is_isomorphic_node_match(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    matcher: PyObject,
) -> PyResult<bool> {
    let compare_nodes = |a: &PyObject, b: &PyObject| -> PyResult<bool> {
        let res = matcher.call1(py, (a, b))?;
        Ok(res.is_true(py).unwrap())
    };

    fn compare_edges(_a: &PyObject, _b: &PyObject) -> PyResult<bool> {
        Ok(true)
    }
    let res = dag_isomorphism::is_isomorphic_matching(
        py,
        first,
        second,
        compare_nodes,
        compare_edges,
    )?;
    Ok(res)
}

/// Return the topological sort of node indexes from the provided graph
///
/// :param PyDiGraph graph: The DAG to get the topological sort on
///
/// :returns: A list of node indices topologically sorted.
/// :rtype: list
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[text_signature = "(graph, /)"]
fn topological_sort(graph: &digraph::PyDiGraph) -> PyResult<Vec<usize>> {
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    Ok(nodes.iter().map(|node| node.index()).collect())
}

/// Return successors in a breadth-first-search from a source node.
///
/// The return format is ``[(Parent Node, [Children Nodes])]`` in a bfs order
/// from the source node provided.
///
/// :param PyDiGraph graph: The DAG to get the bfs_successors from
/// :param int node: The index of the dag node to get the bfs successors for
///
/// :returns: A list of nodes's data and their children in bfs order
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, node, /)"]
fn bfs_successors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut bfs = Bfs::new(graph, index);
    let mut out_list: Vec<(&PyObject, Vec<&PyObject>)> = Vec::new();
    while let Some(nx) = bfs.next(graph) {
        let children = graph
            .graph
            .neighbors_directed(nx, petgraph::Direction::Outgoing);
        let mut succesors: Vec<&PyObject> = Vec::new();
        for succ in children {
            succesors.push(graph.graph.node_weight(succ).unwrap());
        }
        if !succesors.is_empty() {
            out_list.push((graph.graph.node_weight(nx).unwrap(), succesors));
        }
    }
    Ok(PyList::new(py, out_list).into())
}

/// Return the ancestors of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.predecessors` method  in that
/// ``predecessors`` returns only nodes with a direct edge into the provided
/// node. While this function returns all nodes that have a path into the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the descendants from
/// :param int node: The index of the graph node to get the ancestors for
///
/// :returns: A list of node indexes of ancestors of provided node.
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, node, /)"]
fn ancestors(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let reverse_graph = Reversed(graph);
    let res = algo::dijkstra(reverse_graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    out_set
}

/// Return the descendants of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.successors` method in that
/// ``successors``` returns only nodes with a direct edge out of the provided
/// node. While this function returns all nodes that have a path from the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the descendants from
/// :param int node: The index of the graph node to get the descendants for
///
/// :returns: A list of node indexes of descendants of provided node.
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, node, /)"]
fn descendants(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let res = algo::dijkstra(graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    out_set
}

/// Get the lexicographical topological sorted nodes from the provided DAG
///  
/// This function returns a list of nodes data in a graph lexicographically
/// topologically sorted using the provided key function.
///
/// :param PyDiGraph dag: The DAG to get the topological sorted nodes from
/// :param callable key: key is a python function or other callable that
///     gets passed a single argument the node data from the graph and is
///     expected to return a string which will be used for sorting.
///
/// :returns: A list of node's data lexicographically topologically sorted.
/// :rtype: list
#[pyfunction]
#[text_signature = "(dag, key, /)"]
fn lexicographical_topological_sort(
    py: Python,
    dag: &digraph::PyDiGraph,
    key: PyObject,
) -> PyResult<PyObject> {
    let key_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = key.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    // HashMap of node_index indegree
    let mut in_degree_map: HashMap<NodeIndex, usize> = HashMap::new();
    for node in dag.graph.node_indices() {
        in_degree_map.insert(node, dag.in_degree(node.index()));
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        key: String,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &State) -> Ordering {
            // Notice that the we flip the ordering on costs.
            // In case of a tie we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.index().cmp(&self.node.index()))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &State) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut zero_indegree = BinaryHeap::new();
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key_raw = key_callable(&dag.graph[*node])?;
            let map_key: String = map_key_raw.extract(py)?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<&PyObject> = Vec::new();
    let dir = petgraph::Direction::Outgoing;
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.graph.neighbors_directed(node, dir);
        for child in neighbors {
            let child_degree = in_degree_map.get_mut(&child).unwrap();
            *child_degree -= 1;
            if *child_degree == 0 {
                let map_key_raw = key_callable(&dag.graph[child])?;
                let map_key: String = map_key_raw.extract(py)?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            }
        }
        out_list.push(&dag.graph[node])
    }
    Ok(PyList::new(py, out_list).into())
}

/// Color a PyGraph using a largest_first strategy greedy graph coloring.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
#[pyfunction]
#[text_signature = "(graph, /)"]
fn graph_greedy_color(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<PyObject> {
    let mut colors: HashMap<usize, usize> = HashMap::new();
    let mut node_vec: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let mut sort_map: HashMap<NodeIndex, usize> = HashMap::new();
    for k in node_vec.iter() {
        sort_map.insert(*k, graph.graph.edges(*k).count());
    }
    node_vec.par_sort_by_key(|k| Reverse(sort_map.get(k)));
    for u_index in node_vec {
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for edge in graph.graph.edges(u_index) {
            let target = edge.target().index();
            let existing_color = match colors.get(&target) {
                Some(node) => node,
                None => continue,
            };
            neighbor_colors.insert(*existing_color);
        }
        let mut count: usize = 0;
        loop {
            if !neighbor_colors.contains(&count) {
                break;
            }
            count += 1;
        }
        colors.insert(u_index.index(), count);
    }
    let out_dict = PyDict::new(py);
    for (index, color) in colors {
        out_dict.set_item(index, color)?;
    }
    Ok(out_dict.into())
}

/// Return the shortest path lengths between ever pair of nodes that has a
/// path connecting them
///
/// The runtime is :math:`O(|N|^3 + |E|)` where :math:`|N|` is the number
/// of nodes and :math:`|E|` is the number of edges.
///
/// This is done with the Floyd Warshall algorithm:
///      
/// 1. Process all edges by setting the distance from the parent to
///    the child equal to the edge weight.
/// 2. Iterate through every pair of nodes (source, target) and an additional
///    itermediary node (w). If the distance from source :math:`\rightarrow` w
///    :math:`\rightarrow` target is less than the distance from source
///    :math:`\rightarrow` target, update the source :math:`\rightarrow` target
///    distance (to pass through w).
///
/// The return format is ``{Source Node: {Target Node: Distance}}``.
///
/// .. note::
///
///     Paths that do not exist are simply not found in the return dictionary,
///     rather than setting the distance to infinity, or -1.
///
/// .. note::
///
///     Edge weights are restricted to 1 in the current implementation.
///
/// :param PyDigraph graph: The DiGraph to get all shortest paths from
///
/// :returns: A dictionary of shortest paths
/// :rtype: dict
#[pyfunction]
#[text_signature = "(dag, /)"]
fn floyd_warshall(py: Python, dag: &digraph::PyDiGraph) -> PyResult<PyObject> {
    let mut dist: HashMap<(usize, usize), usize> = HashMap::new();
    for node in dag.graph.node_indices() {
        // Distance from a node to itself is zero
        dist.insert((node.index(), node.index()), 0);
    }
    for edge in dag.graph.edge_indices() {
        // Distance between nodes that share an edge is 1
        let source_target = dag.graph.edge_endpoints(edge).unwrap();
        let u = source_target.0.index();
        let v = source_target.1.index();
        // Update dist only if the key hasn't been set to 0 already
        // (i.e. in case edge is a self edge). Assumes edge weight = 1.
        dist.entry((u, v)).or_insert(1);
    }
    // The shortest distance between any pair of nodes u, v is the min of the
    // distance tracked so far from u->v and the distance from u to v thorough
    // another node w, for any w.
    for w in dag.graph.node_indices() {
        for u in dag.graph.node_indices() {
            for v in dag.graph.node_indices() {
                let u_v_dist = match dist.get(&(u.index(), v.index())) {
                    Some(u_v_dist) => *u_v_dist,
                    None => std::usize::MAX,
                };
                let u_w_dist = match dist.get(&(u.index(), w.index())) {
                    Some(u_w_dist) => *u_w_dist,
                    None => std::usize::MAX,
                };
                let w_v_dist = match dist.get(&(w.index(), v.index())) {
                    Some(w_v_dist) => *w_v_dist,
                    None => std::usize::MAX,
                };
                if u_w_dist == std::usize::MAX || w_v_dist == std::usize::MAX {
                    // Avoid overflow!
                    continue;
                }
                if u_v_dist > u_w_dist + w_v_dist {
                    dist.insert((u.index(), v.index()), u_w_dist + w_v_dist);
                }
            }
        }
    }

    // Some re-formatting for Python: Dict[int, Dict[int, int]]
    let out_dict = PyDict::new(py);
    for (nodes, distance) in dist {
        let u_index = nodes.0;
        let v_index = nodes.1;
        if out_dict.contains(u_index)? {
            let u_dict =
                out_dict.get_item(u_index).unwrap().downcast::<PyDict>()?;
            u_dict.set_item(v_index, distance)?;
            out_dict.set_item(u_index, u_dict)?;
        } else {
            let u_dict = PyDict::new(py);
            u_dict.set_item(v_index, distance)?;
            out_dict.set_item(u_index, u_dict)?;
        }
    }
    Ok(out_dict.into())
}

/// Return a list of layers
///  
/// A layer is a subgraph whose nodes are disjoint, i.e.,
/// a layer has depth 1. The layers are constructed using a greedy algorithm.
///
/// :param PyDiGraph graph: The DAG to get the layers from
/// :param list first_layer: A list of node ids for the first layer. This
///     will be the first layer in the output
///
/// :returns: A list of layers, each layer is a list of node data
/// :rtype: list
#[pyfunction]
#[text_signature = "(dag, first_layer, /)"]
fn layers(
    py: Python,
    dag: &digraph::PyDiGraph,
    first_layer: Vec<usize>,
) -> PyResult<PyObject> {
    let mut output: Vec<Vec<&PyObject>> = Vec::new();
    // Convert usize to NodeIndex
    let mut first_layer_index: Vec<NodeIndex> = Vec::new();
    for index in first_layer {
        first_layer_index.push(NodeIndex::new(index));
    }

    let mut cur_layer = first_layer_index;
    let mut next_layer: Vec<NodeIndex> = Vec::new();
    let mut predecessor_count: HashMap<NodeIndex, usize> = HashMap::new();

    let mut layer_node_data: Vec<&PyObject> = Vec::new();
    for layer_node in &cur_layer {
        layer_node_data.push(&dag[*layer_node]);
    }
    output.push(layer_node_data);

    // Iterate until there are no more
    while !cur_layer.is_empty() {
        for node in &cur_layer {
            let children = dag
                .graph
                .neighbors_directed(*node, petgraph::Direction::Outgoing);
            let mut used_indexes: HashSet<NodeIndex> = HashSet::new();
            for succ in children {
                // Skip duplicate successors
                if used_indexes.contains(&succ) {
                    continue;
                }
                used_indexes.insert(succ);
                let mut multiplicity: usize = 0;
                let raw_edges = dag
                    .graph
                    .edges_directed(*node, petgraph::Direction::Outgoing);
                for edge in raw_edges {
                    if edge.target() == succ {
                        multiplicity += 1;
                    }
                }
                predecessor_count
                    .entry(succ)
                    .and_modify(|e| *e -= multiplicity)
                    .or_insert(dag.in_degree(succ.index()) - multiplicity);
                if *predecessor_count.get(&succ).unwrap() == 0 {
                    next_layer.push(succ);
                    predecessor_count.remove(&succ);
                }
            }
        }
        let mut layer_node_data: Vec<&PyObject> = Vec::new();
        for layer_node in &next_layer {
            layer_node_data.push(&dag[*layer_node]);
        }
        if !layer_node_data.is_empty() {
            output.push(layer_node_data);
        }
        cur_layer = next_layer;
        next_layer = Vec::new();
    }
    Ok(PyList::new(py, output).into())
}

/// Return the adjacency matrix for a PyDiGraph object
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be the sum of the edges' weights.
///
/// :param PyDiGraph graph: The DiGraph used to generate the adjacency matrix
///     from
/// :param weight_fn callable: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         dag_adjacency_matrix(dag, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         dag_adjacency_matrix(dag, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
///
///  :return: The adjacency matrix for the input dag as a numpy array
///  :rtype: numpy.ndarray
#[pyfunction]
#[text_signature = "(graph, weight_fn, /)"]
fn digraph_adjacency_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: PyObject,
) -> PyResult<PyObject> {
    let node_map: Option<HashMap<NodeIndex, usize>>;
    let n: usize;
    if graph.node_removed {
        let mut node_hash_map: HashMap<NodeIndex, usize> = HashMap::new();
        let mut count = 0;
        for node in graph.graph.node_indices() {
            node_hash_map.insert(node, count);
            count += 1;
        }
        n = count;
        node_map = Some(node_hash_map);
    } else {
        n = graph.graph.node_bound();
        node_map = None;
    }
    let mut matrix = Array::<f64, _>::zeros((n, n).f());

    let weight_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = weight_fn.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    for edge in graph.graph.edge_references() {
        let edge_weight_raw = weight_callable(&edge.weight())?;
        let edge_weight: f64 = edge_weight_raw.extract(py)?;
        let source = edge.source();
        let target = edge.target();
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                i = *map.get(&source).unwrap();
                j = *map.get(&target).unwrap();
            }
            None => {
                i = source.index();
                j = target.index();
            }
        }
        matrix[[i, j]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return the adjacency matrix for a PyGraph class
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be the sum of the edges' weights.
///
/// :param PyGraph graph: The graph used to generate the adjacency matrix from
/// :param weight_fn callable: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_adjacency_matrix(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_adjacency_matrix(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
///
/// :return: The adjacency matrix for the input dag as a numpy array
/// :rtype: numpy.ndarray
#[pyfunction]
#[text_signature = "(graph, weight_fn, /)"]
fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
) -> PyResult<PyObject> {
    let node_map: Option<HashMap<NodeIndex, usize>>;
    let n: usize;
    if graph.node_removed {
        let mut node_hash_map: HashMap<NodeIndex, usize> = HashMap::new();
        let mut count = 0;
        for node in graph.graph.node_indices() {
            node_hash_map.insert(node, count);
            count += 1;
        }
        n = count;
        node_map = Some(node_hash_map);
    } else {
        n = graph.graph.node_bound();
        node_map = None;
    }
    let mut matrix = Array::<f64, _>::zeros((n, n).f());

    let weight_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = weight_fn.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    for edge in graph.graph.edge_references() {
        let edge_weight_raw = weight_callable(&edge.weight())?;
        let edge_weight: f64 = edge_weight_raw.extract(py)?;
        let source = edge.source();
        let target = edge.target();
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                i = *map.get(&source).unwrap();
                j = *map.get(&target).unwrap();
            }
            None => {
                i = source.index();
                j = target.index();
            }
        }
        matrix[[i, j]] += edge_weight;
        matrix[[j, i]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return all simple paths between 2 nodes in a PyGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path of node indices
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, from, to, /, min=None, cutoff=None)"]
fn graph_all_simple_paths(
    graph: &graph::PyGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::py_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::py_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = match cutoff {
        Some(depth) => Some(depth - 2),
        None => None,
    };
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

/// Return all simple paths between 2 nodes in a PyDiGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyDiGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     sett to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, from, to, /, min_depth=None, cutoff=None)"]
fn digraph_all_simple_paths(
    graph: &digraph::PyDiGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::py_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::py_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = match cutoff {
        Some(depth) => Some(depth - 2),
        None => None,
    };
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

/// Compute the lengths of the shortest paths for a PyGraph object using
/// Dijkstra's algorithm
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: dict
#[pyfunction]
#[text_signature = "(graph, node, edge_cost_fn, /, goal=None)"]
fn graph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PyObject> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        Ok(raw.extract(py)?)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = match goal {
        Some(node) => Some(NodeIndex::new(node)),
        None => None,
    };

    let res = dijkstra::dijkstra(graph, start, goal_index, |e| {
        edge_cost_callable(e.weight())
    })?;
    let out_dict = PyDict::new(py);
    for (index, value) in res {
        let int_index = index.index();
        if int_index == node {
            continue;
        }
        if (goal.is_some() && goal.unwrap() == int_index) || goal.is_none() {
            out_dict.set_item(int_index, value)?;
        }
    }
    Ok(out_dict.into())
}

/// Compute the lengths of the shortest paths for a PyDiGraph object using
/// Dijkstra's algorithm
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: dict
#[pyfunction]
#[text_signature = "(graph, node, edge_cost_fn, /, goal=None)"]
fn digraph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PyObject> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        Ok(raw.extract(py)?)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = match goal {
        Some(node) => Some(NodeIndex::new(node)),
        None => None,
    };

    let res = dijkstra::dijkstra(graph, start, goal_index, |e| {
        edge_cost_callable(e.weight())
    })?;
    let out_dict = PyDict::new(py);
    for (index, value) in res {
        let int_index = index.index();
        if int_index == node {
            continue;
        }
        if (goal.is_some() && goal.unwrap() == int_index) || goal.is_none() {
            out_dict.set_item(int_index, value)?;
        }
    }
    Ok(out_dict.into())
}

/// Compute the A* shortest path for a PyGraph
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's data
///     object and will return a boolean which will be True if it is the finish
///     node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an edge's
///     data object and will return a float that represents the cost of that
///     edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the estimated
///     cost for the next node. The return must be non-negative. For the
///     algorithm to find the actual shortest path, it should be admissible,
///     meaning that it should never overestimate the actual cost to get to the
///     nearest goal node.
///
/// :returns: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)"]
fn graph_astar_shortest_path(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<Vec<usize>> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::py_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(path.1.into_iter().map(|x| x.index()).collect())
}

/// Compute the A* shortest path for a PyDiGraph
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's
///     data object and will return a boolean which will be True if it is the
///     finish node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the cost of
///     that edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the
///     estimated cost for the next node. The return must be non-negative. For
///     the algorithm to find the actual shortest path, it should be
///     admissible, meaning that it should never overestimate the actual cost
///     to get to the nearest goal node.
///
/// :return: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)"]
fn digraph_astar_shortest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<Vec<usize>> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::py_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(path.1.into_iter().map(|x| x.index()).collect())
}

/// Return a :math:`G_{np}` directed random graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// The :math:`G_{n,p}` graph algorithm chooses each of the
/// :math:`n (n - 1)` possible edges with probability :math:`p`.
/// This algorithm [1]_ runs in :math:`O(n + m)` time, where :math:`m` is the
/// expected number of edges, which equals :math:`p n (n - 1)/2`.
///
/// Based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[text_signature = "(num_nodes, probability, seed=None, /)"]
pub fn directed_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(ValueError::py_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableDiGraph::<PyObject, PyObject>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if probability <= 0.0 || probability >= 1.0 {
        return Err(ValueError::py_err(
            "Probability out of range, must be 0 < p < 1",
        ));
    }
    let mut v: isize = 0;
    let mut w: isize = -1;
    let lp: f64 = (1.0 - probability).ln();

    while v < num_nodes {
        let random: f64 = rng.gen_range(0.0, 1.0);
        let lr: f64 = (1.0 - random).ln();
        let ratio: isize = (lr / lp) as isize;
        w = w + 1 + ratio;
        // avoid self loops
        if v == w {
            w += 1;
        }
        while v < num_nodes && num_nodes <= w {
            w -= v;
            v += 1;
            // avoid self loops
            if v == w {
                w -= v;
                v += 1;
            }
        }
        if v < num_nodes {
            let v_index = NodeIndex::new(v as usize);
            let w_index = NodeIndex::new(w as usize);
            inner_graph.add_edge(v_index, w_index, py.None());
        }
    }

    let graph = digraph::PyDiGraph {
        graph: inner_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
    };
    Ok(graph)
}

/// Return a :math:`G_{np}` random undirected graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// The :math:`G_{n,p}` graph algorithm chooses each of the
/// :math:`n (n - 1)/2` possible edges with probability :math:`p`.
/// This algorithm [1]_ runs in :math:`O(n + m)` time, where :math:`m` is the
/// expected number of edges, which equals :math:`p n (n - 1)/2`.
///
/// Based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[text_signature = "(num_nodes, probability, seed=None, /)"]
pub fn undirected_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(ValueError::py_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if probability <= 0.0 || probability >= 1.0 {
        return Err(ValueError::py_err(
            "Probability out of range, must be 0 < p < 1",
        ));
    }
    let mut v: isize = 1;
    let mut w: isize = -1;
    let lp: f64 = (1.0 - probability).ln();

    while v < num_nodes {
        let random: f64 = rng.gen_range(0.0, 1.0);
        let lr = (1.0 - random).ln();
        let ratio: isize = (lr / lp) as isize;
        w = w + 1 + ratio;
        while w >= v && v < num_nodes {
            w -= v;
            v += 1;
        }
        if v < num_nodes {
            let v_index = NodeIndex::new(v as usize);
            let w_index = NodeIndex::new(w as usize);
            inner_graph.add_edge(v_index, w_index, py.None());
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
    };
    Ok(graph)
}

/// Compute the strongly connected components for a directed graph
///
/// This function is implemented using Kosaraju's algorithm
///
/// :param PyDiGraph graph: The input graph to find the strongly connected
///     components for.
///
/// :return: A list of list of node ids for strongly connected components
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, /)"]
pub fn strongly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<Vec<usize>> {
    algo::kosaraju_scc(graph)
        .iter()
        .map(|x| x.iter().map(|id| id.index()).collect())
        .collect()
}

// The provided node is invalid.
create_exception!(retworkx, InvalidNode, Exception);
// Performing this operation would result in trying to add a cycle to a DAG.
create_exception!(retworkx, DAGWouldCycle, Exception);
// There is no edge present between the provided nodes.
create_exception!(retworkx, NoEdgeBetweenNodes, Exception);
// The specified Directed Graph has a cycle and can't be treated as a DAG.
create_exception!(retworkx, DAGHasCycle, Exception);
// No neighbors found matching the provided predicate.
create_exception!(retworkx, NoSuitableNeighbors, Exception);
// No path was found between the specified nodes.
create_exception!(retworkx, NoPathFound, Exception);

#[pymodule]
fn retworkx(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InvalidNode", py.get_type::<InvalidNode>())?;
    m.add("DAGWouldCycle", py.get_type::<DAGWouldCycle>())?;
    m.add("NoEdgeBetweenNodes", py.get_type::<NoEdgeBetweenNodes>())?;
    m.add("DAGHasCycle", py.get_type::<DAGHasCycle>())?;
    m.add("NoSuitableNeighbors", py.get_type::<NoSuitableNeighbors>())?;
    m.add("NoPathFound", py.get_type::<NoPathFound>())?;
    m.add_wrapped(wrap_pyfunction!(bfs_successors))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_directed_acyclic_graph))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic_node_match))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(layers))?;
    m.add_wrapped(wrap_pyfunction!(digraph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_greedy_color))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(strongly_connected_components))?;
    m.add_class::<digraph::PyDiGraph>()?;
    m.add_class::<graph::PyGraph>()?;
    m.add_wrapped(wrap_pymodule!(generators))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
