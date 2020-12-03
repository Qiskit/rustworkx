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

mod astar;
mod dag_isomorphism;
mod digraph;
mod dijkstra;
mod dot_utils;
mod generators;
mod graph;
mod iterators;
mod k_shortest_path;
mod union;

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

use hashbrown::{HashMap, HashSet};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{
    Bfs, Data, GraphBase, GraphProp, IntoEdgeReferences, IntoNeighbors,
    IntoNodeIdentifiers, NodeCount, NodeIndexable, Reversed, VisitMap,
    Visitable,
};

use ndarray::prelude::*;
use numpy::IntoPyArray;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::generators::PyInit_generators;
use crate::iterators::{EdgeList, NodeIndices};

trait NodesRemoved {
    fn nodes_removed(&self) -> bool;
}

fn longest_path(graph: &digraph::PyDiGraph) -> PyResult<Vec<usize>> {
    let dag = &graph.graph;
    let mut path: Vec<usize> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
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
            return Err(PyException::new_err(
                "Encountered something unexpected",
            ))
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
/// :rtype: NodeIndices
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[text_signature = "(graph, /)"]
fn dag_longest_path(graph: &digraph::PyDiGraph) -> PyResult<NodeIndices> {
    Ok(NodeIndices {
        nodes: longest_path(graph)?,
    })
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

/// Find the weakly connected components in a directed graph
///
/// :param PyDiGraph graph: The graph to find the weakly connected components
///     in
///
/// :returns: A list of sets where each set it a weakly connected component of
///     the graph
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, /)"]
pub fn weakly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<BTreeSet<usize>> {
    let mut seen: HashSet<NodeIndex> = HashSet::new();
    let mut out_vec: Vec<BTreeSet<usize>> = Vec::new();
    for node in graph.graph.node_indices() {
        if !seen.contains(&node) {
            // BFS node generator
            let mut component_set: BTreeSet<usize> = BTreeSet::new();
            let mut bfs_seen: HashSet<NodeIndex> = HashSet::new();
            let mut next_level: HashSet<NodeIndex> = HashSet::new();
            next_level.insert(node);
            while !next_level.is_empty() {
                let this_level = next_level;
                next_level = HashSet::new();
                for bfs_node in this_level {
                    if !bfs_seen.contains(&bfs_node) {
                        component_set.insert(bfs_node.index());
                        bfs_seen.insert(bfs_node);
                        for neighbor in
                            graph.graph.neighbors_undirected(bfs_node)
                        {
                            next_level.insert(neighbor);
                        }
                    }
                }
            }
            out_vec.push(component_set);
            seen.extend(bfs_seen);
        }
    }
    out_vec
}

/// Check if the graph is weakly connected
///
/// :param PyDiGraph graph: The graph to check if it is weakly connected
///
/// :returns: Whether the graph is weakly connected or not
/// :rtype: bool
///
/// :raises NullGraph: If an empty graph is passed in
#[pyfunction]
#[text_signature = "(graph, /)"]
pub fn is_weakly_connected(graph: &digraph::PyDiGraph) -> PyResult<bool> {
    if graph.graph.node_count() == 0 {
        return Err(NullGraph::new_err("Invalid operation on a NullGraph"));
    }
    Ok(weakly_connected_components(graph)[0].len() == graph.graph.node_count())
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

/// Return a new PyDiGraph by forming a union from two input PyDiGraph objects
///
/// The algorithm in this function operates in three phases:
///
///  1. Add all the nodes from  ``second`` into ``first``. operates in O(n),
///     with n being number of nodes in `b`.
///  2. Merge nodes from ``second`` over ``first`` given that:
///
///     - The ``merge_nodes`` is ``True``. operates in O(n^2), with n being the
///       number of nodes in ``second``.
///     - The respective node in ``second`` and ``first`` share the same
///       weight/data payload.
///
///  3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
///     parameter is ``True`` and the respective edge in ``second`` and
///     first`` share the same weight/data payload they will be merged
///     together.
///
///  :param PyDiGraph first: The first directed graph object
///  :param PyDiGraph second: The second directed graph object
///  :param bool merge_nodes: If set to ``True`` nodes will be merged between
///     ``second`` and ``first`` if the weights are equal.
///  :param bool merge_edges: If set to ``True`` edges will be merged between
///     ``second`` and ``first`` if the weights are equal.
///
///  :returns: A new PyDiGraph object that is the union of ``second`` and
///     ``first``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
///  :rtype: PyDiGraph
#[pyfunction]
#[text_signature = "(first, second, merge_nodes, merge_edges, /)"]
fn digraph_union(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<digraph::PyDiGraph> {
    let res =
        union::digraph_union(py, first, second, merge_nodes, merge_edges)?;
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
/// :rtype: NodeIndices
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[text_signature = "(graph, /)"]
fn topological_sort(graph: &digraph::PyDiGraph) -> PyResult<NodeIndices> {
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    Ok(NodeIndices {
        nodes: nodes.iter().map(|node| node.index()).collect(),
    })
}

fn dfs_edges<G>(graph: G, source: Option<usize>) -> Vec<(usize, usize)>
where
    G: GraphBase<NodeId = NodeIndex>
        + IntoNodeIdentifiers
        + NodeIndexable
        + IntoNeighbors
        + NodeCount
        + Visitable,
    <G as Visitable>::Map: VisitMap<NodeIndex>,
{
    let nodes: Vec<NodeIndex> = match source {
        Some(start) => vec![NodeIndex::new(start)],
        None => graph
            .node_identifiers()
            .map(|ind| NodeIndex::new(graph.to_index(ind)))
            .collect(),
    };
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut out_vec: Vec<(usize, usize)> = Vec::new();
    for start in nodes {
        if visited.contains(&start) {
            continue;
        }
        visited.insert(start);
        let mut children: Vec<NodeIndex> = graph.neighbors(start).collect();
        children.reverse();
        let mut stack: Vec<(NodeIndex, Vec<NodeIndex>)> =
            vec![(start, children)];
        // Used to track the last position in children vec across iterations
        let mut index_map: HashMap<NodeIndex, usize> = HashMap::new();
        index_map.insert(start, 0);
        while !stack.is_empty() {
            let temp_parent = stack.last().unwrap();
            let parent = temp_parent.0;
            let children = temp_parent.1.clone();
            let count = *index_map.get(&parent).unwrap();
            let mut found = false;
            let mut index = count;
            for child in &children[index..] {
                index += 1;
                if !visited.contains(&child) {
                    out_vec.push((parent.index(), child.index()));
                    visited.insert(*child);
                    let mut grandchildren: Vec<NodeIndex> =
                        graph.neighbors(*child).collect();
                    grandchildren.reverse();
                    stack.push((*child, grandchildren));
                    index_map.insert(*child, 0);
                    *index_map.get_mut(&parent).unwrap() = index;
                    found = true;
                    break;
                }
            }
            if !found || children.is_empty() {
                stack.pop();
            }
        }
    }
    out_vec
}

/// Get edge list in depth first order
///
/// :param PyDiGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[text_signature = "(graph, /, source=None)"]
fn digraph_dfs_edges(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    EdgeList {
        edges: dfs_edges(graph, source),
    }
}

/// Get edge list in depth first order
///
/// :param PyGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[text_signature = "(graph, /, source=None)"]
fn graph_dfs_edges(graph: &graph::PyGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: dfs_edges(graph, source),
    }
}

/// Return successors in a breadth-first-search from a source node.
///
/// The return format is ``[(Parent Node, [Children Nodes])]`` in a bfs order
/// from the source node provided.
///
/// :param PyDiGraph graph: The DAG to get the bfs_successors from
/// :param int node: The index of the dag node to get the bfs successors for
///
/// :returns: A list of nodes's data and their children in bfs order. The
///     BFSSuccessors class that is returned is a custom container class that
///     implements the sequence protocol. This can be used as a python list
///     with index based access.
/// :rtype: BFSSuccessors
#[pyfunction]
#[text_signature = "(graph, node, /)"]
fn bfs_successors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> PyResult<iterators::BFSSuccessors> {
    let index = NodeIndex::new(node);
    let mut bfs = Bfs::new(graph, index);
    let mut out_list: Vec<(PyObject, Vec<PyObject>)> = Vec::new();
    while let Some(nx) = bfs.next(graph) {
        let children = graph
            .graph
            .neighbors_directed(nx, petgraph::Direction::Outgoing);
        let mut succesors: Vec<PyObject> = Vec::new();
        for succ in children {
            succesors
                .push(graph.graph.node_weight(succ).unwrap().clone_ref(py));
        }
        if !succesors.is_empty() {
            out_list.push((
                graph.graph.node_weight(nx).unwrap().clone_ref(py),
                succesors,
            ));
        }
    }
    Ok(iterators::BFSSuccessors {
        bfs_successors: out_list,
        index: 0,
    })
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

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: dict
#[pyfunction]
#[text_signature = "(graph, start, k, edge_cost, /, goal=None)"]
fn digraph_k_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PyObject> {
    let out_goal = match goal {
        Some(g) => Some(NodeIndex::new(g)),
        None => None,
    };
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        Ok(res.extract(py)?)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    let out_dict = PyDict::new(py);
    for (index, length) in out_map {
        if (out_goal.is_some() && out_goal.unwrap() == index)
            || out_goal.is_none()
        {
            out_dict.set_item(index.index(), length)?;
        }
    }
    Ok(out_dict.into())
}

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: dict
#[pyfunction]
#[text_signature = "(graph, start, k, edge_cost, /, goal=None)"]
fn graph_k_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PyObject> {
    let out_goal = match goal {
        Some(g) => Some(NodeIndex::new(g)),
        None => None,
    };
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        Ok(res.extract(py)?)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    let out_dict = PyDict::new(py);
    for (index, length) in out_map {
        if (out_goal.is_some() && out_goal.unwrap() == index)
            || out_goal.is_none()
        {
            out_dict.set_item(index.index(), length)?;
        }
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

fn get_edge_iter_with_weights<G>(
    graph: G,
) -> impl Iterator<Item = (usize, usize, PyObject)>
where
    G: GraphBase
        + IntoEdgeReferences
        + IntoNodeIdentifiers
        + NodeIndexable
        + GraphProp
        + NodesRemoved,
    G: Data<NodeWeight = PyObject, EdgeWeight = PyObject>,
{
    let node_map: Option<HashMap<NodeIndex, usize>>;
    if graph.nodes_removed() {
        let mut node_hash_map: HashMap<NodeIndex, usize> = HashMap::new();
        for (count, node) in graph.node_identifiers().enumerate() {
            let index = NodeIndex::new(graph.to_index(node));
            node_hash_map.insert(index, count);
        }
        node_map = Some(node_hash_map);
    } else {
        node_map = None;
    }

    graph.edge_references().map(move |edge| {
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                let source_index =
                    NodeIndex::new(graph.to_index(edge.source()));
                let target_index =
                    NodeIndex::new(graph.to_index(edge.target()));
                i = *map.get(&source_index).unwrap();
                j = *map.get(&target_index).unwrap();
            }
            None => {
                i = graph.to_index(edge.source());
                j = graph.to_index(edge.target());
            }
        }
        (i, j, edge.weight().clone())
    })
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// :param PyGraph graph: The graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0")]
#[text_signature = "(graph, /, weight_fn=None, default_weight=1.0)"]
fn graph_floyd_warshall_numpy(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        mat[[j, i]] = mat[[j, i]].min(edge_weight);
    }

    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d_ijk = mat[[i, k]] + mat[[k, j]];
                if d_ijk < mat[[i, j]] {
                    mat[[i, j]] = d_ijk;
                }
            }
        }
    }
    Ok(mat.into_pyarray(py).into())
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
/// :param as_undirected: If set to true each directed edge will be treated as
///     bidirectional/undirected.
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(as_undirected = "false", default_weight = "1.0")]
#[text_signature = "(graph, /, weight_fn=None as_undirected=False, default_weight=1.0)"]
fn digraph_floyd_warshall_numpy(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();

    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        if as_undirected {
            mat[[j, i]] = mat[[j, i]].min(edge_weight);
        }
    }
    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d_ijk = mat[[i, k]] + mat[[k, j]];
                if d_ijk < mat[[i, j]] {
                    mat[[i, j]] = d_ijk;
                }
            }
        }
    }
    Ok(mat.into_pyarray(py).into())
}

/// Collect runs that match a filter function
///
/// A run is a path of nodes where there is only a single successor and all
/// nodes in the path match the given condition. Each node in the graph can
/// appear in only a single run.
///
/// :param PyDiGraph graph: The graph to find runs in
/// :param filter_fn: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not. If it returns
///     ``False`` it will skip that node.
///
/// :returns: a list of runs, where each run is a list of node data
///     payload/weight for the nodes in the run
/// :rtype: list
#[pyfunction]
#[text_signature = "(graph, filter)"]
fn collect_runs(
    py: Python,
    graph: &digraph::PyDiGraph,
    filter_fn: PyObject,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mut out_list: Vec<Vec<PyObject>> = Vec::new();
    let mut seen: HashSet<NodeIndex> = HashSet::new();

    let filter_node = |node: &PyObject| -> PyResult<bool> {
        let res = filter_fn.call1(py, (node,))?;
        Ok(res.extract(py)?)
    };

    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    for node in nodes {
        if !filter_node(&graph.graph[node])? || seen.contains(&node) {
            continue;
        }
        seen.insert(node);
        let mut group: Vec<PyObject> = vec![graph.graph[node].clone_ref(py)];
        let mut successors: Vec<NodeIndex> = graph
            .graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .collect();
        successors.dedup();

        while successors.len() == 1
            && filter_node(&graph.graph[successors[0]])?
            && !seen.contains(&successors[0])
        {
            group.push(graph.graph[successors[0]].clone_ref(py));
            seen.insert(successors[0]);
            successors = graph
                .graph
                .neighbors_directed(
                    successors[0],
                    petgraph::Direction::Outgoing,
                )
                .collect();
            successors.dedup();
        }
        if !group.is_empty() {
            out_list.push(group);
        }
    }
    Ok(out_list)
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
///
/// :raises InvalidNode: If a node index in ``first_layer`` is not in the graph
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
        let node_data = match dag.graph.node_weight(*layer_node) {
            Some(data) => data,
            None => {
                return Err(InvalidNode::new_err(format!(
                    "An index input in 'first_layer' {} is not a valid node index in the graph",
                    layer_node.index()),
                ))
            }
        };
        layer_node_data.push(node_data);
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

/// Get the distance matrix for a directed graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyDiGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
/// :param bool as_undirected: If set to ``True`` the input directed graph
///     will be treat as if each edge was bidirectional/undirected in the
///     output distance matrix.
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", as_undirected = "false")]
#[text_signature = "(graph, /, parallel_threshold=300, as_undirected=False)"]
pub fn digraph_distance_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    parallel_threshold: usize,
    as_undirected: bool,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::new();
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph
                    .graph
                    .neighbors_directed(node, petgraph::Direction::Outgoing)
                {
                    next_level.insert(v);
                }
                if as_undirected {
                    for v in graph
                        .graph
                        .neighbors_directed(node, petgraph::Direction::Incoming)
                    {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Get the distance matrix for an undirected graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``paralllel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300")]
#[text_signature = "(graph, /, parallel_threshold=300)"]
pub fn graph_distance_matrix(
    py: Python,
    graph: &graph::PyGraph,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::new();
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph.graph.neighbors(node) {
                    next_level.insert(v);
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return the adjacency matrix for a PyDiGraph object
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be the sum of the edges' weights.
///
/// :param PyDiGraph graph: The DiGraph used to generate the adjacency matrix
///     from
/// :param callable weight_fn: A callable object (function, lambda, etc) which
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
///     to cast the edge object as a float as the weight. If this is not
///     specified a default value (either ``default_weight`` or 1) will be used
///     for all edges.
/// :param float default_weight: If ``weight_fn`` is not used this can be
///     optionally used to specify a default weight to use for all edges.
///
///  :return: The adjacency matrix for the input dag as a numpy array
///  :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0")]
#[text_signature = "(graph, /, weight_fn=None, default_weight=1.0)"]
fn digraph_adjacency_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
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
/// :param weight_fn: A callable object (function, lambda, etc) which
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
///     to cast the edge object as a float as the weight. If this is not
///     specified a default value (either ``default_weight`` or 1) will be used
///     for all edges.
/// :param float default_weight: If ``weight_fn`` is not used this can be
///     optionally used to specify a default weight to use for all edges.
///
/// :return: The adjacency matrix for the input dag as a numpy array
/// :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0")]
#[text_signature = "(graph, /, weight_fn=None, default_weight=1.0)"]
fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
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
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
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
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
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

fn weight_callable(
    py: Python,
    weight_fn: &Option<PyObject>,
    weight: &PyObject,
    default: f64,
) -> PyResult<f64> {
    match weight_fn {
        Some(weight_fn) => {
            let res = weight_fn.call1(py, (weight,))?;
            res.extract(py)
        }
        None => Ok(default),
    }
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target to find a path to
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0)"]
pub fn graph_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = match target {
        Some(node) => Some(NodeIndex::new(node)),
        None => None,
    };
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
        Some(&mut paths),
    )?;

    let out_dict = PyDict::new(py);
    for (index, value) in paths {
        let int_index = index.index();
        if int_index == source {
            continue;
        }
        if (target.is_some() && target.unwrap() == int_index)
            || target.is_none()
        {
            out_dict.set_item(
                int_index,
                value
                    .iter()
                    .map(|index| index.index())
                    .collect::<Vec<usize>>(),
            )?;
        }
    }
    Ok(out_dict.into())
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyDiGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target path to find the path
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0, as_undirected=False)"]
pub fn digraph_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    as_undirected: bool,
) -> PyResult<PyObject> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = match target {
        Some(node) => Some(NodeIndex::new(node)),
        None => None,
    };
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    if as_undirected {
        dijkstra::dijkstra(
            // TODO: Use petgraph undirected adapter after
            // https://github.com/petgraph/petgraph/pull/318 is available in
            // a petgraph release.
            &graph.to_undirected(py),
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    } else {
        dijkstra::dijkstra(
            graph,
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    }

    let out_dict = PyDict::new(py);
    for (index, value) in paths {
        let int_index = index.index();
        if int_index == source {
            continue;
        }
        if (target.is_some() && target.unwrap() == int_index)
            || target.is_none()
        {
            out_dict.set_item(
                int_index,
                value
                    .iter()
                    .map(|index| index.index())
                    .collect::<Vec<usize>>(),
            )?;
        }
    }
    Ok(out_dict.into())
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

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
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

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
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
/// :rtype: NodeIndices
#[pyfunction]
#[text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)"]
fn graph_astar_shortest_path(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
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
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
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
/// :rtype: NodeIndices
#[pyfunction]
#[text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)"]
fn digraph_astar_shortest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
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
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
}

/// Return a :math:`G_{np}` directed random graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete directed graph has :math:`n (n - 1)` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1))`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
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
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableDiGraph::<PyObject, PyObject>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if probability < 0.0 || probability > 1.0 {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in 0..num_nodes {
                    if u != v {
                        // exclude self-loops
                        let u_index = NodeIndex::new(u as usize);
                        let v_index = NodeIndex::new(v as usize);
                        inner_graph.add_edge(u_index, v_index, py.None());
                    }
                }
            }
        } else {
            let mut v: isize = 0;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
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
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)/2` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)/2`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete undirected graph has :math:`n (n - 1)/2` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1)/2)`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
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
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if probability < 0.0 || probability > 1.0 {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in u + 1..num_nodes {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        } else {
            let mut v: isize = 1;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
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
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
    };
    Ok(graph)
}

/// Return a :math:`G_{nm}` of a directed graph
///
/// Generates a random directed graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
#[pyfunction]
#[text_signature = "(num_nodes, num_edges, seed=None, /)"]
pub fn directed_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableDiGraph::<PyObject, PyObject>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) {
        for u in 0..num_nodes {
            for v in 0..num_nodes {
                // avoid self-loops
                if u != v {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
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

/// Return a :math:`G_{nm}` of an undirected graph
///
/// Generates a random undirected graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)/2`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)/2` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph

#[pyfunction]
#[text_signature = "(num_nodes, probability, seed=None, /)"]
pub fn undirected_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) / 2 {
        for u in 0..num_nodes {
            for v in u + 1..num_nodes {
                let u_index = NodeIndex::new(u as usize);
                let v_index = NodeIndex::new(v as usize);
                inner_graph.add_edge(u_index, v_index, py.None());
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
        }
    }
    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
    };
    Ok(graph)
}

/// Return a list of cycles which form a basis for cycles of a given PyGraph
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive or of the edges.
///
/// This is adapted from algorithm CACM 491 [1]_.
///
/// :param PyGraph graph: The graph to find the cycle basis in
/// :param int root: Optional index for starting node for basis
///
/// :returns: A list of cycle lists. Each list is a list of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: list
///
/// .. [1] Paton, K. An algorithm for finding a fundamental set of
///    cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.
#[pyfunction]
#[text_signature = "(graph, /, root=None)"]
pub fn cycle_basis(
    graph: &graph::PyGraph,
    root: Option<usize>,
) -> Vec<Vec<usize>> {
    let mut root_node = root;
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    while !graph_nodes.is_empty() {
        let temp_value: NodeIndex;
        // If root_node is not set get an arbitrary node from the set of graph
        // nodes we've not "examined"
        let root_index = match root_node {
            Some(root_value) => NodeIndex::new(root_value),
            None => {
                temp_value = *graph_nodes.iter().next().unwrap();
                graph_nodes.remove(&temp_value);
                temp_value
            }
        };
        // Stack (ie "pushdown list") of vertices already in the spanning tree
        let mut stack: Vec<NodeIndex> = Vec::new();
        stack.push(root_index);
        // Map of node index to predecessor node index
        let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        pred.insert(root_index, root_index);
        // Set of examined nodes during this iteration
        let mut used: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        used.insert(root_index, HashSet::new());
        // Walk the spanning tree
        while !stack.is_empty() {
            // Use the last element added so that cycles are easier to find
            let z = stack.pop().unwrap();
            for neighbor in graph.graph.neighbors(z) {
                // A new node was encountered:
                if !used.contains_key(&neighbor) {
                    pred.insert(neighbor, z);
                    stack.push(neighbor);
                    let mut temp_set: HashSet<NodeIndex> = HashSet::new();
                    temp_set.insert(z);
                    used.insert(neighbor, temp_set);
                // A self loop:
                } else if z == neighbor {
                    let mut cycle: Vec<usize> = Vec::new();
                    cycle.push(z.index());
                    cycles.push(cycle);
                // A cycle was found:
                } else if !used.get(&z).unwrap().contains(&neighbor) {
                    let pn = used.get(&neighbor).unwrap();
                    let mut cycle: Vec<NodeIndex> = Vec::new();
                    cycle.push(neighbor);
                    cycle.push(z);
                    let mut p = pred.get(&z).unwrap();
                    while !pn.contains(p) {
                        cycle.push(*p);
                        p = pred.get(p).unwrap();
                    }
                    cycle.push(*p);
                    cycles.push(cycle.iter().map(|x| x.index()).collect());
                    let neighbor_set = used.get_mut(&neighbor).unwrap();
                    neighbor_set.insert(z);
                }
            }
        }
        let mut temp_hashset: HashSet<NodeIndex> = HashSet::new();
        for key in pred.keys() {
            temp_hashset.insert(*key);
        }
        graph_nodes = graph_nodes.difference(&temp_hashset).copied().collect();
        root_node = None;
    }
    cycles
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

/// Return the first cycle encountered during DFS of a given PyDiGraph,
/// empty list is returned if no cycle is found
///
/// :param PyDiGraph graph: The graph to find the cycle in
/// :param int source: Optional index to find a cycle for. If not specified an
///     arbitrary node will be selected from the graph.
///
/// :returns: A list describing the cycle. The index of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: EdgeList
#[pyfunction]
#[text_signature = "(graph, /, source=None)"]
pub fn digraph_find_cycle(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
    let mut cycle: Vec<(usize, usize)> = Vec::new();
    let temp_value: NodeIndex;
    // If source is not set get an arbitrary node from the set of graph
    // nodes we've not "examined"
    let source_index = match source {
        Some(source_value) => NodeIndex::new(source_value),
        None => {
            temp_value = *graph_nodes.iter().next().unwrap();
            graph_nodes.remove(&temp_value);
            temp_value
        }
    };

    // Stack (ie "pushdown list") of vertices already in the spanning tree
    let mut stack: Vec<NodeIndex> = Vec::new();
    stack.push(source_index);
    // map to store parent of a node
    let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // a node is in the visiting set if at least one of its child is unexamined
    let mut visiting = HashSet::new();
    // a node is in visited set if all of its children have been examined
    let mut visited = HashSet::new();
    while !stack.is_empty() {
        let mut z = *stack.last().unwrap();
        visiting.insert(z);

        let children = graph
            .graph
            .neighbors_directed(z, petgraph::Direction::Outgoing);

        for child in children {
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((z.index(), child.index()));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((pred[&z].index(), z.index()));
                    z = pred[&z];
                }
                return EdgeList { edges: cycle };
            }
            //if an unexplored node is encountered
            if !visited.contains(&child) {
                stack.push(child);
                pred.insert(child, z);
            }
        }

        let top = *stack.last().unwrap();
        //if no further children and explored, move to visited
        if top.index() == z.index() {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    EdgeList { edges: cycle }
}

// The provided node is invalid.
create_exception!(retworkx, InvalidNode, PyException);
// Performing this operation would result in trying to add a cycle to a DAG.
create_exception!(retworkx, DAGWouldCycle, PyException);
// There is no edge present between the provided nodes.
create_exception!(retworkx, NoEdgeBetweenNodes, PyException);
// The specified Directed Graph has a cycle and can't be treated as a DAG.
create_exception!(retworkx, DAGHasCycle, PyException);
// No neighbors found matching the provided predicate.
create_exception!(retworkx, NoSuitableNeighbors, PyException);
// Invalid operation on a null graph
create_exception!(retworkx, NullGraph, PyException);
// No path was found between the specified nodes.
create_exception!(retworkx, NoPathFound, PyException);

#[pymodule]
fn retworkx(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InvalidNode", py.get_type::<InvalidNode>())?;
    m.add("DAGWouldCycle", py.get_type::<DAGWouldCycle>())?;
    m.add("NoEdgeBetweenNodes", py.get_type::<NoEdgeBetweenNodes>())?;
    m.add("DAGHasCycle", py.get_type::<DAGHasCycle>())?;
    m.add("NoSuitableNeighbors", py.get_type::<NoSuitableNeighbors>())?;
    m.add("NoPathFound", py.get_type::<NoPathFound>())?;
    m.add("NullGraph", py.get_type::<NullGraph>())?;
    m.add_wrapped(wrap_pyfunction!(bfs_successors))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_weakly_connected))?;
    m.add_wrapped(wrap_pyfunction!(is_directed_acyclic_graph))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(digraph_union))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic_node_match))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(collect_runs))?;
    m.add_wrapped(wrap_pyfunction!(layers))?;
    m.add_wrapped(wrap_pyfunction!(graph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_greedy_color))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(cycle_basis))?;
    m.add_wrapped(wrap_pyfunction!(strongly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dfs_edges))?;
    m.add_wrapped(wrap_pyfunction!(graph_dfs_edges))?;
    m.add_wrapped(wrap_pyfunction!(digraph_find_cycle))?;
    m.add_wrapped(wrap_pyfunction!(digraph_k_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_k_shortest_path_lengths))?;
    m.add_class::<digraph::PyDiGraph>()?;
    m.add_class::<graph::PyGraph>()?;
    m.add_class::<iterators::BFSSuccessors>()?;
    m.add_class::<iterators::NodeIndices>()?;
    m.add_class::<iterators::EdgeList>()?;
    m.add_class::<iterators::WeightedEdgeList>()?;
    m.add_wrapped(wrap_pymodule!(generators))?;
    Ok(())
}
