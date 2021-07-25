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

#![allow(clippy::float_cmp)]

mod dfs_edges;

use super::{digraph, graph, iterators, DAGHasCycle};

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, NodeCount, Reversed};

use crate::iterators::{EdgeList, NodeIndices};

/// Return the topological sort of node indexes from the provided graph
///
/// :param PyDiGraph graph: The DAG to get the topological sort on
///
/// :returns: A list of node indices topologically sorted.
/// :rtype: NodeIndices
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
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
#[pyo3(text_signature = "(graph, /, source=None)")]
fn digraph_dfs_edges(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    EdgeList {
        edges: dfs_edges::dfs_edges(graph, source, graph.graph.edge_count()),
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
#[pyo3(text_signature = "(graph, /, source=None)")]
fn graph_dfs_edges(graph: &graph::PyGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: dfs_edges::dfs_edges(graph, source, graph.graph.edge_count()),
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
#[pyo3(text_signature = "(graph, node, /)")]
fn bfs_successors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> iterators::BFSSuccessors {
    let index = NodeIndex::new(node);
    let mut bfs = Bfs::new(graph, index);
    let mut out_list: Vec<(PyObject, Vec<PyObject>)> =
        Vec::with_capacity(graph.node_count());
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
    iterators::BFSSuccessors {
        bfs_successors: out_list,
    }
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
#[pyo3(text_signature = "(graph, node, /)")]
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
#[pyo3(text_signature = "(graph, node, /)")]
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
#[pyo3(text_signature = "(dag, key, /)")]
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
    let node_count = dag.node_count();
    let mut in_degree_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_count);
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
    let mut zero_indegree = BinaryHeap::with_capacity(node_count);
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
    let mut out_list: Vec<&PyObject> = Vec::with_capacity(node_count);
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
#[pyo3(text_signature = "(graph, filter)")]
fn collect_runs(
    py: Python,
    graph: &digraph::PyDiGraph,
    filter_fn: PyObject,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mut out_list: Vec<Vec<PyObject>> = Vec::new();
    let mut seen: HashSet<NodeIndex> =
        HashSet::with_capacity(graph.node_count());

    let filter_node = |node: &PyObject| -> PyResult<bool> {
        let res = filter_fn.call1(py, (node,))?;
        res.extract(py)
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
