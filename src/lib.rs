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

mod dag_isomorphism;
mod digraph;
mod graph;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use hashbrown::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{Bfs, IntoEdgeReferences, NodeIndexable, Reversed};

use ndarray::prelude::*;
use numpy::IntoPyArray;

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

#[pyfunction]
fn dag_longest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> PyResult<PyObject> {
    let path = longest_path(graph)?;
    Ok(PyList::new(py, path).into())
}

#[pyfunction]
fn dag_longest_path_length(graph: &digraph::PyDiGraph) -> PyResult<usize> {
    let path = longest_path(graph)?;
    if path.is_empty() {
        return Ok(0);
    }
    let path_length: usize = path.len() - 1;
    Ok(path_length)
}

#[pyfunction]
fn number_weakly_connected_components(graph: &digraph::PyDiGraph) -> usize {
    algo::connected_components(graph)
}

#[pyfunction]
fn is_directed_acyclic_graph(graph: &digraph::PyDiGraph) -> bool {
    let cycle_detected = algo::is_cyclic_directed(graph);
    !cycle_detected
}

#[pyfunction]
fn is_isomorphic(
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> PyResult<bool> {
    let res = dag_isomorphism::is_isomorphic(first, second)?;
    Ok(res)
}

#[pyfunction]
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
        first,
        second,
        compare_nodes,
        compare_edges,
    )?;
    Ok(res)
}

#[pyfunction]
fn topological_sort(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> PyResult<PyObject> {
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    let mut out: Vec<usize> = Vec::new();
    for node in nodes {
        out.push(node.index());
    }
    Ok(PyList::new(py, out).into())
}

#[pyfunction]
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

#[pyfunction]
fn ancestors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let reverse_graph = Reversed(graph);
    let res = algo::dijkstra(reverse_graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    Ok(out_set.to_object(py))
}

#[pyfunction]
fn descendants(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let res = algo::dijkstra(graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    Ok(out_set.to_object(py))
}

#[pyfunction]
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

// Find the shortest path lengths between all pairs of nodes using Floyd's
// algorithm
// Note: Edge weights are assumed to be 1
#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
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

#[pymodule]
fn retworkx(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
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
    m.add_class::<digraph::PyDiGraph>()?;
    m.add_class::<graph::PyGraph>()?;
    Ok(())
}

create_exception!(retworkx, DAGWouldCycle, Exception);
create_exception!(retworkx, NoEdgeBetweenNodes, Exception);
create_exception!(retworkx, DAGHasCycle, Exception);
create_exception!(retworkx, NoSuitableNeighbors, Exception);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
