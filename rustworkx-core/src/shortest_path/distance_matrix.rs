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

use std::hash::Hash;

use hashbrown::{HashMap, HashSet};

use ndarray::prelude::*;
use petgraph::visit::{
    GraphProp, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use rayon::prelude::*;

/// Get the distance matrix for a graph
///
/// The generated distance matrix assumes the edge weight for all edges is
/// 1.0 and returns a matrix.
///
/// This function computes the distance matrix for a graph assuming compact
/// indices. Normally the [`NodeCompactIndexable`] trait would be used to enforce
/// this, but it is intentionally not set on this function. This enables the
/// user to assert their normally not [`NodeCompactIndexable`] graph type is
/// currently compact when calling. For example, if you have a [`StableGraph`]
/// that has no removals you can call this function and avoid the overhead of
/// mapping. If the input graph is determined to not be compact this function
/// will panic. If you have a graph type that is know to not be compact, or
/// you're unsure and it is does not implement [`NodeCompactIndexable`] you should
/// use [`distance_matrix_compacted`] instead.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of `parallel_threshold`. If the function
/// will be running in parallel the env var
/// `RAYON_NUM_THREADS` can be used to adjust how many threads will be used.
///
/// # Arguments:
///
/// * graph - The graph object to compute the distance matrix for.
/// * parallel_threshold - The threshold in number of nodes to run this function in parallel.
///   If `graph` has fewer nodes than this the algorithm will run serially. A good default
///   to use for this is 300.
/// * as_undirected - If the input graph is directed and this is set to true the output
///   matrix generated
/// * null_value - The value to use for the absence of a path in the graph.
///
/// # Returns
///  
/// A 2d ndarray [`Array`] of the distance matrix
///
/// # Example
///
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::shortest_path::distance_matrix;
/// use ndarray::{array, Array2};
///
/// let graph = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
///     (0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
/// ]);
/// let distance_matrix = distance_matrix(&graph, 300, false, 0.);
/// let expected: Array2<f64> = array![
///     [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
///     [1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0],
///     [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0],
///     [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
///     [3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
///     [2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0],
///     [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
/// ];
/// assert_eq!(distance_matrix, expected)
/// ```
pub fn distance_matrix<G>(
    graph: G,
    parallel_threshold: usize,
    as_undirected: bool,
    null_value: f64,
) -> Array2<f64>
where
    G: Sync + IntoNeighborsDirected + NodeCount + NodeIndexable + IntoNodeIdentifiers + GraphProp,
    G::NodeId: Hash + Eq + Sync,
{
    let n = graph.node_count();
    if n != graph.node_bound() {
        panic!(concat!(
            "The specified graph is not compact this will generate an invalid output. Either ",
            "use distance_matrix_compacted() or compact the graph"
        ))
    }
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<G::NodeId, usize> = HashMap::with_capacity(n);
        let start_index = graph.from_index(index);
        let mut level = 0;
        let mut next_level: HashSet<G::NodeId> = HashSet::with_capacity(1);
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<G::NodeId> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[graph.to_index(v)] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                    next_level.insert(v);
                }
                if graph.is_directed() && as_undirected {
                    for v in graph.neighbors_directed(node, petgraph::Direction::Incoming) {
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
    matrix
}

/// Get the distance matrix for a graph
///
/// The generated distance matrix assumes the edge weight for all edges is
/// 1.0 and returns a matrix.
///
/// This function computes the distance matrix for a graph assuming the graph
/// indices are not compact. It tracks a mapping of the indicies position in
/// the identifier list and uses that for the output matrix. For example, if
/// you have a [`StableGraph`] that has removals and your node identifiers are:
/// `[NodeIndex(2), NodeIndex(3), NodeIndex(6)]` the output matrix will use
/// index 0 for `NodeIndex(2)`, 1 for `NodeIndex(3), and 2 for `NodeIndex(6)`.
/// If you have a graph type that is known to be compact (i.e. it implements
/// [`NodeCompactIndexable`]) or you're certain your graph is compact you can
/// use [`distance_matrix`] instead which avoids the overhead of maintaining the
/// mapping.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of `parallel_threshold`. If the function
/// will be running in parallel the env var
/// `RAYON_NUM_THREADS` can be used to adjust how many threads will be used.
///
/// # Arguments:
///
/// * graph - The graph object to compute the distance matrix for.
/// * parallel_threshold - The threshold in number of nodes to run this function in parallel.
///   If `graph` has fewer nodes than this the algorithm will run serially. A good default
///   to use for this is 300.
/// * as_undirected - If the input graph is directed and this is set to true the output
///   matrix generated
/// * null_value - The value to use for the absence of a path in the graph.
///
/// # Returns
///  
/// A 2d ndarray [`Array`] of the distance matrix
///
/// # Example
///
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::shortest_path::distance_matrix_compacted;
/// use rustworkx_core::generators::path_graph;
/// use ndarray::{array, Array2};
///
/// let mut graph: petgraph::stable_graph::StableDiGraph<(), ()> = path_graph(
///     Some(4),
///     None,
///     || {()},
///     || {()},
///     false,
/// ).unwrap();
/// graph.remove_node(0.into());
/// let distance_matrix = distance_matrix_compacted(&graph, 300, false, 0.);
/// let expected: Array2<f64> = array![
///     [0.0, 1.0, 2.0],
///     [0.0, 0.0, 1.0],
///     [0.0, 0.0, 0.0],
/// ];
/// assert_eq!(distance_matrix, expected)
/// ```

pub fn distance_matrix_compacted<G>(
    graph: G,
    parallel_threshold: usize,
    as_undirected: bool,
    null_value: f64,
) -> Array2<f64>
where
    G: Sync + IntoNeighborsDirected + NodeCount + IntoNodeIdentifiers + GraphProp,
    G::NodeId: Hash + Eq + Sync,
{
    let node_map: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    let node_map_inv: Vec<G::NodeId> = graph.node_identifiers().collect();

    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<G::NodeId, usize> = HashMap::with_capacity(n);
        let start_index = node_map_inv[index];
        let mut level = 0;
        let mut next_level: HashSet<G::NodeId> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<G::NodeId> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[node_map[&v]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                    next_level.insert(v);
                }
                if graph.is_directed() && as_undirected {
                    for v in graph.neighbors_directed(node, petgraph::Direction::Incoming) {
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
    matrix
}
