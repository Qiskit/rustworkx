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

use hashbrown::HashMap;

use fixedbitset::FixedBitSet;
use ndarray::prelude::*;
use petgraph::visit::{
    GraphProp, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use petgraph::{Incoming, Outgoing};
use rayon::prelude::*;

/// Get the distance matrix for a graph
///
/// The generated distance matrix assumes the edge weight for all edges is
/// 1.0 and returns a matrix.
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
    let node_map: HashMap<G::NodeId, usize> = if n != graph.node_bound() {
        graph
            .node_identifiers()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect()
    } else {
        HashMap::new()
    };
    let node_map_inv: Vec<G::NodeId> = if n != graph.node_bound() {
        graph.node_identifiers().collect()
    } else {
        Vec::new()
    };
    let mut node_map_fn: Box<dyn FnMut(G::NodeId) -> usize> = if n != graph.node_bound() {
        Box::new(|n: G::NodeId| -> usize { node_map[&n] })
    } else {
        Box::new(|n: G::NodeId| -> usize { graph.to_index(n) })
    };
    let mut reverse_node_map: Box<dyn FnMut(usize) -> G::NodeId> = if n != graph.node_bound() {
        Box::new(|n: usize| -> G::NodeId { node_map_inv[n] })
    } else {
        Box::new(|n: usize| -> G::NodeId { graph.from_index(n) })
    };
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let neighbors = if as_undirected {
        (0..n)
            .map(|index| {
                graph
                    .neighbors_directed(reverse_node_map(index), Incoming)
                    .chain(graph.neighbors_directed(reverse_node_map(index), Outgoing))
                    .map(&mut node_map_fn)
                    .collect::<FixedBitSet>()
            })
            .collect::<Vec<_>>()
    } else {
        (0..n)
            .map(|index| {
                graph
                    .neighbors(reverse_node_map(index))
                    .map(&mut node_map_fn)
                    .collect::<FixedBitSet>()
            })
            .collect::<Vec<_>>()
    };
    let bfs_traversal = |start: usize, mut row: ArrayViewMut1<f64>| {
        let mut distance = 0.0;
        let mut seen = FixedBitSet::with_capacity(n);
        let mut next = FixedBitSet::with_capacity(n);
        let mut cur = FixedBitSet::with_capacity(n);
        cur.put(start);
        while !cur.is_clear() {
            next.clear();
            for found in cur.ones() {
                row[[found]] = distance;
                next |= &neighbors[found];
            }
            seen.union_with(&cur);
            next.difference_with(&seen);
            distance += 1.0;
            ::std::mem::swap(&mut cur, &mut next);
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
