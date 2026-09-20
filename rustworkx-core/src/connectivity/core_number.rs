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

use petgraph::Direction::{Incoming, Outgoing};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount};

use crate::dictmap::{DictMap, InitWithHasher};

/// Return the core number for each node in the graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
///
/// The function implicitly assumes that there are no parallel edges
/// or self loops. It may produce incorrect/unexpected results if the
/// input graph has self loops or parallel edges.
///
/// Arguments:
///
/// * `graph` - The graph in which to find the core numbers.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::core_number;
///
/// let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
/// let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
/// let res: Vec<(usize, usize)> = core_number(graph)
///     .iter()
///     .map(|(k, v)| (k.index(), *v))
///     .collect();
/// assert_eq!(res, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
/// ```
pub fn core_number<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: GraphBase + NodeCount,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoNodeIdentifiers + IntoNeighborsDirected,
    G::NodeId: Eq + Hash + Send + Sync,
{
    let node_num = graph.node_count();
    if node_num == 0 {
        return DictMap::new();
    }

    // DictMap provides compact working indices without requiring
    // NodeIndexable or allocating up to the largest node ID. Keep its insertion
    // order unchanged so the returned map still follows node_identifiers().
    let mut cores: DictMap<G::NodeId, usize> =
        graph.node_identifiers().map(|node| (node, 0)).collect();
    // Store adjacency in one buffer. Each row's dense index is a fresh marker,
    // so incoming/outgoing duplicates are removed without sorting or hashing
    // another set. The sentinel cannot equal a live row index, including zero.
    let mut neighbors = Vec::new();
    let mut offsets = Vec::with_capacity(node_num + 1);
    let mut seen = vec![usize::MAX; node_num];
    offsets.push(0);
    for (row, &node) in cores.keys().enumerate() {
        for neighbor in graph
            .neighbors_directed(node, Incoming)
            .chain(graph.neighbors_directed(node, Outgoing))
        {
            let index = cores.get_index_of(&neighbor).unwrap();
            // Match the existing neighbor set, including reciprocal edges.
            if seen[index] != row {
                seen[index] = row;
                neighbors.push(index);
            }
        }
        offsets.push(neighbors.len());
    }
    drop(seen);
    let mut degree: Vec<_> = offsets.windows(2).map(|row| row[1] - row[0]).collect();
    let mut bins = vec![0; degree.iter().copied().max().unwrap() + 1];
    for &value in &degree {
        bins[value] += 1;
    }
    let mut start = 0;
    for count in &mut bins {
        let next = start + *count;
        *count = start;
        start = next;
    }

    // Counting sort puts every vertex in its degree bucket. Placement advances
    // bins to bucket ends; shifting them restores starts without a second buffer.
    let mut positions = vec![0; node_num];
    let mut vertices = vec![0; node_num];
    for (node, &value) in degree.iter().enumerate() {
        positions[node] = bins[value];
        vertices[bins[value]] = node;
        bins[value] += 1;
    }
    for value in (1..bins.len()).rev() {
        bins[value] = bins[value - 1];
    }
    bins[0] = 0;

    for order in 0..node_num {
        let node = vertices[order];
        for &neighbor in &neighbors[offsets[node]..offsets[node + 1]] {
            if degree[neighbor] > degree[node] {
                let neighbor_degree = degree[neighbor];
                let position = positions[neighbor];
                let bucket_start = bins[neighbor_degree];
                let bucket_node = vertices[bucket_start];
                vertices.swap(position, bucket_start);
                positions[neighbor] = bucket_start;
                positions[bucket_node] = position;
                bins[neighbor_degree] += 1;
                degree[neighbor] -= 1;
            }
        }
    }
    for (value, core) in cores.values_mut().zip(degree) {
        *value = core;
    }
    cores
}

#[cfg(test)]
mod tests {
    use crate::connectivity::core_number;
    use petgraph::prelude::*;

    // Exercise IDs that cannot be used as dense vector indices.
    #[test]
    fn test_graph_map_node_ids_and_order() {
        let mut graph = DiGraphMap::<&str, ()>::new();
        let nodes = ["isolated", "tail", "z", "a", "middle"];
        for node in nodes {
            graph.add_node(node);
        }
        for (a, b) in [
            ("z", "a"),
            ("a", "z"),
            ("a", "middle"),
            ("middle", "z"),
            ("tail", "middle"),
        ] {
            graph.add_edge(a, b, ());
        }
        let result = core_number(&graph);
        assert_eq!(result.keys().copied().collect::<Vec<_>>(), nodes);
        assert_eq!(
            result.values().copied().collect::<Vec<_>>(),
            [0, 1, 2, 2, 2]
        );
    }

    #[test]
    fn test_deleted_and_reused_node_indices() {
        fn check<Ty: petgraph::EdgeType>() {
            for stride in [2, 1000] {
                let mut graph = StableGraph::<(), (), Ty>::default();
                for _ in 0..7 * stride {
                    graph.add_node(());
                }
                for i in 0..7 * stride {
                    if i % stride != 0 {
                        graph.remove_node(NodeIndex::new(i));
                    }
                }
                graph.remove_node(NodeIndex::new(stride));
                assert_eq!(graph.add_node(()), NodeIndex::new(stride));
                for (a, b) in [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5)] {
                    graph.add_edge(NodeIndex::new(a * stride), NodeIndex::new(b * stride), ());
                }
                let result = core_number(&graph);
                assert_eq!(
                    result.keys().map(|node| node.index()).collect::<Vec<_>>(),
                    (0..7).map(|i| i * stride).collect::<Vec<_>>()
                );
                assert_eq!(
                    result.values().copied().collect::<Vec<_>>(),
                    [2, 2, 2, 1, 1, 1, 0]
                );
            }
        }
        check::<Directed>();
        check::<Undirected>();
    }

    #[test]
    fn test_reciprocal_edges_count_one_neighbor() {
        let graph = DiGraph::<(), ()>::from_edges([(0, 1), (1, 0)]);
        let result = core_number(&graph);
        assert_eq!(result.values().copied().collect::<Vec<_>>(), [1, 1]);
    }

    // Independent oracle: a node's core number is the largest minimum degree
    // among all vertex subsets containing it. No degree buckets or peeling.
    fn subset_core_numbers(adjacency: &[Vec<bool>]) -> Vec<usize> {
        let n = adjacency.len();
        let mut result = vec![0; n];
        for subset in 1..1_usize << n {
            let degree = (0..n)
                .filter(|&a| subset & (1 << a) != 0)
                .map(|a| {
                    (0..n)
                        .filter(|&b| subset & (1 << b) != 0 && (adjacency[a][b] || adjacency[b][a]))
                        .count()
                })
                .min()
                .unwrap();
            for (node, core) in result.iter_mut().enumerate() {
                if subset & (1 << node) != 0 {
                    *core = (*core).max(degree);
                }
            }
        }
        result
    }

    #[test]
    fn test_all_four_node_directed_graphs() {
        let edges: Vec<_> = (0..4)
            .flat_map(|a| (0..4).filter(move |&b| a != b).map(move |b| (a, b)))
            .collect();
        for mask in 0..1_usize << edges.len() {
            let mut graph = DiGraph::<(), ()>::new();
            let mut adjacency = vec![vec![false; 4]; 4];
            for _ in 0..4 {
                graph.add_node(());
            }
            for (bit, &(a, b)) in edges.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    graph.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
                    adjacency[a][b] = true;
                }
            }
            let expected = subset_core_numbers(&adjacency);
            let actual = core_number(&graph);
            assert_eq!(
                actual.values().copied().collect::<Vec<_>>(),
                expected,
                "edge mask {mask}"
            );
        }
    }

    // Reuse the same graph through mixed core levels, sparse rows and
    // empty adjacency. Working state must belong to one call, not a previous graph.
    #[test]
    fn test_irregular_graphs_and_repeated_calls() {
        fn check<Ty: petgraph::EdgeType>() {
            let mixed = [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 3),
                (5, 6),
            ];
            let star = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)];
            for stride in [1, 3] {
                let mut graph = StableGraph::<(), (), Ty>::default();
                for _ in 0..8 * stride {
                    graph.add_node(());
                }
                for node in 0..8 * stride {
                    if node % stride != 0 {
                        graph.remove_node(NodeIndex::new(node));
                    }
                }
                for edges in [&[][..], &mixed[..], &star[..], &mixed[..], &[][..]] {
                    graph.clear_edges();
                    let mut adjacency = vec![vec![false; 8]; 8];
                    for &(source, target) in edges {
                        graph.add_edge(
                            NodeIndex::new(source * stride),
                            NodeIndex::new(target * stride),
                            (),
                        );
                        adjacency[source][target] = true;
                        if Ty::is_directed() && (source + target) % 2 == 0 {
                            graph.add_edge(
                                NodeIndex::new(target * stride),
                                NodeIndex::new(source * stride),
                                (),
                            );
                        }
                    }
                    let expected = subset_core_numbers(&adjacency);
                    for _ in 0..2 {
                        let actual = core_number(&graph);
                        assert_eq!(
                            actual.keys().map(|node| node.index()).collect::<Vec<_>>(),
                            (0..8).map(|node| node * stride).collect::<Vec<_>>()
                        );
                        assert_eq!(actual.values().copied().collect::<Vec<_>>(), expected);
                    }
                }
            }
        }
        check::<Directed>();
        check::<Undirected>();
    }

    #[test]
    fn test_neighbor_set_compatibility() {
        // Preserve the existing set-based behavior for unsupported loops and
        // parallel edges. In particular, a self-neighbor in row zero occurs once.
        let mut graph = DiGraph::<(), ()>::new();
        for _ in 0..4 {
            graph.add_node(());
        }
        graph.extend_with_edges([(0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (2, 2)]);
        for _ in 0..3 {
            let actual = core_number(&graph);
            assert_eq!(actual.values().copied().collect::<Vec<_>>(), [1, 1, 1, 0]);
        }
    }

    #[test]
    fn test_directed_empty() {
        let graph = DiGraph::<i32, i32>::new();
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![]);
    }

    #[test]
    fn test_directed_all_0() {
        let mut graph = DiGraph::<i32, i32>::new();
        for _ in 0..4 {
            graph.add_node(0);
        }
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
    }

    #[test]
    fn test_directed_all_3() {
        let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let graph = DiGraph::<i32, i32>::from_edges(edge_list);
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn test_directed_paper_example() {
        // This is the example graph in Figure 1 from Batagelj and
        // Zaversnik's paper titled An O(m) Algorithm for Cores
        // Decomposition of Networks, 2003,
        // http://arXiv.org/abs/cs/0310049.  With nodes labeled as
        // shown, the 3-core is given by nodes 0-7, the 2-core by nodes
        // 8-15, the 1-core by nodes 16-19 and node 20 is in the
        // 0-core.
        let edge_list = [
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 4),
            (1, 6),
            (1, 7),
            (2, 3),
            (3, 5),
            (2, 5),
            (5, 6),
            (4, 6),
            (4, 7),
            (6, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (8, 9),
            (0, 10),
            (1, 10),
            (1, 11),
            (10, 11),
            (12, 13),
            (13, 15),
            (14, 15),
            (12, 14),
            (8, 19),
            (11, 16),
            (11, 17),
            (12, 18),
        ];
        let mut example_core = vec![];
        for i in 0..8 {
            example_core.push((i, 3));
        }
        for i in 8..16 {
            example_core.push((i, 2));
        }
        for i in 16..20 {
            example_core.push((i, 1));
        }
        example_core.push((20, 0));
        let mut graph = DiGraph::<i32, i32>::from_edges(edge_list);
        graph.add_node(0);
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, example_core);
    }
}
