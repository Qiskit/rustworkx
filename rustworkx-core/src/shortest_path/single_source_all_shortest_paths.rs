use crate::dictmap::{DictMap, InitWithHasher};
use crate::shortest_path::dijkstra;
use petgraph::visit::{EdgeRef, IntoEdgesDirected, IntoNodeIdentifiers, NodeIndexable, Visitable};
use petgraph::Direction::Incoming;
use std::hash::Hash;

type AllShortestPathsMap<N> = DictMap<N, Vec<Vec<N>>>;
pub fn single_source_all_shortest_paths<G, F, K, E>(
    graph: G,
    source: G::NodeId,
    mut edge_cost: F,
) -> Result<AllShortestPathsMap<G::NodeId>, E>
where
    G: IntoEdgesDirected + Visitable + NodeIndexable + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone + Ord,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: petgraph::algo::Measure + Copy + std::ops::Add<Output = K> + PartialEq + PartialOrd,
{
    // Compute shortest path distances using Dijkstra's algorithm
    let distance: DictMap<G::NodeId, K> = dijkstra(graph, source, None, &mut edge_cost, None)?;

    // Build predecessor map for all nodes
    let max_index = graph
        .node_identifiers()
        .map(|n| graph.to_index(n))
        .max()
        .unwrap_or(0);
    let mut pred = vec![Vec::new(); max_index + 1];

    for v in graph.node_identifiers() {
        if let Some(dist_v) = distance.get(&v) {
            let mut predecessors = Vec::new();
            for edge in graph.edges_directed(v, Incoming) {
                let u = edge.source();
                if let Some(dist_u) = distance.get(&u) {
                    let cost = edge_cost(edge)?;
                    if *dist_u + cost == *dist_v {
                        predecessors.push(u);
                    }
                }
            }
            if !predecessors.is_empty() {
                pred[graph.to_index(v)] = predecessors;
            }
        }
    }

    // Collect all shortest paths from source to a node
    fn collect_paths<G>(
        v: G::NodeId,
        pred: &Vec<Vec<G::NodeId>>,
        source: G::NodeId,
        current_path: &mut Vec<G::NodeId>,
        all_paths: &mut Vec<Vec<G::NodeId>>,
        graph: &G,
    ) where
        G: IntoEdgesDirected + Visitable + NodeIndexable + IntoNodeIdentifiers,
        G::NodeId: Eq + Hash + Clone + Ord,
    {
        if v == source {
            let mut path = current_path.clone();
            path.push(source);
            path.reverse();
            all_paths.push(path);
            return;
        }
        for &p in &pred[graph.to_index(v)] {
            if !current_path.contains(&p) {
                current_path.push(v);
                collect_paths(p, pred, source, current_path, all_paths, graph);
                current_path.pop();
            }
        }
    }

    // Compute all shortest paths for each reachable node
    let mut all_paths_map: DictMap<G::NodeId, Vec<Vec<G::NodeId>>> = DictMap::new();
    for node in graph.node_identifiers() {
        if distance.contains_key(&node) {
            let mut all_paths = Vec::new();
            let mut current_path = Vec::new();
            collect_paths(
                node,
                &pred,
                source,
                &mut current_path,
                &mut all_paths,
                &graph,
            );
            all_paths_map.insert(node, all_paths);
        }
    }

    Ok(all_paths_map)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictmap::DictMap;
    use crate::generators::grid_graph;
    use hashbrown::HashSet;
    use petgraph::prelude::*;
    use petgraph::Graph;
    use std::convert::Infallible;
    #[test]
    fn test_single_source_all_shortest_paths_cycle() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(c, d, 1.0);
        g.add_edge(d, a, 1.0);

        let paths = single_source_all_shortest_paths(&g, a, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();

        let mut expected = DictMap::new();
        expected.insert(a, vec![vec![a]]);
        expected.insert(b, vec![vec![a, b]]);
        expected.insert(c, vec![vec![a, b, c], vec![a, d, c]]);
        expected.insert(d, vec![vec![a, d]]);

        for paths_list in expected.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }
        let mut actual_paths = paths.clone();
        for paths_list in actual_paths.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }

        assert_eq!(actual_paths, expected);
    }

    #[test]
    fn test_single_source_all_shortest_paths_grid() {
        // Create a 4x4 grid graph with explicit type annotation
        let g: Graph<(), f64, petgraph::Undirected> = grid_graph(
            Some(4), // rows
            Some(4), // cols
            None,    // no specific weights
            || (),   // default node weight
            || 1.0,  // default edge weight
            false,   // unidirectional edges
        )
        .unwrap();

        let source = NodeIndex::new(1); // Node 1
        let paths = single_source_all_shortest_paths(
            &g,
            source,
            |e: petgraph::graph::EdgeReference<'_, f64>| {
                Ok::<_, std::convert::Infallible>(*e.weight())
            },
        )
        .unwrap();

        let target = NodeIndex::new(11); // Node 11
        let expected_paths = vec![
            vec![
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(3),
                NodeIndex::new(7),
                NodeIndex::new(11),
            ],
            vec![
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(6),
                NodeIndex::new(7),
                NodeIndex::new(11),
            ],
            vec![
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(6),
                NodeIndex::new(10),
                NodeIndex::new(11),
            ],
            vec![
                NodeIndex::new(1),
                NodeIndex::new(5),
                NodeIndex::new(6),
                NodeIndex::new(7),
                NodeIndex::new(11),
            ],
            vec![
                NodeIndex::new(1),
                NodeIndex::new(5),
                NodeIndex::new(6),
                NodeIndex::new(10),
                NodeIndex::new(11),
            ],
            vec![
                NodeIndex::new(1),
                NodeIndex::new(5),
                NodeIndex::new(9),
                NodeIndex::new(10),
                NodeIndex::new(11),
            ],
        ];

        let actual_paths = paths.get(&target).unwrap();
        let mut actual_paths_sorted = actual_paths.clone();
        actual_paths_sorted.sort_by(|p1, p2| p1.cmp(p2));
        let mut expected_paths_sorted = expected_paths.clone();
        expected_paths_sorted.sort_by(|p1, p2| p1.cmp(p2));

        assert_eq!(actual_paths_sorted, expected_paths_sorted);
    }

    #[test]
    fn test_single_source_all_shortest_paths_disconnected() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        let e = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(c, d, 1.0);
        g.add_edge(d, a, 1.0);

        let paths_from_a = single_source_all_shortest_paths(&g, a, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();
        let expected_a = vec![vec![a, b, c], vec![a, d, c]];
        let actual_paths_a = paths_from_a.get(&c).unwrap();
        let mut actual_paths_sorted_a = actual_paths_a.clone();
        actual_paths_sorted_a.sort_by(|p1, p2| p1.cmp(p2));
        let mut expected_paths_sorted_a = expected_a.clone();
        expected_paths_sorted_a.sort_by(|p1, p2| p1.cmp(p2));
        assert_eq!(actual_paths_sorted_a, expected_paths_sorted_a);

        let paths_from_e = single_source_all_shortest_paths(&g, e, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();
        assert_eq!(paths_from_e.len(), 1); // only e
        assert_eq!(paths_from_e.get(&e), Some(&vec![vec![e]]));
    }

    #[test]
    fn test_single_source_all_shortest_paths_invalid_weights() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(());
        let b = g.add_node(());
        g.add_edge(a, b, f64::NAN);

        let result_nan = single_source_all_shortest_paths(&g, a, |e| {
            let weight = *e.weight();
            if weight.is_nan() {
                Err("Weight is NaN".to_string())
            } else {
                Ok(weight)
            }
        });
        assert!(result_nan.is_err());

        g.update_edge(a, b, -1.0);
        let result_neg = single_source_all_shortest_paths(&g, a, |e| {
            let weight = *e.weight();
            if weight < 0.0 {
                Err("Weight is negative".to_string())
            } else {
                Ok(weight)
            }
        });
        assert!(result_neg.is_err());
    }

    #[test]
    fn test_single_source_all_shortest_paths_directed() {
        let mut g = DiGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(a, c, 1.0);
        g.add_edge(b, d, 1.0);
        g.add_edge(c, d, 1.0);

        let paths = single_source_all_shortest_paths(&g, a, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();

        let mut expected = DictMap::new();
        expected.insert(a, vec![vec![a]]);
        expected.insert(b, vec![vec![a, b]]);
        expected.insert(c, vec![vec![a, c]]);
        expected.insert(d, vec![vec![a, b, d], vec![a, c, d]]);

        for paths_list in expected.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }
        let mut actual_paths = paths.clone();
        for paths_list in actual_paths.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }

        assert_eq!(actual_paths, expected);
    }
    #[test]
    fn test_single_source_all_shortest_paths_zero_weight_no_cycle() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 0.0);

        let paths = single_source_all_shortest_paths(&g, a, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();

        let mut expected = DictMap::new();
        expected.insert(a, vec![vec![a]]);
        expected.insert(b, vec![vec![a, b]]);
        expected.insert(c, vec![vec![a, b, c]]);

        for paths_list in expected.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }
        let mut actual_paths = paths.clone();
        for paths_list in actual_paths.values_mut() {
            paths_list.sort_by(|p1, p2| p1.cmp(p2));
        }

        assert_eq!(actual_paths, expected);
    }

    #[test]
    fn test_single_source_all_shortest_paths_zero_weight_with_cycle() {
        let mut g = Graph::<(), f64, Undirected>::new_undirected();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, 0.0);
        g.add_edge(b, c, 0.0);
        g.add_edge(c, a, 0.0);

        let mut edge_cost =
            |e: petgraph::graph::EdgeReference<'_, f64>| -> Result<f64, Infallible> {
                Ok(*e.weight())
            };
        let result = single_source_all_shortest_paths(&g, a, &mut edge_cost).unwrap();

        for (_node, paths) in result.iter() {
            for path in paths {
                // Check no repeated nodes
                let mut seen = HashSet::new();
                for &n in path {
                    assert!(
                        seen.insert(n),
                        "Path {:?} contains repeated node {:?}",
                        path,
                        n
                    );
                }
            }
        }
    }
    #[test]
    fn test_single_source_all_shortest_paths_zero_weight() {
        use crate::dictmap::DictMap;
        use petgraph::graph::{NodeIndex, UnGraph};
        use std::convert::Infallible;

        let mut graph = UnGraph::<(), f64>::new_undirected();
        let a = graph.add_node(()); // Node 0
        let b = graph.add_node(()); // Node 1
        let c = graph.add_node(()); // Node 2
        let d = graph.add_node(()); // Node 3

        // Add edges with weights
        graph.add_edge(a, b, 0.0); // 0 -- 1 with weight 0
        graph.add_edge(b, c, 0.0); // 1 -- 2 with weight 0
        graph.add_edge(c, a, 0.0); // 2 -- 0 with weight 0
        graph.add_edge(c, d, 1.0); // 2 -- 3 with weight 1

        // Define the edge cost function
        let edge_cost =
            |e: petgraph::graph::EdgeReference<f64>| -> Result<f64, Infallible> { Ok(*e.weight()) };

        // Compute all shortest paths from source node 0 (a)
        let paths = single_source_all_shortest_paths(&graph, a, edge_cost).unwrap();

        // Define expected shortest paths
        let mut expected: DictMap<NodeIndex, Vec<Vec<NodeIndex>>> = DictMap::new();
        expected.insert(a, vec![vec![a]]); // To 0: [[0]]
        expected.insert(b, vec![vec![a, b], vec![a, c, b]]); // To 1: [[0, 1], [0, 2, 1]]
        expected.insert(c, vec![vec![a, c], vec![a, b, c]]); // To 2: [[0, 2], [0, 1, 2]]
        expected.insert(d, vec![vec![a, c, d], vec![a, b, c, d]]); // To 3: [[0, 2, 3], [0, 1, 2, 3]]

        // Verify all paths match the expected output
        for (node, expected_paths) in expected.iter() {
            let computed_paths = paths.get(node).unwrap();
            let mut computed_paths_sorted = computed_paths.clone();
            computed_paths_sorted.sort(); // Sort for comparison
            let mut expected_paths_sorted = expected_paths.clone();
            expected_paths_sorted.sort(); // Sort for comparison
            assert_eq!(computed_paths_sorted, expected_paths_sorted);
        }
    }
}
