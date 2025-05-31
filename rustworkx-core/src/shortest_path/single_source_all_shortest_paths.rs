use crate::dictmap::{DictMap, InitWithHasher};
use crate::shortest_path::dijkstra;
use petgraph::visit::{EdgeRef, IntoEdgesDirected, IntoNodeIdentifiers, NodeIndexable, Visitable};
use petgraph::Direction::Incoming;
use std::hash::Hash;

pub fn single_source_all_shortest_paths<G, F, K, E>(
    graph: G,
    source: G::NodeId,
    mut edge_cost: F,
) -> Result<DictMap<G::NodeId, Vec<Vec<G::NodeId>>>, E>
where
    G: IntoEdgesDirected + Visitable + NodeIndexable + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: petgraph::algo::Measure + Copy + std::ops::Add<Output = K> + PartialEq,
{
    // Compute shortest path distances using Dijkstra's algorithm
    let distance: DictMap<G::NodeId, K> = dijkstra(graph, source, None, &mut edge_cost, None)?;

    // Build predecessor map for all nodes
    let mut pred: DictMap<G::NodeId, Vec<G::NodeId>> = DictMap::new();

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
                pred.insert(v, predecessors);
            }
        }
    }

    // Memoized recursive function to compute all shortest paths
    let mut memo: DictMap<G::NodeId, Vec<Vec<G::NodeId>>> = DictMap::new();
    fn all_paths<G: NodeIndexable>(
        n: G::NodeId,
        source: G::NodeId,
        pred: &DictMap<G::NodeId, Vec<G::NodeId>>,
        memo: &mut DictMap<G::NodeId, Vec<Vec<G::NodeId>>>,
    ) -> Vec<Vec<G::NodeId>>
    where
        G::NodeId: Eq + Hash + Clone,
    {
        if let Some(paths) = memo.get(&n) {
            return paths.clone();
        }
        if n == source {
            let paths = vec![vec![source]];
            memo.insert(n, paths.clone());
            return paths;
        }
        let mut paths = Vec::new();
        if let Some(pred_n) = pred.get(&n) {
            for p in pred_n {
                let paths_p = all_paths::<G>(*p, source, pred, memo);
                for mut path in paths_p {
                    path.push(n);
                    paths.push(path);
                }
            }
        }
        memo.insert(n, paths.clone());
        paths
    }

    // Compute all shortest paths for each reachable node
    let mut all_paths_map: DictMap<G::NodeId, Vec<Vec<G::NodeId>>> = DictMap::new();
    for n in graph.node_identifiers() {
        if distance.contains_key(&n) {
            let paths = all_paths::<G>(n, source, &pred, &mut memo);
            all_paths_map.insert(n, paths);
        }
    }

    Ok(all_paths_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictmap::DictMap;
    use petgraph::prelude::*;
    use petgraph::Graph;

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
        let mut g = Graph::new_undirected();
        let mut nodes = Vec::new();
        for _ in 0..16 {
            nodes.push(g.add_node(()));
        }
        // Add horizontal edges
        for r in 0..4 {
            for c in 0..3 {
                let i = r * 4 + c;
                let j = r * 4 + c + 1;
                g.add_edge(nodes[i], nodes[j], 1.0);
            }
        }

        // Add vertical edges
        for c in 0..4 {
            for r in 0..3 {
                let i = r * 4 + c;
                let j = (r + 1) * 4 + c;
                g.add_edge(nodes[i], nodes[j], 1.0);
            }
        }

        let source = nodes[1]; // Node 1
        let paths = single_source_all_shortest_paths(&g, source, |e| {
            Ok::<_, std::convert::Infallible>(*e.weight())
        })
        .unwrap();

        let target = nodes[11]; // Node 11
        let expected_paths = vec![
            vec![nodes[1], nodes[2], nodes[3], nodes[7], nodes[11]],
            vec![nodes[1], nodes[2], nodes[6], nodes[7], nodes[11]],
            vec![nodes[1], nodes[2], nodes[6], nodes[10], nodes[11]],
            vec![nodes[1], nodes[5], nodes[6], nodes[7], nodes[11]],
            vec![nodes[1], nodes[5], nodes[6], nodes[10], nodes[11]],
            vec![nodes[1], nodes[5], nodes[9], nodes[10], nodes[11]],
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
}
