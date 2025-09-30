use petgraph::graph::{DiGraph, UnGraph};
use petgraph::visit::EdgeRef;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::complete_graph;
use std::collections::HashSet;

#[test]
fn prop_complete_graph_structure_directed() {
    fn prop(n: usize) -> TestResult {
        let n = n % 100; // limit for runtime issues
        if n == 0 {
            return TestResult::discard();
        }

        let g = match complete_graph::<DiGraph<(), ()>, (), _, _, ()>(Some(n), None, || (), || ()) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in directed complete graph"),
        };

        if g.node_count() != n || g.edge_count() != n * (n - 1) {
            return TestResult::failed();
        }

        // Ensure that for every pair (i, j) with i != j, an edge exists exists
        let expected_edges: HashSet<_> = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .collect();

        let actual_edges: HashSet<_> = g
            .edge_references()
            .map(|e| (e.source().index(), e.target().index()))
            .collect();

        TestResult::from_bool(expected_edges == actual_edges)
    }

    quickcheck(prop as fn(usize) -> TestResult);
}

#[test]
fn prop_complete_graph_structure_undirected() {
    fn prop(n: usize) -> TestResult {
        let n = n % 100;
        if n == 0 {
            return TestResult::discard();
        }

        let g = match complete_graph::<UnGraph<(), ()>, (), _, _, ()>(Some(n), None, || (), || ()) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in undirected complete graph"),
        };

        if g.node_count() != n || g.edge_count() != n * (n - 1) / 2 {
            return TestResult::failed();
        }

        // in undirected graphs, (i, j) and (j, i) are the same edge, so we only include one of them.
        let expected_edges: HashSet<_> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();

        let actual_edges: HashSet<_> = g
            .edge_references()
            .map(|e| {
                let u = e.source().index();
                let v = e.target().index();
                (u.min(v), u.max(v))
            })
            .collect();

        TestResult::from_bool(expected_edges == actual_edges)
    }

    quickcheck(prop as fn(usize) -> TestResult);
}
