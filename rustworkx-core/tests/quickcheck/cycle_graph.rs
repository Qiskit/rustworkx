use petgraph::graph::{DiGraph, UnGraph};
use petgraph::visit::EdgeRef;
use quickcheck::{quickcheck, TestResult};
use rustworkx_core::generators::cycle_graph;
use std::collections::HashSet;

#[test]
fn prop_cycle_graph_directed() {
    fn prop(n: usize) -> TestResult {
        let n = (n % 100).max(2);
        let g = match cycle_graph::<DiGraph<(), ()>, (), _, _, ()>(
            Some(n),
            None,
            || (),
            || (),
            false,
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in directed cycle_graph"),
        };

        if g.node_count() != n || g.edge_count() != n {
            return TestResult::failed();
        }

        // Expected edges in circular fashion 0→1, 1→2, ... n-1→0
        let expected_edges: HashSet<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        let actual_edges: HashSet<_> = g
            .edge_references()
            .map(|e| (e.source().index(), e.target().index()))
            .collect();

        TestResult::from_bool(expected_edges == actual_edges)
    }

    quickcheck(prop as fn(usize) -> TestResult);
}

#[test]
fn prop_cycle_graph_bidirectional() {
    fn prop(n: usize) -> TestResult {
        let n = (n % 100).max(2);
        let g =
            match cycle_graph::<DiGraph<(), ()>, (), _, _, ()>(Some(n), None, || (), || (), true) {
                Ok(g) => g,
                Err(_) => {
                    return TestResult::error("Unexpected error in bidirectional cycle_graph")
                }
            };

        if g.node_count() != n || g.edge_count() != 2 * n {
            return TestResult::failed();
        }

        // For each directed edge, its reverse must exist
        for edge in g.edge_references() {
            let u = edge.source();
            let v = edge.target();
            if g.find_edge(v, u).is_none() {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(usize) -> TestResult);
}

#[test]
fn prop_cycle_graph_undirected() {
    fn prop(n: usize) -> TestResult {
        let n = (n % 100).max(2);
        let g = match cycle_graph::<UnGraph<(), ()>, (), _, _, ()>(
            Some(n),
            None,
            || (),
            || (),
            false,
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in undirected cycle_graph"),
        };

        if g.node_count() != n || g.edge_count() != n {
            return TestResult::failed();
        }

        let expected_edges: HashSet<_> = (0..n)
            .map(|i| {
                let u = i;
                let v = (i + 1) % n;
                (u.min(v), u.max(v))
            })
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
