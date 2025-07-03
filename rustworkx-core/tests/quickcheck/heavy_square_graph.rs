use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use quickcheck::{quickcheck, TestResult};
use rustworkx_core::generators::heavy_square_graph;
use std::collections::HashSet;

#[test]
fn prop_heavy_square_graph_structure() {
    fn prop(d: u8) -> TestResult {
        let d = ((d % 11) | 1) as usize; // ensure d is odd
        if d == 0 {
            return TestResult::discard();
        }

        let g: Graph<(), ()> = match heavy_square_graph(d, || (), || (), false) {
            Ok(graph) => graph,
            Err(_) => return TestResult::error("Failed to generate heavy square graph"),
        };

        let expected_nodes = 3 * d * d - 2 * d;
        let expected_edges = 4 * d * (d - 1);

        if g.node_count() != expected_nodes || g.edge_count() != expected_edges {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8) -> TestResult);
}

#[test]
fn prop_heavy_square_bidirectionality() {
    fn prop(d: u8) -> TestResult {
        let d = ((d % 11) | 1) as usize;
        if d == 0 {
            return TestResult::discard();
        }

        let g: Graph<(), ()> = match heavy_square_graph(d, || (), || (), true) {
            Ok(graph) => graph,
            Err(_) => return TestResult::error("Bidirectional graph generation failed"),
        };

        let mut seen = HashSet::new();
        for edge in g.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            if u == v {
                continue;
            }
            seen.insert((u.min(v), u.max(v)));
        }

        // Each edge should appear twice in bidirectional graph
        if g.edge_count() != 2 * seen.len() {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8) -> TestResult);
}

#[test]
fn prop_heavy_square_even_distance_should_fail() {
    fn prop(d: u8) -> bool {
        let d = (d & !1) as usize; // make d even
        if d == 0 {
            return true;
        }
        heavy_square_graph::<Graph<(), ()>, (), _, _, ()>(d, || (), || (), false).is_err()
    }

    quickcheck(prop as fn(u8) -> bool);
}
