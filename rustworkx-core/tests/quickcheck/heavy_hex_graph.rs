use petgraph::graph::UnGraph;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::heavy_hex_graph;

#[test]
fn prop_heavy_hex_structure() {
    fn prop(d: usize) -> TestResult {
        let d = (d % 31) | 1;
        if d == 0 {
            return TestResult::discard();
        }

        let g = match heavy_hex_graph::<UnGraph<(), ()>, (), _, _, ()>(d, || (), || (), false) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in graph generation"),
        };

        let expected_nodes = (5 * d * d - 2 * d - 1) / 2;
        let expected_edges = 2 * d * (d - 1) + (d + 1) * (d - 1);

        let node_ok = g.node_count() == expected_nodes;
        let edge_ok = g.edge_count() == expected_edges;

        TestResult::from_bool(node_ok && edge_ok)
    }

    quickcheck(prop as fn(usize) -> TestResult);
}
