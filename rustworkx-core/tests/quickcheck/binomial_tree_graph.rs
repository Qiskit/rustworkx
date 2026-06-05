use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::binomial_tree_graph;

#[test]
fn prop_binomial_tree_structure() {
    fn prop(order: u32) -> TestResult {
        let order = order % 10; // upto 2^10 nodes due to memory constraints.
        if order >= 60 {
            return TestResult::discard();
        }

        let g = match binomial_tree_graph::<UnGraph<(), ()>, (), _, _, ()>(
            order,
            None,
            || (),
            || (),
            true,
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in graph generation"),
        };

        let expected_nodes = 2_usize.pow(order);
        let expected_edges = 2 * (expected_nodes - 1);

        if g.node_count() != expected_nodes || g.edge_count() != expected_edges {
            return TestResult::failed();
        }

        // Check that for every edge, a reverse edge exists
        for edge in g.edge_references() {
            let u = edge.source();
            let v = edge.target();
            let reverse_exists = g.find_edge(v, u).is_some();
            if !reverse_exists {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u32) -> TestResult);
}
