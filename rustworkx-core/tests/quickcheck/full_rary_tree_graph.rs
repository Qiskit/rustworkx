use petgraph::graph::UnGraph;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::full_rary_tree_graph;

#[test]
fn prop_full_rary_tree_structure() {
    fn prop(branching: usize, num_nodes: usize) -> TestResult {
        let branching = branching % 10 + 1;
        let num_nodes = num_nodes % 100;

        if num_nodes == 0 {
            return TestResult::discard();
        }

        let graph = match full_rary_tree_graph::<UnGraph<(), ()>, (), _, _, ()>(
            branching,
            num_nodes,
            None,
            || (),
            || (),
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in tree generation"),
        };

        // Property 1: Node count must match
        if graph.node_count() != num_nodes {
            return TestResult::failed();
        }

        // Property 2: Edge count must be exactly n - 1
        if graph.edge_count() != num_nodes - 1 {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(usize, usize) -> TestResult);
}
