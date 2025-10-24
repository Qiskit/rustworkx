use petgraph::graph::UnGraph;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::lollipop_graph;

#[test]
fn prop_lollipop_graph_structure() {
    fn prop(mesh_size: usize, path_size: usize) -> TestResult {
        // Constrain input space for sanity
        let mesh_size = mesh_size % 64;
        let path_size = path_size % 64;

        if mesh_size + path_size == 0 {
            return TestResult::discard();
        }
        if mesh_size <= 1 {
            return TestResult::discard();
        }

        let graph = match lollipop_graph::<UnGraph<(), ()>, (), _, _, ()>(
            Some(mesh_size),
            Some(path_size),
            None,
            None,
            || (),
            || (),
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Unexpected error in graph generation"),
        };

        let expected_nodes = mesh_size + path_size;
        let mesh_edges = mesh_size * (mesh_size - 1) / 2;
        let path_edges = path_size.saturating_sub(1);
        let connector = if path_size > 0 { 1 } else { 0 };
        let expected_edges = mesh_edges + path_edges + connector;

        let node_match = graph.node_count() == expected_nodes;
        let edge_match = graph.edge_count() == expected_edges;

        TestResult::from_bool(node_match && edge_match)
    }

    quickcheck(prop as fn(usize, usize) -> TestResult);
}
