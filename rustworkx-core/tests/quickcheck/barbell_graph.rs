use petgraph::graph::UnGraph;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::barbell_graph;

#[test]
fn prop_barbell_graph_structure() {
    fn prop(mesh_size: usize, path_size: usize) -> TestResult {
        let mesh_size = mesh_size % 32 + 2;
        let path_size = path_size % 32;

        let graph = match barbell_graph::<UnGraph<(), ()>, (), _, _, ()>(
            Some(mesh_size),
            Some(path_size),
            None,
            None,
            || (),
            || (),
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Failed to generate barbell graph"),
        };

        // Expected node count
        let expected_nodes = 2 * mesh_size + path_size;
        if graph.node_count() != expected_nodes {
            return TestResult::failed();
        }

        // Expected edge count
        let mesh_edges = mesh_size * (mesh_size - 1) / 2;
        let path_edges = if path_size > 0 { path_size - 1 } else { 0 };
        let connectors = if path_size > 0 { 2 } else { 1 };
        let expected_edges = 2 * mesh_edges + path_edges + connectors;

        if graph.edge_count() != expected_edges {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(usize, usize) -> TestResult);
}
