use petgraph::graph::DiGraph;
use petgraph::visit::{Bfs, EdgeRef, NodeIndexable};
use quickcheck::{quickcheck, TestResult};
use rustworkx_core::generators::grid_graph;

#[test]
fn prop_grid_graph_validity() {
    fn prop(rows: usize, cols: usize, bidirectional_flag: bool) -> TestResult {
        let rows = rows % 32 + 1; // Ensure >= 1
        let cols = cols % 32 + 1;

        let graph = match grid_graph::<DiGraph<(), ()>, (), _, _, ()>(
            Some(rows),
            Some(cols),
            None,
            || (),
            || (),
            bidirectional_flag,
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::error("Graph generation failed"),
        };

        let expected_nodes = rows * cols;
        let base_edges = (rows - 1) * cols + (cols - 1) * rows;
        let expected_edges = if bidirectional_flag {
            base_edges * 2
        } else {
            base_edges
        };

        // Property 1: Node count
        if graph.node_count() != expected_nodes {
            return TestResult::failed();
        }

        // Property 2: Edge count
        if graph.edge_count() != expected_edges {
            return TestResult::failed();
        }

        // Property3: All edge indices are in bounds
        let max_index = graph.node_bound();
        for edge in graph.edge_references() {
            let src = edge.source().index();
            let tgt = edge.target().index();
            if src >= max_index || tgt >= max_index {
                return TestResult::failed();
            }
        }

        // Property 4: Weak connectivity (if bidirectional, should be connected)
        if bidirectional_flag {
            let mut bfs = Bfs::new(&graph, graph.from_index(0));
            let mut visited = 0;
            while bfs.next(&graph).is_some() {
                visited += 1;
            }
            if visited != expected_nodes {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(usize, usize, bool) -> TestResult);
}
