#[cfg(test)]
mod tests {
    use petgraph::graph::DiGraph;
    use petgraph::visit::EdgeRef;
    use quickcheck::{TestResult, quickcheck};

    use rustworkx_core::generators::path_graph;

    #[test]
    fn prop_path_graph_structure() {
        fn prop(n: usize, bidirectional: bool) -> TestResult {
            let n = n % 100; // Prevent large test cases

            let g: DiGraph<(), ()> = match path_graph(Some(n), None, || (), || (), bidirectional) {
                Ok(graph) => graph,
                Err(_) => return TestResult::failed(),
            };

            // node_count
            if g.node_count() != n {
                return TestResult::failed();
            }

            // edge_count
            let expected_edge_count = if n < 2 {
                0
            } else if bidirectional {
                2 * (n - 1)
            } else {
                n - 1
            };
            if g.edge_count() != expected_edge_count {
                return TestResult::failed();
            }

            // No self-loops
            if g.edge_references().any(|e| e.source() == e.target()) {
                return TestResult::failed();
            }

            // All edges connect to the next nodes (i to i+1)
            let mut expected_edges = vec![];
            for i in 0..n.saturating_sub(1) {
                expected_edges.push((i, i + 1));
                if bidirectional {
                    expected_edges.push((i + 1, i));
                }
            }

            let mut actual_edges: Vec<(usize, usize)> = g
                .edge_references()
                .map(|e| (e.source().index(), e.target().index()))
                .collect();

            expected_edges.sort_unstable();
            actual_edges.sort_unstable();

            if actual_edges != expected_edges {
                return TestResult::failed();
            }

            TestResult::passed()
        }

        quickcheck(prop as fn(usize, bool) -> TestResult);
    }
}
