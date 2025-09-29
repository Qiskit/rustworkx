#[cfg(test)]
mod tests {
    use petgraph::graph::UnGraph;
    use petgraph::visit::EdgeRef;
    use quickcheck::{quickcheck, TestResult};

    use rustworkx_core::generators::petersen_graph;

    #[test]
    fn prop_petersen_graph_structure() {
        fn prop(n_in: u8, k_in: u8) -> TestResult {
            let n = (n_in as usize % 97).max(3); // n >= 3
            let k = ((k_in as usize) % n).max(1); // k >= 1

            // Discard invalid input: k must be < n/2
            if k >= n / 2 {
                return TestResult::discard();
            }

            let g: UnGraph<(), ()> = match petersen_graph(n, k, || (), || ()) {
                Ok(graph) => graph,
                Err(_) => return TestResult::failed(),
            };

            // node_count should be exactly 2n
            if g.node_count() != 2 * n {
                return TestResult::failed();
            }

            // edge_count should be exactly 3n
            if g.edge_count() != 3 * n {
                return TestResult::failed();
            }

            // No self-loops
            if g.edge_references().any(|e| e.source() == e.target()) {
                return TestResult::failed();
            }

            // All node indices must be within bounds
            let max_node = g.node_count();
            for edge in g.edge_references() {
                if edge.source().index() >= max_node || edge.target().index() >= max_node {
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        quickcheck(prop as fn(u8, u8) -> TestResult);
    }
}
