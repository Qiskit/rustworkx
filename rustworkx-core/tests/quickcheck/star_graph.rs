#[cfg(test)]
mod tests {
    use petgraph::graph::DiGraph;
    use petgraph::visit::EdgeRef;
    use quickcheck::{quickcheck, TestResult};

    use rustworkx_core::generators::star_graph;

    #[test]
    fn prop_star_graph_structure() {
        fn prop(n: usize, bidirectional: bool, inward: bool) -> TestResult {
            let n = n % 50; // prevent overly large graphs

            let g: DiGraph<(), ()> =
                match star_graph(Some(n), None, || (), || (), inward, bidirectional) {
                    Ok(graph) => graph,
                    Err(_) => return TestResult::failed(),
                };

            // Number of nodes should be exactly n
            if g.node_count() != n {
                return TestResult::failed();
            }

            // Star graph has one central node (0), all others connected to/from it
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

            // All edges must involve the center node (index 0)
            for edge in g.edge_references() {
                let src = edge.source().index();
                let dst = edge.target().index();
                if src != 0 && dst != 0 {
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        quickcheck(prop as fn(usize, bool, bool) -> TestResult);
    }
}
