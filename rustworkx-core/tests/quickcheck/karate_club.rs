#[cfg(test)]
mod tests {
    use petgraph::graph::UnGraph;
    use petgraph::visit::EdgeRef;
    use quickcheck::{quickcheck, TestResult};

    use rustworkx_core::generators::karate_club_graph;

    #[test]
    fn prop_karate_club_basic_structure() {
        fn prop(x: u8, y: u16) -> TestResult {
            let g: UnGraph<u8, u16> = karate_club_graph(
                |is_hi| if is_hi { x } else { x.wrapping_add(1) },
                |strength| y.wrapping_add(strength as u16),
            );

            // Always 34 nodes
            if g.node_count() != 34 {
                return TestResult::failed();
            }

            // Always 78 edges
            if g.edge_count() != 78 {
                return TestResult::failed();
            }

            // No self-loops
            if g.edge_references().any(|e| e.source() == e.target()) {
                return TestResult::failed();
            }

            // All nodes referenced are valid
            let max_node = g.node_count();
            for edge in g.edge_references() {
                if edge.source().index() >= max_node || edge.target().index() >= max_node {
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        quickcheck(prop as fn(u8, u16) -> TestResult);
    }
}
