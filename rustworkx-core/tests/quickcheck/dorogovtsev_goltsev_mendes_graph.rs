use petgraph::graph::Graph;
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::dorogovtsev_goltsev_mendes_graph;

#[test]
fn prop_dgm_graph_structure() {
    fn prop(n: u8) -> TestResult {
        let n = (n % 7) as usize;
        let g: Graph<(), ()> = match dorogovtsev_goltsev_mendes_graph(n, || (), || ()) {
            Ok(graph) => graph,
            Err(_) => return TestResult::error("Failed to generate DGM graph"),
        };

        let expected_edges = 3_usize.pow(n as u32);
        let expected_nodes = (expected_edges + 3) / 2;

        if g.node_count() != expected_nodes || g.edge_count() != expected_edges {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8) -> TestResult);
}
