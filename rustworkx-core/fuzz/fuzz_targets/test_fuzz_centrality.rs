#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::centrality::closeness_centrality;
use rustworkx_core::petgraph::graph::UnGraph;

#[derive(Debug, Arbitrary)]
struct CentralityFuzzInput {
    edges: Vec<(usize, usize)>,
    node_count: usize,
}

macro_rules! assert_almost_equal {
    ($x:expr, $y:expr, $d:expr, $i:expr, $wf:expr) => {
        if ($x - $y).abs() >= $d {
            panic!(
                "{} != {} within delta of {} at node {} with wf_improved = {}",
                $x, $y, $d, $i, $wf
            );
        }
    };
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = CentralityFuzzInput::arbitrary(&mut Unstructured::new(data)) {
        fuzz_centrality(input);
    }
});

fn fuzz_centrality(input: CentralityFuzzInput) {
    if input.node_count == 0 || input.node_count > 1000 {
        return;
    }

    let mut graph = UnGraph::<(), ()>::default();
    let nodes: Vec<_> = (0..input.node_count).map(|_| graph.add_node(())).collect();

    for (u, v) in input.edges {
        if u < input.node_count && v < input.node_count {
            graph.add_edge(nodes[u], nodes[v], ());
        }
    }

    let epsilon = 1e-4;

    for &wf_improved in &[true, false] {
        let seq_output = closeness_centrality(&graph, wf_improved, 200);
        let par_output = closeness_centrality(&graph, wf_improved, 1);

        let c_seq: Vec<f64> = seq_output.iter().map(|x| x.unwrap()).collect();
        let c_par: Vec<f64> = par_output.iter().map(|x| x.unwrap()).collect();

        assert_eq!(c_seq.len(), c_par.len(), "Centrality result size mismatch");

        for (i, (a, b)) in c_seq.iter().zip(c_par.iter()).enumerate() {
            assert_almost_equal!(a, b, epsilon, i, wf_improved);
        }
    }
}
