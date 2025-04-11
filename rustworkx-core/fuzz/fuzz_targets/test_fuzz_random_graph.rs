#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::generators::gnm_random_graph;
use rustworkx_core::petgraph::graph::DiGraph;

#[derive(Debug, Arbitrary)]
struct GnmInput {
    n: usize,
    m: usize,
    seed: Option<u64>,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = GnmInput::arbitrary(&mut Unstructured::new(data)) {
        fuzz_gnm_random_graph(input);
    }
});

fn fuzz_gnm_random_graph(input: GnmInput) {
    if input.n > 512 || input.m > 512 * 512 {
        return;
    }

    let max_m = input.n.saturating_mul(input.n.saturating_sub(1));
    let capped_m = input.m.min(max_m);

    if let Ok(graph) = gnm_random_graph::<DiGraph<(), ()>, _, _, _, ()>(
        input.n,
        capped_m,
        input.seed,
        || (),
        || (),
    ) {
        assert_eq!(graph.node_count(), input.n);
        assert!(
            graph.edge_count() <= max_m,
            "edge_count {} > max_m {}",
            graph.edge_count(),
            max_m
        );
    }
}
