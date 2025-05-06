#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::generators::gnm_random_graph;
use rustworkx_core::petgraph::graph::DiGraph;
use rustworkx_core::petgraph::visit::EdgeRef;

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

    if let Ok(graph1) = gnm_random_graph::<DiGraph<(), ()>, _, _, _, ()>(
        input.n,
        capped_m,
        input.seed,
        || (),
        || (),
    ) {
        assert_eq!(graph1.node_count(), input.n);
        assert!(
            graph1.edge_count() <= max_m,
            "edge_count {} > max_m {}",
            graph1.edge_count(),
            max_m
        );

        if input.seed.is_some() {
            if let Ok(graph2) = gnm_random_graph::<DiGraph<(), ()>, _, _, _, ()>(
                input.n,
                capped_m,
                input.seed,
                || (),
                || (),
            ) {
                assert_eq!(graph1.node_count(), graph2.node_count());
                assert_eq!(graph1.edge_count(), graph2.edge_count());

                let mut edges1: Vec<_> = graph1
                    .edge_references()
                    .map(|e| (e.source().index(), e.target().index()))
                    .collect();
                let mut edges2: Vec<_> = graph2
                    .edge_references()
                    .map(|e| (e.source().index(), e.target().index()))
                    .collect();

                edges1.sort_unstable();
                edges2.sort_unstable();
                assert_eq!(edges1, edges2, "Graphs differ even with the same seed");
            }
        }
    }
}
