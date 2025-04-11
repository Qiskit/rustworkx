#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::generators::heavy_hex_graph;
use rustworkx_core::petgraph::graph::UnGraph;

#[derive(Debug, Arbitrary)]
struct HeavyHexInput {
    d: usize,
    bidirectional: bool,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = HeavyHexInput::arbitrary(&mut Unstructured::new(data)) {
        fuzz_heavy_hex_graph(input);
    }
});

fn fuzz_heavy_hex_graph(input: HeavyHexInput) {
    let d = (input.d % 21) | 1;
    let _ = heavy_hex_graph::<UnGraph<(), ()>, (), _, _, ()>(d, || (), || (), input.bidirectional);
}
