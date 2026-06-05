#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::petgraph::graph::UnGraph;
use rustworkx_core::planar::is_planar;

#[derive(Debug, Arbitrary)]
struct FuzzGraph {
    edges: Vec<(usize, usize)>,
    node_count: usize,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(fuzz_input) = FuzzGraph::arbitrary(&mut Unstructured::new(data)) {
        fuzz_check_is_planar(&fuzz_input);
    }
});

fn fuzz_check_is_planar(input: &FuzzGraph) {
    if input.node_count == 0 || input.edges.is_empty() || input.node_count > 1000 {
        return;
    }

    let mut graph = UnGraph::<(), ()>::default();
    let mut nodes = Vec::with_capacity(input.node_count);

    for _ in 0..input.node_count {
        nodes.push(graph.add_node(()));
    }

    for &(u, v) in &input.edges {
        if u < input.node_count && v < input.node_count {
            graph.add_edge(nodes[u], nodes[v], ());
        }
    }

    let _ = is_planar(&graph);
}
