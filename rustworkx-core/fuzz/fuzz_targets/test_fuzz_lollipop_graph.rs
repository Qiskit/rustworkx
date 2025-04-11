#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::generators::lollipop_graph;
use rustworkx_core::petgraph::graph::UnGraph;

#[derive(Debug, Arbitrary)]
struct LollipopInput {
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    mesh_weights: Option<Vec<u8>>,
    path_weights: Option<Vec<u8>>,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = LollipopInput::arbitrary(&mut Unstructured::new(data)) {
        fuzz_lollipop_graph(input);
    }
});

fn fuzz_lollipop_graph(input: LollipopInput) {
    if let Some(n) = input.num_mesh_nodes {
        if n > 512 {
            return;
        }
    }
    if let Some(n) = input.num_path_nodes {
        if n > 512 {
            return;
        }
    }
    if let Some(ref w) = input.mesh_weights {
        if w.len() > 512 {
            return;
        }
    }
    if let Some(ref w) = input.path_weights {
        if w.len() > 512 {
            return;
        }
    }

    let _ = lollipop_graph::<UnGraph<u8, ()>, u8, _, _, ()>(
        input.num_mesh_nodes,
        input.num_path_nodes,
        input.mesh_weights.clone(),
        input.path_weights.clone(),
        || 0,
        || (),
    );
}
