#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::err::ContractError;
use rustworkx_core::graph_ext::*;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::*;

#[derive(Debug, Arbitrary)]
struct ContractFuzzInput {
    edges: Vec<(usize, usize, usize)>,
    node_count: usize,
    contract_indices: Vec<usize>,
    replacement_weight: char,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = ContractFuzzInput::arbitrary(&mut Unstructured::new(data)) {
        fuzz_contract_nodes(input);
    }
});

fn fuzz_contract_nodes(input: ContractFuzzInput) {
    if input.node_count == 0 || input.node_count > 500 || input.edges.len() > 5000 {
        return;
    }

    let mut graph: StableDiGraph<char, usize> = StableDiGraph::new();
    let mut nodes = Vec::with_capacity(input.node_count);
    for i in 0..input.node_count {
        let label = (b'a' + ((i % 26) as u8)) as char;
        nodes.push(graph.add_node(label));
    }

    for (u, v, w) in input.edges {
        if u < input.node_count && v < input.node_count && w > 0 {
            graph.add_edge(nodes[u], nodes[v], w);
        }
    }

    let to_contract: Vec<NodeIndex> = input
        .contract_indices
        .into_iter()
        .filter_map(|i| nodes.get(i).copied())
        .collect();

    if to_contract.len() < 2 {
        return;
    }

    let mut graph_no_check = graph.clone();

    // Run contraction without cycle check (should never fail)
    let _ = graph_no_check.contract_nodes(to_contract.clone(), input.replacement_weight, false);

    // Run contraction with cycle check, match on the result
    #[allow(unreachable_patterns)]
    match graph.contract_nodes(to_contract.clone(), input.replacement_weight, true) {
        Ok(_) => {
            // Idempotency: running again should not panic
            let _ = graph.contract_nodes(to_contract, input.replacement_weight, true);
        }
        Err(ContractError::DAGWouldCycle) => {
            // Expected error — no-op
        }
        Err(err) => {
            panic!("Unexpected error during node contraction: {:?}", err);
        }
    }
}
