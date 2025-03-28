#![no_main]


use libfuzzer_sys::fuzz_target;
use arbitrary::{Arbitrary, Unstructured};
use std::collections::HashSet;
use rustworkx_core::petgraph::graph::Graph;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::visit::Visitable;
use rustworkx_core::petgraph::Undirected;

use rustworkx_core::connectivity;

// Struct for fuzz input
#[derive(Debug, Arbitrary)]
struct FuzzGraph {
    edges: Vec<(usize, usize)>, // Randomly generated edges
    node_count: usize,          // Total nodes in the graph
}

// Fuzzing function
fuzz_target!(|data: &[u8]| {
    if let Ok(fuzz_input) = FuzzGraph::arbitrary(&mut Unstructured::new(data)) {
        fuzz_test_is_connected(fuzz_input);
    }
});

fn fuzz_test_is_connected(input: FuzzGraph) {
    if input.node_count == 0 {
        return; // Ignore empty graphs
    }

    let mut graph = Graph::<(), (), Undirected>::new_undirected();

    // Add nodes
    let mut nodes = Vec::new();
    for _ in 0..input.node_count {
        nodes.push(graph.add_node(()));
    }

    // Add edges
    for &(src, tgt) in &input.edges {
        if src < input.node_count && tgt < input.node_count {
            graph.add_edge(nodes[src], nodes[tgt], ());
        }
    }

    // Pick a random node
    let start_node = nodes[0];

    println!("{:?}", graph);
    // BFS traversal
    let mut visit_map = graph.visit_map();
    let component: HashSet<usize> = connectivity::bfs_undirected(&graph, start_node, &mut visit_map)
        .into_iter()
        .map(|x| x.index())
        .collect();
    
        

    assert_eq!(
        component.len(),
        input.node_count,
        "Graph is expected to be connected, but BFS did not visit all nodes!"
    );
    
}
