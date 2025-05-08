#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::petgraph::graph::Graph;
use rustworkx_core::petgraph::Undirected;
use std::collections::HashSet;

use rustworkx_core::traversal::{breadth_first_search, depth_first_search, BfsEvent, DfsEvent};

// Struct for fuzz input
#[derive(Debug, Arbitrary)]
struct FuzzGraph {
    edges: Vec<(usize, usize)>,
    node_count: usize,
}

// Fuzzing function
fuzz_target!(|data: &[u8]| {
    if let Ok(fuzz_input) = FuzzGraph::arbitrary(&mut Unstructured::new(data)) {
        fuzz_test_traversal_consistency(&fuzz_input);
    }
});

fn fuzz_test_traversal_consistency(input: &FuzzGraph) {
    if input.node_count == 0 {
        return;
    }

    let mut graph = Graph::<(), (), Undirected>::new_undirected();
    let mut nodes = Vec::new();

    for _ in 0..input.node_count {
        nodes.push(graph.add_node(()));
    }

    for &(src, tgt) in &input.edges {
        if src < input.node_count && tgt < input.node_count {
            graph.add_edge(nodes[src], nodes[tgt], ());
        }
    }

    let mut bfs_visited = HashSet::new();
    let mut dfs_visited = HashSet::new();

    for &start_node in &nodes {
        if !bfs_visited.contains(&start_node.index()) {
            breadth_first_search(&graph, vec![start_node], |event| {
                if let BfsEvent::Discover(node) = event {
                    bfs_visited.insert(node.index());
                }
            });
        }

        if !dfs_visited.contains(&start_node.index()) {
            depth_first_search(&graph, vec![start_node], |event| {
                if let DfsEvent::Discover(node, _) = event {
                    dfs_visited.insert(node.index());
                }
            });
        }
    }

    assert_eq!(
        bfs_visited, dfs_visited,
        "BFS and DFS should visit the same set of nodes in any graph."
    );
}
