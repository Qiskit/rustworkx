#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::dictmap::DictMap;
use rustworkx_core::petgraph::graph::Graph;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::Undirected;
use rustworkx_core::shortest_path::{bellman_ford, dijkstra};
use rustworkx_core::Result;

#[derive(Debug, Arbitrary)]
struct FuzzGraph {
    edges: Vec<(usize, usize, u32)>,
    node_count: usize,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = FuzzGraph::arbitrary(&mut Unstructured::new(data)) {
        fuzz_check_bellman_vs_dijkstra(&input);
    }
});

fn fuzz_check_bellman_vs_dijkstra(input: &FuzzGraph) {
    if input.node_count == 0 || input.edges.is_empty() || input.node_count > 1000 {
        return;
    }

    let node_count = input.node_count.min(512);

    let mut graph = Graph::<(), i32, Undirected>::new_undirected();
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(graph.add_node(()));
    }

    for &(u, v, w) in &input.edges {
        if u < node_count && v < node_count {
            let safe_weight = (w % 10_000) as i32;
            graph.add_edge(nodes[u], nodes[v], safe_weight);
        }
    }

    let start_node = nodes[0];

    let bf_res: Result<Option<DictMap<NodeIndex, i32>>> =
        bellman_ford(&graph, start_node, |e| Ok(*e.weight()), None);

    let dijk_res: Result<DictMap<NodeIndex, i32>> =
        dijkstra(&graph, start_node, None, |e| Ok(*e.weight()), None);

    match (bf_res, dijk_res) {
        (Ok(Some(bf_map)), Ok(dijk_map)) => {
            assert_eq!(
                bf_map, dijk_map,
                "Mismatch between Bellman-Ford and Dijkstra"
            );
        }
        (Ok(None), _) => {
            panic!("Bellman-Ford returned None (negative cycle) on non-negative-weight graph");
        }
        (Err(e), _) | (_, Err(e)) => {
            panic!("Fuzzing caused an unexpected error: {e:?}");
        }
    }
}
