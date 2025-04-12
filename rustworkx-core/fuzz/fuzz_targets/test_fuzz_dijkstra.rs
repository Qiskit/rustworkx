#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustworkx_core::dictmap::DictMap;
use rustworkx_core::petgraph::graph::Graph;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::Undirected;
use rustworkx_core::shortest_path::dijkstra;
use rustworkx_core::Result;

#[derive(Debug, Arbitrary)]
struct FuzzGraph {
    edges: Vec<(usize, usize, usize)>,
    node_count: usize,
}

fuzz_target!(|data: &[u8]| {
    if let Ok(fuzz_input) = FuzzGraph::arbitrary(&mut Unstructured::new(data)) {
        fuzz_check_dijkstra_triangle_inequality(&fuzz_input);
    }
});

fn fuzz_check_dijkstra_triangle_inequality(input: &FuzzGraph) {
    if input.node_count == 0
        || input.edges.is_empty()
        || input.node_count > 1000
        || input.edges.len() > 10_000
    {
        return;
    }
    let mut graph = Graph::<(), usize, Undirected>::new_undirected();
    let mut nodes = Vec::with_capacity(input.node_count);
    for _ in 0..input.node_count {
        nodes.push(graph.add_node(()));
    }

    for &(u, v, w) in &input.edges {
        if u < input.node_count && v < input.node_count && w > 0 {
            graph.add_edge(nodes[u], nodes[v], w);
        }
    }

    let start_node = nodes[0];

    let res: Result<DictMap<NodeIndex, usize>> =
        dijkstra(&graph, start_node, None, |e| Ok(*e.weight()), None);

    let dist_map = res.unwrap();

    for edge in graph.edge_references() {
        let u = edge.source();
        let v = edge.target();
        let w = *edge.weight();

        if let (Some(&du), Some(&dv)) = (dist_map.get(&u), dist_map.get(&v)) {
            if let Some(bound) = du.checked_add(w) {
                assert!(
                    dv <= bound,
                    "Triangle inequality failed: dist[{v:?}] = {dv}, dist[{u:?}] = {du}, w = {w}"
                );
            }

            if let Some(bound) = dv.checked_add(w) {
                assert!(
                    du <= bound,
                    "Triangle inequality failed: dist[{u:?}] = {du}, dist[{v:?}] = {dv}, w = {w}"
                );
            }
        }
    }
}
