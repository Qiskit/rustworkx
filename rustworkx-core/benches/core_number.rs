// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

// Reproducible full-call benchmark with no extra dependencies.
// Run the same source against the base and proposed crate versions:
// RAYON_NUM_THREADS=1 cargo bench -p rustworkx-core --bench core_number -- 10000 20
// Arguments are node count and sample count. Output is CSV; one warmup per case.
// Graph construction and exact answer validation are outside the timed calls.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hint::black_box;
use std::time::Instant;

use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use rustworkx_core::connectivity::core_number;

// Match the Python fixture's integer generator and edge insertion order.
fn irregular_edges(n: usize) -> Vec<(usize, usize)> {
    let connected = n - 1;
    let dense = (connected * 3 / 4).max(2);
    let mut state = 0xA076_1D64_78BD_642F_u64;
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |source, target| {
        if seen.insert((source, target)) {
            edges.push((source, target));
        }
    };
    for source in 0..dense {
        let attempts = if source % 31 == 0 {
            64
        } else {
            1 + (source * 17 + 11) % 16
        };
        for attempt in 0..attempts.min(dense - 1) {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let target = (source + 1 + (state % (dense - 1) as u64) as usize) % dense;
            add(source, target);
            if attempt % 3 == 0 {
                add(target, source);
            }
        }
    }
    for source in dense..connected {
        for offset in 0..1 + source % 3 {
            add(source, (source * 17 + offset * 13) % dense);
        }
    }
    edges
}

fn reference_core_numbers(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    // Independent min-degree elimination with a heap, no degree buckets or dense markers.
    let mut adjacency = vec![HashSet::new(); n];
    for &(source, target) in edges {
        adjacency[source].insert(target);
        adjacency[target].insert(source);
    }
    let mut degree: Vec<_> = adjacency.iter().map(HashSet::len).collect();
    let mut pending: BinaryHeap<_> = degree
        .iter()
        .enumerate()
        .map(|(node, &value)| Reverse((value, node)))
        .collect();
    let mut removed = vec![false; n];
    let mut expected = vec![0; n];
    let mut level = 0;
    while let Some(Reverse((value, node))) = pending.pop() {
        if removed[node] || value != degree[node] {
            continue;
        }
        level = level.max(value);
        expected[node] = level;
        removed[node] = true;
        for &neighbor in &adjacency[node] {
            if !removed[neighbor] {
                degree[neighbor] -= 1;
                pending.push(Reverse((degree[neighbor], neighbor)));
            }
        }
    }
    expected
}

fn fixture(shape: &str, n: usize, stride: usize) -> (StableDiGraph<(), ()>, Vec<usize>) {
    let mut graph = StableDiGraph::default();
    for _ in 0..n * stride {
        graph.add_node(());
    }
    for i in 0..n * stride {
        if i % stride != 0 {
            graph.remove_node(NodeIndex::new(i));
        }
    }
    let mut expected = vec![0; n];
    let mut edge = |a, b| {
        graph.add_edge(NodeIndex::new(a * stride), NodeIndex::new(b * stride), ());
    };
    // The final live node is an isolate in every case.
    let connected = n - 1;
    match shape {
        "ring" => {
            let steps = 4.min((connected - 1) / 2);
            for (a, core) in expected[..connected].iter_mut().enumerate() {
                *core = 2 * steps;
                for step in 1..=steps {
                    let b = (a + step) % connected;
                    edge(a, b);
                    edge(b, a);
                }
            }
        }
        "hub" => {
            expected[..connected].fill(1);
            for b in 1..connected {
                edge(0, b);
            }
        }
        "irregular" => {
            let edges = irregular_edges(n);
            expected = reference_core_numbers(n, &edges);
            for (source, target) in edges {
                edge(source, target);
            }
        }
        "cliques" | "clique" => {
            let width = if shape == "clique" { connected } else { 16 };
            for start in (0..connected).step_by(width) {
                let end = (start + width).min(connected);
                expected[start..end].fill(end - start - 1);
                for a in start..end {
                    for b in a + 1..end {
                        edge(a, b);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    (graph, expected)
}

fn main() {
    let mut args = std::env::args().skip(1).filter(|arg| arg != "--bench");
    let n = args.next().map_or(10_000, |value| value.parse().unwrap());
    let samples = args.next().map_or(20, |value| value.parse().unwrap());
    assert!(n >= 8 && samples > 0 && args.next().is_none());
    println!("shape,nodes,edges,stride,sample,nanoseconds");
    for shape in ["ring", "hub", "cliques", "clique", "irregular"] {
        // Bound the quadratic fixture independently of the sparse graph size.
        let n = if shape == "clique" { n.min(256) } else { n };
        for stride in [1, 2] {
            let (graph, expected) = fixture(shape, n, stride);
            for sample in 0..=samples {
                let start = Instant::now();
                let result = core_number(black_box(&graph));
                let elapsed = start.elapsed().as_nanos();
                black_box(&result);
                assert_eq!(result.len(), n);
                for (i, ((node, actual), expected)) in result.iter().zip(&expected).enumerate() {
                    assert_eq!(node.index(), i * stride);
                    assert_eq!(actual, expected);
                }
                if sample != 0 {
                    println!(
                        "{shape},{n},{},{stride},{sample},{elapsed}",
                        graph.edge_count()
                    );
                }
            }
        }
    }
}
