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

use pyo3::prelude::*;

use hashbrown::HashMap;

use petgraph::prelude::*;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::VisitMap;
use petgraph::visit::{depth_first_search, DfsEvent, NodeIndexable, Visitable};

fn _build_chain<VM: VisitMap<usize>>(
    parent: &[usize],
    mut u: usize,
    mut v: usize,
    visited: &mut VM,
) -> Vec<(usize, usize)> {
    let mut chain = Vec::new();
    while visited.visit(v) {
        chain.push((u, v));
        u = v;
        v = parent[u];
    }
    chain.push((u, v));

    chain
}

pub fn chain_decomposition(
    graph: &StableUnGraph<PyObject, PyObject>,
    source: Option<usize>,
) -> Vec<Vec<(usize, usize)>> {
    let roots = match source {
        Some(node) => vec![NodeIndex::new(node)],
        None => graph.node_indices().collect(),
    };

    let mut parent = vec![std::usize::MAX; graph.node_bound()];
    let mut back_edges: HashMap<usize, Vec<usize>> = HashMap::new();

    // depth-first-index (DFI) ordered nodes.
    let mut nodes = Vec::with_capacity(graph.node_count());
    depth_first_search(graph, roots, |event| match event {
        DfsEvent::Discover(u, _) => {
            let u = u.index();
            nodes.push(u);
        }
        DfsEvent::TreeEdge(u, v) => {
            let u = u.index();
            let v = v.index();
            parent[v] = u;
        }
        DfsEvent::BackEdge(u, v) => {
            let u = u.index();
            let v = v.index();

            // do *not* consider ``(u, v)`` as a back edge if ``(v, u)`` is a tree edge.
            if parent[u] != v {
                back_edges
                    .entry(v)
                    .and_modify(|v_edges| v_edges.push(u))
                    .or_insert(vec![u]);
            }
        }
        _ => {}
    });

    let visited = &mut graph.visit_map();
    let mut chains = Vec::new();

    for u in nodes {
        visited.visit(u);
        if let Some(vs) = back_edges.get(&u) {
            for v in vs {
                let chain = _build_chain(&parent, u, *v, visited);
                chains.push(chain);
            }
        }
    }
    chains
}
