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

//! This module defines graph traits for multi-graphs.

use hashbrown::HashSet;
use petgraph::visit::{EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, Visitable};
use petgraph::{Directed, Undirected};
use std::hash::Hash;

pub trait HasParallelEdgesUndirected: GraphBase {
    fn has_parallel_edges(&self) -> bool;
}

impl<G> HasParallelEdgesUndirected for G
where
    G: GraphProp<EdgeType = Undirected> + Visitable + EdgeCount,
    G::NodeId: Eq + Hash,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
{
    fn has_parallel_edges(&self) -> bool {
        let mut edges: HashSet<[Self::NodeId; 2]> = HashSet::with_capacity(2 * self.edge_count());
        for edge in self.edge_references() {
            let endpoints = [edge.source(), edge.target()];
            let endpoints_rev = [edge.target(), edge.source()];
            if edges.contains(&endpoints) || edges.contains(&endpoints_rev) {
                return true;
            }
            edges.insert(endpoints);
            edges.insert(endpoints_rev);
        }
        false
    }
}

pub trait HasParallelEdgesDirected: GraphBase {
    fn has_parallel_edges(&self) -> bool;
}

impl<G> HasParallelEdgesDirected for G
where
    G: GraphProp<EdgeType = Directed> + Visitable + EdgeCount,
    G::NodeId: Eq + Hash,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
{
    fn has_parallel_edges(&self) -> bool {
        let mut edges: HashSet<[Self::NodeId; 2]> = HashSet::with_capacity(self.edge_count());
        for edge in self.edge_references() {
            let endpoints = [edge.source(), edge.target()];
            if edges.contains(&endpoints) {
                return true;
            }
            edges.insert(endpoints);
        }
        false
    }
}
