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

use petgraph::algo::{toposort, Cycle};
use petgraph::data::Build;
use petgraph::visit::{
    GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, Visitable,
};

use crate::traversal::descendants_at_distance;

/// Build a transitive closure out of a given DAG
///
/// This function will mutate a given DAG object (which is typically moved to
/// this function) into a transitive closure of the graph and then returned.
/// If you'd like to preserve the input graph pass a clone of the original graph.
/// The transitive closure of :math:`G = (V, E)` is a graph :math:`G+ = (V, E+)`
/// such that for all pairs of :math:`v, w` in :math:`V` there is an edge
/// :math:`(v, w) in :math:`E+` if and only if there is a non-null path from
/// :math:`v` to :math:`w` in :math:`G`. This funciton provides an optimized
/// path for computing the the transitive closure of a DAG, if the input graph
/// contains cycles it will error.
///
/// Arguments:
///
///  - `graph`: A mutable graph object representing the DAG
///  - `topological_order`: An optional `Vec` of node identifiers representing
///     the topological order to traverse the DAG with. If not specified the
///     `petgraph::algo::toposort` function will be called to generate this
///  - `default_edge_weight`: A callable function that takes no arguments and
///     returns the `EdgeWeight` type object to use for each edge added to
///     `graph
///
/// # Example
///
/// ```rust
/// use rustworkx_core::traversal::build_transitive_closure_dag;
///
/// let g = petgraph::graph::DiGraph::<i32, i32>::from_edges(&[(0, 1, 0), (1, 2, 0), (2, 3, 0)]);
///
/// let res = build_transitive_closure_dag(g, None, || -> i32 {0});
/// let out_graph = res.unwrap();
/// let out_edges: Vec<(usize, usize)> = out_graph
///     .edge_indices()
///     .map(|e| {
///         let endpoints = out_graph.edge_endpoints(e).unwrap();
///         (endpoints.0.index(), endpoints.1.index())
///     })
///     .collect();
/// assert_eq!(vec![(0, 1), (1, 2), (2, 3), (1, 3), (0, 3), (0, 2)], out_edges)
/// ```
pub fn build_transitive_closure_dag<'a, G, F>(
    mut graph: G,
    topological_order: Option<Vec<G::NodeId>>,
    default_edge_weight: F,
) -> Result<G, Cycle<G::NodeId>>
where
    G: NodeCount + Build + Clone,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Visitable + IntoNeighborsDirected + IntoNodeIdentifiers,
    G::NodeId: std::cmp::Eq + std::hash::Hash,
    F: Fn() -> G::EdgeWeight,
{
    let node_order: Vec<G::NodeId> = match topological_order {
        Some(topo_order) => topo_order,
        None => toposort(&graph, None)?,
    };
    for node in node_order.into_iter().rev() {
        for descendant in descendants_at_distance(&graph, node, 2) {
            graph.add_edge(node, descendant, default_edge_weight());
        }
    }
    Ok(graph)
}
