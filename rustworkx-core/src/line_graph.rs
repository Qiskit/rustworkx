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

use std::hash::Hash;

use hashbrown::HashMap;
use petgraph::data::Create;
use petgraph::visit::{Data, EdgeCount, EdgeRef, IntoEdges, IntoNodeIdentifiers};

/// Constructs the line graph of an undirected graph.
///
/// The line graph `L(G)` of a graph `G` represents the adjacencies between edges of G.
/// `L(G)` contains a vertex for every edge in `G`, and `L(G)` contains an edge between two
/// vertices if the corresponding edges in `G` have a vertex in common.
///
/// Arguments:
///
/// * `input_graph` - The input graph `G`.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// Returns the constructed line graph `L(G)`, and the map from the edges of `L(G)` to
/// the vertices of `G`.
///
/// # Example
/// ```rust
/// use rustworkx_core::line_graph::line_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
/// use rustworkx_core::petgraph::Graph;
/// use hashbrown::HashMap;
/// use petgraph::graph::{EdgeIndex, NodeIndex};
/// use petgraph::Undirected;
///
/// let input_graph =
///   Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2), (1, 2), (0, 3)]);
///
/// let (output_graph, output_edge_map): (
///     petgraph::graph::UnGraph<(), ()>,
///     HashMap<petgraph::prelude::EdgeIndex, petgraph::prelude::NodeIndex>,
/// ) = line_graph(&input_graph, || (), || ());
///
/// let output_edge_list = output_graph
///     .edge_references()
///     .map(|edge| (edge.source().index(), edge.target().index()))
///     .collect::<Vec<(usize, usize)>>();
///
/// let expected_edge_list = vec![(3, 1), (3, 0), (1, 0), (2, 0), (2, 1)];
/// let expected_edge_map: HashMap<EdgeIndex, NodeIndex> = [
///     (EdgeIndex::new(0), NodeIndex::new(0)),
///     (EdgeIndex::new(1), NodeIndex::new(1)),
///     (EdgeIndex::new(2), NodeIndex::new(2)),
///     (EdgeIndex::new(3), NodeIndex::new(3)),
/// ]
/// .into_iter()
/// .collect();
///
/// assert_eq!(output_edge_list, expected_edge_list);
/// assert_eq!(output_edge_map, expected_edge_map);
/// ```
pub fn line_graph<K, G, T, F, H, M>(
    input_graph: K,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> (G, HashMap<K::EdgeId, G::NodeId>)
where
    K: EdgeCount + IntoNodeIdentifiers + IntoEdges,
    G: Create + Data<NodeWeight = T, EdgeWeight = M>,
    F: FnMut() -> T,
    H: FnMut() -> M,
    K::EdgeId: Hash + Eq,
{
    let num_edges = input_graph.edge_count();
    let mut output_graph = G::with_capacity(num_edges, 0);
    let mut output_edge_map =
        HashMap::<K::EdgeId, G::NodeId>::with_capacity(input_graph.edge_count());

    for edge in input_graph.edge_references() {
        let new_node = output_graph.add_node(default_node_weight());
        output_edge_map.insert(edge.id(), new_node);
    }

    for node in input_graph.node_identifiers() {
        let edges: Vec<K::EdgeRef> = input_graph.edges(node).collect();
        for i in 0..edges.len() {
            for j in i + 1..edges.len() {
                let node0 = output_edge_map.get(&edges[i].id()).unwrap();
                let node1 = output_edge_map.get(&edges[j].id()).unwrap();
                output_graph.add_edge(*node0, *node1, default_edge_weight());
            }
        }
    }
    (output_graph, output_edge_map)
}

#[cfg(test)]

mod test_line_graph {
    use crate::line_graph::line_graph;
    use crate::petgraph::visit::EdgeRef;
    use crate::petgraph::Graph;
    use hashbrown::HashMap;
    use petgraph::graph::{EdgeIndex, NodeIndex};
    use petgraph::Undirected;

    #[test]
    fn test_simple_graph() {
        // Simple graph
        let input_graph =
            Graph::<(), (), Undirected>::from_edges(&[(0, 1), (2, 3), (3, 4), (4, 5)]);

        let (output_graph, output_edge_map): (
            petgraph::graph::UnGraph<(), ()>,
            HashMap<petgraph::prelude::EdgeIndex, petgraph::prelude::NodeIndex>,
        ) = line_graph(&input_graph, || (), || ());

        let output_edge_list = output_graph
            .edge_references()
            .map(|edge| (edge.source().index(), edge.target().index()))
            .collect::<Vec<(usize, usize)>>();

        let expected_edge_list = vec![(2, 1), (3, 2)];
        let expected_edge_map: HashMap<EdgeIndex, NodeIndex> = [
            (EdgeIndex::new(0), NodeIndex::new(0)),
            (EdgeIndex::new(1), NodeIndex::new(1)),
            (EdgeIndex::new(2), NodeIndex::new(2)),
            (EdgeIndex::new(3), NodeIndex::new(3)),
        ]
        .into_iter()
        .collect();

        assert_eq!(output_edge_list, expected_edge_list);
        assert_eq!(output_edge_map, expected_edge_map);
    }
}
