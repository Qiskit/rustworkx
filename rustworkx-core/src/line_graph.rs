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
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, EdgeCount, IntoEdgeReferences, GraphBase, EdgeIndexable, Data};
use petgraph::data::{Build, Create};
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::fmt::Debug;
// pub fn line_graph<G>(graph: G) -> (G, HashMap<EdgeIndex, NodeIndex>) {

pub fn line_graph<K, G, T, F, H, M>(
    graph: K,
    mut default_node_weight: F,
    mut default_edge_weight: H
) -> (G, HashMap::<K::EdgeId, G::NodeId>)
    where
        K: NodeCount + EdgeCount + IntoEdgeReferences + EdgeIndexable + std::fmt::Debug + IntoNodeIdentifiers + IntoEdges,
        G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + Debug,
        G::NodeId: Debug,
        F: FnMut() -> T,
        H: FnMut() -> M,
        T: Clone,
        K::EdgeId: Hash + Eq + Debug,
        K::NodeId: Debug,
        K::EdgeRef: Debug,
{
    println!("I am here");
    println!("Graph {:?}", graph);
    let num_edges = graph.edge_count();
    println!("G has {} edges", num_edges);
    let mut out_graph = G::with_capacity(num_edges, 0);
    let mut out_edge_map = HashMap::<K::EdgeId, G::NodeId>::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let n0 = out_graph.add_node(default_node_weight());
        out_edge_map.insert(edge.id(), n0);
    }
    println!("out_edge_map {:?}: ", out_edge_map);

    for node in graph.node_identifiers() {
        println!("found node {:?}", node);
        // for edge0 in graph.edges(node) {
        //     println!("edge0 is {:?}", edge0);
        //
        // }
        let edges: Vec<K::EdgeRef> = graph.edges(node).collect();
        println!("edges {:?}", edges);
        for i in 0..edges.len() {
            for j in i+1..edges.len() {
                println!("i = {}, edge_i = {:?}, j = {}, edge_j = {:?}", i, edges[i], j, edges[j]);
                let node0 = out_edge_map.get(&edges[i].id()).unwrap();
                let node1 = out_edge_map.get(&edges[j].id()).unwrap();
                println!("node0 = {:?}, node1 = {:?}", node0, node1);
                out_graph.add_edge(*node0, *node1, default_edge_weight());
            }
        }
    }
    println!("Out Graph {:?}", out_graph);
    println!("Out edge map {:?}", out_edge_map);

    (out_graph, out_edge_map)
}


#[cfg(test)]

mod test_line_graph {
    use crate::line_graph::line_graph;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;
    use hashbrown::HashMap;

    use petgraph::graph::{EdgeIndex, NodeIndex};
    use petgraph::Undirected;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_greedy_node_color_simple_graph() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2), (1, 2), (0, 3)]);
        let (out_graph, out_edge_map): (petgraph::graph::UnGraph<(), ()>, HashMap::<petgraph::prelude::EdgeIndex, petgraph::prelude::NodeIndex>) = line_graph(&graph, || (), || ());
        let expected_edge_list = vec![(3, 1), (3, 0), (1, 0), (2, 0), (2, 1)];
        let actual_edge_list = out_graph.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>();

        assert_eq!(expected_edge_list, actual_edge_list);

        let expected_map: HashMap<EdgeIndex, NodeIndex> = [
            (EdgeIndex::new(0), NodeIndex::new(0)),
            (EdgeIndex::new(1), NodeIndex::new(1)),
            (EdgeIndex::new(2), NodeIndex::new(2)),
            (EdgeIndex::new(3), NodeIndex::new(3)),
        ]
        .into_iter()
        .collect();
        assert_eq!(expected_map, out_edge_map);


    }
}
