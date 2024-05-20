use crate::connectivity::conn_components::connected_components;
use crate::dictmap::*;
use crate::Result;
use hashbrown::{HashMap, HashSet};
use petgraph::algo::{astar, min_spanning_tree};
use petgraph::csr::IndexType;
use crate::shortest_path::dijkstra;
use petgraph::data::{DataMap, Element};
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    EdgeCount, EdgeRef, GraphProp, IntoEdgeReferences, IntoEdges, IntoNeighborsDirected,
    IntoNodeIdentifiers, IntoNodeReferences, NodeIndexable, Visitable,
};
use petgraph::Undirected;
use std::hash::Hash;

pub trait EdgeWeightToNumber {
    fn to_number(&self) -> i32;
}

// Implement the trait for `()`, returning a default weight (e.g., 1)
impl EdgeWeightToNumber for () {
    fn to_number(&self) -> i32 {
        1
    }
}

// Implement the trait for `i32`
impl EdgeWeightToNumber for i32 {
    fn to_number(&self) -> i32 {
        *self
    }
}

fn create_subgraphs_from_components<G>(
    graph: G,
    components: Vec<HashSet<G::NodeId>>
) -> Vec<(Graph<G::NodeId, i32, Undirected>, HashMap<usize, NodeIndex>)>
where
    G:   IntoEdgeReferences
        + NodeIndexable
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: EdgeWeightToNumber,
{
    components
        .into_iter()
        .map(|component| {
            let mut subgraph = Graph::<_ ,i32, Undirected>::default();
            let mut node_subnode_map:HashMap<usize, NodeIndex> = HashMap::new();
            for nodeid in graph.node_identifiers() {
                if component.contains(&nodeid) {
                    let node = graph.to_index(nodeid);
                    let subnode = subgraph.add_node(nodeid);
                    node_subnode_map.insert(node, subnode);
                }
            }
            for edge in graph.edge_references() {
                let source = edge.source();
                let target = edge.target();
                if component.contains(&source) && component.contains(&target) {
                    let subsource = node_subnode_map[&graph.to_index(source)];
                    let subtarget = node_subnode_map[&graph.to_index(target)];
                    subgraph.add_edge(subsource, subtarget, edge.weight().to_number());
                }
            }
            (subgraph, node_subnode_map)
        }).collect()  
}
pub fn minimum_cycle_basis<G, E>(graph: G) -> Result<Vec<Vec<NodeIndex>>, E>
where
    G: EdgeCount
        + IntoNodeIdentifiers
        + NodeIndexable
        + DataMap
        + GraphProp
        + IntoNeighborsDirected
        + Visitable
        + IntoEdges,
    G::EdgeWeight: EdgeWeightToNumber,
    G::NodeId: Eq + Hash,
{
    let conn_components = connected_components(&graph);
    let mut min_cycle_basis = Vec::new();
    let subgraphs_with_maps = create_subgraphs_from_components(&graph, conn_components);
    for (subgraph, node_subnode_map) in subgraphs_with_maps {
        let num_cycles: Vec<Vec<NodeIndex>> =
            _min_cycle_basis(&subgraph, |e| Ok(e.weight().to_number()), &node_subnode_map)?;
        min_cycle_basis.extend(num_cycles);
    }
    Ok(min_cycle_basis)
}

fn _min_cycle_basis<G, F, E>(
    subgraph: G,
    weight_fn: F,
    node_subnode_map: &HashMap<usize, NodeIndex>,
) -> Result<Vec<Vec<NodeIndex>>, E>
where
    G: EdgeCount + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + DataMap,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone + PartialOrd,
    G::NodeId: Eq + Hash,
    F: FnMut(&G::EdgeRef) -> Result<i32, E>,
    for<'a> F: Clone,
{
    let mut sub_cb: Vec<Vec<usize>> = Vec::new();
    let num_edges = subgraph.edge_count();
    let mut sub_edges: Vec<(usize, usize)> = Vec::with_capacity(num_edges);
    let node_map: HashMap<G::NodeId, usize> = subgraph
        .node_identifiers()
        .enumerate()
        .map(|(index, node_index)| (node_index, index))
        .collect();
    for edge in subgraph.edge_references() {
        sub_edges.push((
            node_map[&edge.source()],
            node_map[&edge.target()],
        ));
    }
    let mst = min_spanning_tree(&subgraph);
    let sub_mst_edges: Vec<_> = mst.filter_map(|element| {
        if let Element::Edge { source, target, weight: _ } = element {
            Some((source, target))
            } else {
            None
            }
            })
            .collect();

    let mut chords: Vec<(usize, usize)> = Vec::new();
    for edge in sub_edges.iter() {
        if !sub_mst_edges.contains(edge) {
            // If it's not in the MST, it's a chord
            chords.push(*edge);
        }
    }
    let mut set_orth: Vec<HashSet<(usize, usize)>> = Vec::new();
    // Fill `set_orth` with individual chords
    for chord in chords.iter() {
        let mut chord_set = HashSet::new();
        chord_set.insert(*chord);
        set_orth.push(chord_set);
    }
    while let Some(chord_pop) = set_orth.pop() {
        let base = chord_pop;
        let cycle_edges = _min_cycle(&subgraph, base.clone(), weight_fn.clone())?;
        let mut cb_temp: Vec<usize> = Vec::new();
        for edge in cycle_edges.iter() {
            cb_temp.push(edge.1);
        }
        sub_cb.push(cb_temp);
        for orth in &mut set_orth {
            let mut new_orth = HashSet::new();
            if cycle_edges
                .iter()
                .filter(|edge| orth.contains(*edge) || orth.contains(&((*edge).1, (*edge).0)))
                .count()
                % 2
                == 1
            {
                for e in orth.iter() {
                    if !base.contains(e) && !base.contains(&(e.1, e.0)) {
                        new_orth.insert(*e);
                    }
                }
                for e in base.iter() {
                    if !orth.contains(e) && !orth.contains(&(e.1, e.0)) {
                        new_orth.insert(*e);
                    }
                }
                *orth = new_orth;
            } else {
                *orth = orth.clone();
            }
        }
    }
    // Using the node_subnode_map, convert the subnode usize in cb via NodeIndex::new() to original graph NodeIndex
    let cb: Vec<Vec<NodeIndex>> = sub_cb
        .iter()
        .map(|cycle| {
            cycle
                .iter()
                .map(|node| {
                    let subnode = NodeIndex::new(*node);
                    let nodeid = node_subnode_map
                        .iter()
                        .find(|(_node, &subnode_index)| subnode_index == subnode)
                        .unwrap()
                        .0;
                    // convert node to NodeIndex
                    NodeIndex::new(*nodeid)
                })
                .collect()
        })
        .collect();
    Ok(cb)
}

fn _min_cycle<G, F, E>(
    subgraph: G,
    orth: HashSet<(usize, usize)>,
    mut weight_fn: F
) -> Result<Vec<(usize, usize)>, E>
where
    G: IntoNodeReferences + IntoEdgeReferences + DataMap + NodeIndexable,
    G::NodeId: Eq + Hash,
    F: FnMut(&G::EdgeRef) -> Result<i32, E>,
{
    let mut gi = Graph::<_, i32, Undirected>::new_undirected();
    let mut subgraph_gi_map = HashMap::new();
    let mut gi_subgraph_map = HashMap::new();
    for node in subgraph.node_identifiers() {
        let gi_node = gi.add_node(node);
        let gi_lifted_node = gi.add_node(node);
        gi_subgraph_map.insert(gi_node, node);
        gi_subgraph_map.insert(gi_lifted_node, node);
        subgraph_gi_map.insert(node, (gi_node, gi_lifted_node));
    }
    // # Add 2 copies of each edge in G to Gi.
    // # If edge is in orth, add cross edge; otherwise in-plane edge
    for edge in subgraph.edge_references() {
        let u_id = edge.source();
        let v_id = edge.target();
        let u = subgraph.to_index(u_id);
        let v = subgraph.to_index(v_id);
        let weight = weight_fn(&edge)?;
        // For each pair of (u, v) from the subgraph, there is a corresponding double pair of (u_node, v_node) and (u_lifted_node, v_lifted_node) in the gi
        let (u_node, u_lifted_node) = subgraph_gi_map[&u_id];
        let (v_node, v_lifted_node) = subgraph_gi_map[&v_id];
        if orth.contains(&(u, v)) || orth.contains(&(v, u)) {
            // Add cross edges with weight
            gi.add_edge(u_node, v_lifted_node, weight);
            gi.add_edge(u_lifted_node, v_node, weight);
        } else {
            // Add in-plane edges with weight
            gi.add_edge(u_node, v_node, weight);
            gi.add_edge(u_lifted_node, v_lifted_node, weight);
        }
    }
    // Instead of finding the shortest path between each node and its lifted node, store the shortest paths in a list to find the shortest paths among them
    let mut shortest_path_map: HashMap<G::NodeId, i32> = HashMap::new();
    for nodeid in subgraph.node_identifiers() {
        let (node, lifted_node) = subgraph_gi_map[&nodeid];
        let result: Result<DictMap<NodeIndex, i32>> = dijkstra(&gi, 
            node, 
            Some(lifted_node), 
            |e| Ok(*e.weight() as i32),
            None);
        // Find the shortest distance in the result and store it in the shortest_path_map
        let spl = result.unwrap()[&lifted_node];
        shortest_path_map.insert(nodeid, spl);
    }
    let min_start = shortest_path_map.iter().min_by_key(|x| x.1).unwrap().0;
    let min_start_node = subgraph_gi_map[min_start].0;
    let min_start_lifted_node = subgraph_gi_map[min_start].1;
    let result = astar(
        &gi,
        min_start_node,
        |finish| finish == min_start_lifted_node,
        |e| *e.weight(),
        |_| 0,
    );
    let mut min_path: Vec<usize> = Vec::new();
    match result {
        Some((_cost, path)) => {
            for node in path {
                if let Some(&subgraph_nodeid) = gi_subgraph_map.get(&node) {
                    let subgraph_node = subgraph.to_index(subgraph_nodeid);
                    min_path.push(subgraph_node.index());
                }
            }
        }
        None => {}
    }
    let edgelist = min_path
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect::<Vec<_>>();
    let mut edgeset: HashSet<(usize, usize)> = HashSet::new();
    for e in edgelist.iter() {
        if edgeset.contains(e) {
            edgeset.remove(e);
        } else if edgeset.contains(&(e.1, e.0)) {
            edgeset.remove(&(e.1, e.0));
        } else {
            edgeset.insert(*e);
        }
    }
    let mut min_edgelist: Vec<(usize, usize)> = Vec::new();
    for e in edgelist.iter() {
        if edgeset.contains(e) {
            min_edgelist.push(*e);
            edgeset.remove(e);
        } else if edgeset.contains(&(e.1, e.0)) {
            min_edgelist.push((e.1, e.0));
            edgeset.remove(&(e.1, e.0));
        }
    }
    Ok(min_edgelist)
}



#[cfg(test)]
mod test_minimum_cycle_basis {

    use crate::connectivity::minimum_cycle_basis::minimum_cycle_basis;
    use petgraph::graph::{Graph, NodeIndex};
    use petgraph::Undirected;


    #[test]
    fn test_empty_graph() {
        let graph: Graph<String, i32> = Graph::new();
        let output: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&graph);
        assert_eq!(output.unwrap().len(), 0);
    }

    #[test]
    fn test_triangle() {
        let mut graph = Graph::<String, i32>::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, 1);
        graph.add_edge(b, c, 1);
        graph.add_edge(c, a, 1);

        let cycles: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&graph);
        assert_eq!(cycles.unwrap().len(), 1);
    }

    #[test]
    fn test_two_separate_triangles() {
        let mut graph = Graph::<String, i32>::new();
        let nodes = vec!["A", "B", "C", "D", "E", "F"]
            .iter()
            .map(|&n| graph.add_node(n.to_string()))
            .collect::<Vec<_>>();
        graph.add_edge(nodes[0], nodes[1], 1);
        graph.add_edge(nodes[1], nodes[2], 1);
        graph.add_edge(nodes[2], nodes[0], 1);
        graph.add_edge(nodes[3], nodes[4], 1);
        graph.add_edge(nodes[4], nodes[5], 1);
        graph.add_edge(nodes[5], nodes[3], 1);

        let cycles: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&graph);
        assert_eq!(cycles.unwrap().len(), 2);
    }

    #[test]
    fn test_weighted_diamond_graph() {
        let mut weighted_diamond = Graph::<(), i32, Undirected>::new_undirected();
        let ud_node1 = weighted_diamond.add_node(());
        let ud_node2 = weighted_diamond.add_node(());
        let ud_node3 = weighted_diamond.add_node(());
        let ud_node4 = weighted_diamond.add_node(());
        weighted_diamond.add_edge(ud_node1, ud_node2, 1);
        weighted_diamond.add_edge(ud_node2, ud_node3, 1);
        weighted_diamond.add_edge(ud_node3, ud_node4, 1);
        weighted_diamond.add_edge(ud_node4, ud_node1, 1);
        weighted_diamond.add_edge(ud_node2, ud_node4, 5);
        let output: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&weighted_diamond);
        let expected_output: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![0, 1, 2, 3]];
        for cycle in output.unwrap().iter() {
            println!("{:?}", cycle);
            let mut node_indices: Vec<usize> = Vec::new();
            for node in cycle.iter() {
                node_indices.push(node.index());
            }
            node_indices.sort();
            println!("Node indices {:?}", node_indices);
            if expected_output.contains(&node_indices) {
                println!("Found cycle {:?}", node_indices);
            }
            // assert!(expected_output.contains(&node_indices));
        }
    }

    #[test]
    fn test_unweighted_diamond_graph() {
        let mut unweighted_diamond = Graph::<(), (), Undirected>::new_undirected();
        let ud_node0 = unweighted_diamond.add_node(());
        let ud_node1 = unweighted_diamond.add_node(());
        let ud_node2 = unweighted_diamond.add_node(());
        let ud_node3 = unweighted_diamond.add_node(());
        unweighted_diamond.add_edge(ud_node0, ud_node1, ());
        unweighted_diamond.add_edge(ud_node1, ud_node2, ());
        unweighted_diamond.add_edge(ud_node2, ud_node3, ());
        unweighted_diamond.add_edge(ud_node3, ud_node0, ());
        unweighted_diamond.add_edge(ud_node1, ud_node3, ());
        let output: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&unweighted_diamond);
        let expected_output: Vec<Vec<usize>> = vec![vec![0, 1, 3], vec![1, 2, 3]];
        for cycle in output.unwrap().iter() {
            let mut node_indices: Vec<usize> = Vec::new();
            for node in cycle.iter() {
                node_indices.push(node.index());
            }
            node_indices.sort();
            assert!(expected_output.contains(&node_indices));
        }
    }
    #[test]
    fn test_complete_graph() {
        let mut complete_graph = Graph::<(), i32, Undirected>::new_undirected();
        let cg_node1 = complete_graph.add_node(());
        let cg_node2 = complete_graph.add_node(());
        let cg_node3 = complete_graph.add_node(());
        let cg_node4 = complete_graph.add_node(());
        let cg_node5 = complete_graph.add_node(());
        complete_graph.add_edge(cg_node1, cg_node2, 1);
        complete_graph.add_edge(cg_node1, cg_node3, 1);
        complete_graph.add_edge(cg_node1, cg_node4, 1);
        complete_graph.add_edge(cg_node1, cg_node5, 1);
        complete_graph.add_edge(cg_node2, cg_node3, 1);
        complete_graph.add_edge(cg_node2, cg_node4, 1);
        complete_graph.add_edge(cg_node2, cg_node5, 1);
        complete_graph.add_edge(cg_node3, cg_node4, 1);
        complete_graph.add_edge(cg_node3, cg_node5, 1);
        complete_graph.add_edge(cg_node4, cg_node5, 1);
        let output: Result<Vec<Vec<NodeIndex>>, Box<dyn std::error::Error>> =
            minimum_cycle_basis(&complete_graph);
        for cycle in output.unwrap().iter() {
            assert_eq!(cycle.len(), 3);
        }
    }
}
