use crate::connectivity::conn_components::connected_components;
use hashbrown::{HashMap, HashSet};
use petgraph::algo::{astar, min_spanning_tree};
use petgraph::data::{DataMap, Element};
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    EdgeCount, EdgeRef, GraphProp, IntoEdgeReferences, IntoEdges, IntoNeighborsDirected,
    IntoNodeIdentifiers, IntoNodeReferences, NodeIndexable, Visitable,
};
use petgraph::Undirected;
use std::fmt::Debug;
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
    components: Vec<HashSet<G::NodeId>>,
) -> Vec<(Graph<String, i32>, HashMap<String, usize>)>
where
    G: IntoEdgeReferences + NodeIndexable,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: EdgeWeightToNumber,
{
    components
        .into_iter()
        .map(|component| {
            let mut subgraph = Graph::<String, i32>::new();
            // Create map index to NodeIndex of the nodes in each component
            let mut name_idx_map: HashMap<String, usize> = HashMap::new();
            for &node_id in &component {
                let node_index = graph.to_index(node_id);
                // Create the name of the node in subgraph from the original graph
                let node_name = format!("{}", node_index).trim_matches('"').to_string();
                let new_node = subgraph.add_node(node_name.clone());
                // get the index of the node in the subgraph
                let subgraph_node_index = subgraph.to_index(new_node);
                name_idx_map.insert(node_name, subgraph_node_index);
            }
            // Add edges to the subgraph
            for edge in graph.edge_references() {
                if component.contains(&edge.source()) && component.contains(&edge.target()) {
                    let source_name = format!("{}", graph.to_index(edge.source()));
                    let target_name = format!("{}", graph.to_index(edge.target()));
                    let source = name_idx_map[&source_name];
                    let target = name_idx_map[&target_name];
                    let source_nodeidx = NodeIndex::new(source);
                    let target_nodeidx = NodeIndex::new(target);
                    let weight = edge.weight().to_number();
                    subgraph.add_edge(source_nodeidx, target_nodeidx, weight);
                }
            }
            (subgraph, name_idx_map.clone())
        })
        .collect()
}

pub fn minimum_cycle_basis<G, E>(graph: G) -> Result<Vec<Vec<NodeIndex>>, E>
where
    G: EdgeCount
        + IntoNodeIdentifiers
        + IntoNodeReferences
        + NodeIndexable
        + DataMap
        + GraphProp
        + IntoNeighborsDirected
        + Visitable
        + IntoEdges,
    G::NodeWeight: Clone + Debug,
    G::EdgeWeight: Clone + PartialOrd + EdgeWeightToNumber,
    G::NodeId: Eq + Hash + Debug,
{
    let conn_components = connected_components(&graph);
    let mut min_cycle_basis: Vec<Vec<NodeIndex>> = Vec::new();
    let subgraphs_with_maps = create_subgraphs_from_components(&graph, conn_components);
    for (subgraph, name_idx_map) in subgraphs_with_maps {
        let num_cycles =
            _min_cycle_basis(&subgraph, |e| Ok(e.weight().to_number()), &name_idx_map)?;
        min_cycle_basis.extend(num_cycles);
    }
    Ok(min_cycle_basis)
}

fn _min_cycle_basis<G, F, E>(
    graph: G,
    weight_fn: F,
    name_idx_map: &HashMap<String, usize>,
) -> Result<Vec<Vec<NodeIndex>>, E>
where
    G: EdgeCount + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + DataMap,
    G::NodeWeight: Clone + Debug,
    G::EdgeWeight: Clone + PartialOrd,
    G::NodeId: Eq + Hash,
    F: FnMut(&G::EdgeRef) -> Result<i32, E>,
    for<'a> F: Clone,
{
    let mut cb: Vec<Vec<usize>> = Vec::new();
    let num_edges = graph.edge_count();
    let node_map: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(index, node_index)| (node_index, index))
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(num_edges);
    for edge in graph.edge_references() {
        edges.push((node_map[&edge.source()], node_map[&edge.target()]));
    }
    let mst = min_spanning_tree(&graph);
    let mut mst_edges: Vec<(usize, usize)> = Vec::new();
    for element in mst {
        // println!("Element: {:?}", element);
        match element {
            Element::Edge {
                source,
                target,
                weight: _,
            } => {
                mst_edges.push((source, target));
            }
            _ => {}
        }
    }
    let mut chords: Vec<(usize, usize)> = Vec::new();
    for edge in edges.iter() {
        // Check if the edge is not part of the MST
        if !mst_edges.contains(edge) {
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
        let cycle_edges = _min_cycle(&graph, base.clone(), weight_fn.clone(), name_idx_map)?;
        let mut cb_temp: Vec<usize> = Vec::new();
        for edge in cycle_edges.iter() {
            cb_temp.push(edge.1);
        }
        cb.push(cb_temp);
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
    // Using the name_idx_map, convert the node indices in cb to node names
    let cb_node_name: Vec<Vec<String>> = cb
        .iter()
        .map(|cycle| {
            cycle
                .iter()
                .map(|node| {
                    name_idx_map
                        .iter()
                        .find(|(_name, idx)| **idx == *node)
                        .unwrap()
                        .0
                        .clone()
                })
                .collect()
        })
        .collect();
    // Convert the node names in cb_node_name to node indices by convert the node names to numbers then use NodeIndex::new()
    let cb_nodeidx: Vec<Vec<NodeIndex>> = cb_node_name
        .into_iter()
        .map(|inner_vec| {
            inner_vec
                .into_iter()
                .map(|num_str| NodeIndex::new(num_str.parse::<usize>().unwrap()))
                .collect()
        })
        .collect();
    Ok(cb_nodeidx)
}

fn _min_cycle<G, F, E>(
    graph: G,
    orth: HashSet<(usize, usize)>,
    mut weight_fn: F,
    name_idx_map: &HashMap<String, usize>,
) -> Result<Vec<(usize, usize)>, E>
where
    G: IntoNodeReferences + IntoEdgeReferences + DataMap + NodeIndexable,
    G::NodeWeight: Debug,
    F: FnMut(&G::EdgeRef) -> Result<i32, E>,
{
    let mut gi = Graph::<String, i32, Undirected>::new_undirected();
    let mut gi_name_to_node_index = HashMap::new();
    for node_id in graph.node_identifiers() {
        let graph_node_name = graph.node_weight(node_id).unwrap();
        let gi_node_name = format!("{:?}", graph_node_name)
            .trim_matches('"')
            .to_string();
        let gi_lifted_node_name = format!("{}_lifted", gi_node_name);
        let new_node = gi.add_node(gi_node_name.clone());
        let new_node_index = gi.to_index(new_node);
        let lifted_node = gi.add_node(gi_lifted_node_name.clone());
        let lifted_node_index = gi.to_index(lifted_node);
        gi_name_to_node_index.insert(gi_node_name, new_node_index);
        gi_name_to_node_index.insert(gi_lifted_node_name, lifted_node_index);
    }
    // # Add 2 copies of each edge in G to Gi.
    // # If edge is in orth, add cross edge; otherwise in-plane edge
    for edge in graph.edge_references() {
        let u_index = graph.to_index(edge.source());
        let v_index = graph.to_index(edge.target());
        let u_name = format!("{:?}", graph.node_weight(edge.source()).unwrap())
            .trim_matches('"')
            .to_string();
        let v_name = format!("{:?}", graph.node_weight(edge.target()).unwrap())
            .trim_matches('"')
            .to_string();
        let u_lifted_name = format!("{}_lifted", u_name);
        let v_lifted_name = format!("{}_lifted", v_name);
        let weight = weight_fn(&edge)?;
        let gi_u_id = gi_name_to_node_index[&u_name];
        let gi_v_id = gi_name_to_node_index[&v_name];
        let gi_u = NodeIndex::new(gi_u_id);
        let gi_v = NodeIndex::new(gi_v_id);
        if orth.contains(&(u_index, v_index)) || orth.contains(&(v_index, u_index)) {
            // Add cross edges with weight
            gi.add_edge(
                gi_u,
                NodeIndex::new(gi_name_to_node_index[&v_lifted_name]),
                weight,
            );
            gi.add_edge(
                NodeIndex::new(gi_name_to_node_index[&u_lifted_name]),
                gi_v,
                weight,
            );
        } else {
            // Add in-plane edges with weight
            gi.add_edge(
                NodeIndex::new(gi_name_to_node_index[&u_name]),
                NodeIndex::new(gi_name_to_node_index[&v_name]),
                weight,
            );
            gi.add_edge(
                NodeIndex::new(gi_name_to_node_index[&u_lifted_name]),
                NodeIndex::new(gi_name_to_node_index[&v_lifted_name]),
                weight,
            );
        }
    }
    // Instead of finding the shortest path between each node and its lifted node, store the shortest paths in a list to find the shortest paths among them
    let mut shortest_path_map: HashMap<String, i32> = HashMap::new();
    for nodeid in graph.node_identifiers() {
        let node_weight = graph.node_weight(nodeid).unwrap();
        let node_name = format!("{:?}", node_weight).trim_matches('"').to_string();
        let lifted_node_name = format!("{}_lifted", node_name);
        let node = gi_name_to_node_index[&node_name];
        let nodeidx = NodeIndex::new(node);
        let lifted_node = gi_name_to_node_index[&lifted_node_name];
        let lifted_nodeidx = NodeIndex::new(lifted_node);
        let result = astar(
            &gi,
            nodeidx,
            |finish| finish == lifted_nodeidx,
            |e| *e.weight(),
            |_| 0,
        );
        match result {
            Some((cost, _path)) => {
                shortest_path_map.insert(node_name, cost);
            }
            None => {}
        }
    }
    let min_start = shortest_path_map
        .keys()
        .min_by_key(|k| &shortest_path_map[k.as_str()])
        .unwrap();
    let min_start_node_index = gi_name_to_node_index[min_start];
    let min_start_lifted_node_index = gi_name_to_node_index[&format!("{}_lifted", min_start)];
    let min_start_node = NodeIndex::new(min_start_node_index);
    let min_start_lifted_node = NodeIndex::new(min_start_lifted_node_index);
    let result = astar(
        &gi,
        min_start_node,
        |finish| finish == min_start_lifted_node,
        |e| *e.weight(),
        |_| 0,
    );
    // Store the shortest path in a list and translate lifted nodes to original nodes
    let mut min_path: Vec<usize> = Vec::new();
    match result {
        Some((_cost, path)) => {
            for node in path {
                let node_name = gi.node_weight(node).unwrap();
                if node_name.contains("_lifted") {
                    let original_node_name = node_name.replace("_lifted", "");
                    let original_node = name_idx_map[&original_node_name];
                    min_path.push(original_node);
                } else {
                    let original_node = name_idx_map[node_name];
                    min_path.push(original_node);
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
    use crate::connectivity::minimum_cycle_basis;
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
        let expected_output: Vec<Vec<usize>> = vec![vec![0, 1, 3], vec![0, 1, 2, 3]];
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
