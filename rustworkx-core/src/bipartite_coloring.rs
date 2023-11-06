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

use crate::dictmap::*;
use std::cmp::{max, min};

use std::hash::Hash;

use hashbrown::HashMap;

use crate::coloring::two_color;
use petgraph::data::DataMap;
use petgraph::graph::{edge_index, Node, NodeIndex, UnGraph};
use petgraph::prelude::StableGraph;
use petgraph::stable_graph::EdgeIndex;
use petgraph::visit::{
    EdgeCount, EdgeIndexable, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};
use petgraph::{Graph, Undirected};
use rayon_cond::CondIterator::Parallel;
use std::error::Error;
use std::fmt;
use std::fmt::Debug;

/// Error returned by bipartite coloring if the graph is not bipartite
/// is impossible
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
pub struct GraphNotBipartite;

impl Error for GraphNotBipartite {}
impl fmt::Display for GraphNotBipartite {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No mapping possible.")
    }
}

#[derive(Clone, Debug)]
struct EdgeData {
    // multiplicity of the edge
    multiplicity: usize,
    // true if considered a bad edge
    bad: bool,
}

type EdgedGraph = StableGraph<(), EdgeData, Undirected>;
type Matching = Vec<(NodeIndex, NodeIndex)>;

struct RegularBipartiteMultiGraph {
    graph: EdgedGraph,
    degree: usize,
    l_nodes: Vec<NodeIndex>,
    r_nodes: Vec<NodeIndex>,
}

impl RegularBipartiteMultiGraph {
    fn new() -> Self {
        let graph: EdgedGraph = StableGraph::default();

        RegularBipartiteMultiGraph {
            graph: graph,
            degree: 0,
            l_nodes: vec![],
            r_nodes: vec![],
        }
    }

    fn copy(parent: &RegularBipartiteMultiGraph) -> Self {
        RegularBipartiteMultiGraph {
            graph: parent.graph.clone(),
            degree: parent.degree.clone(),
            l_nodes: parent.l_nodes.clone(),
            r_nodes: parent.r_nodes.clone(),
        }
    }

    fn copy_without_edges(parent: &RegularBipartiteMultiGraph) -> Self {
        let mut graph_copy: EdgedGraph = parent.graph.clone();
        graph_copy.clear_edges();
        RegularBipartiteMultiGraph {
            graph: graph_copy,
            degree: 0,
            l_nodes: parent.l_nodes.clone(),
            r_nodes: parent.r_nodes.clone(),
        }
    }

    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, multiplicity: usize, bad: bool) {
        let edge = self.graph.find_edge(a, b);
        if edge.is_some() {
            let mut edge_data = self.graph.edge_weight_mut(edge.unwrap()).unwrap();
            edge_data.multiplicity += multiplicity;
        } else {
            let edge_data = EdgeData { multiplicity, bad };
            self.graph.add_edge(a, b, edge_data);
        }
    }

    fn remove_edge(&mut self, a: NodeIndex, b: NodeIndex, multiplicity: usize) {
        let edge = self.graph.find_edge(a, b);
        let mut edge_data = self.graph.edge_weight_mut(edge.unwrap()).unwrap();
        edge_data.multiplicity -= multiplicity;
        if edge_data.multiplicity == 0 {
            self.graph.remove_edge(edge.unwrap());
        }
    }

    fn add_matching(&mut self, matching: &Matching) {
        for (a, b) in matching {
            self.add_edge(*a, *b, 1, false);
        }
        self.degree += 1;
    }

    fn remove_matching(&mut self, matching: &Matching) {
        for (a, b) in matching {
            self.remove_edge(*a, *b, 1);
        }
        self.degree -= 1;
    }

    fn print(&self) {
        let nodes: Vec<_> = self.graph.node_indices().collect();
        let edges: Vec<_> = self.graph.edge_references().collect();
        println!("===============");
        println!("degree: {:?}", self.degree);
        println!("nodes:");
        for node in nodes {
            println!("  {:?}", node);
        }
        println!("edges:");
        for edge in edges {
            println!("  {:?}", edge);
        }
        // println!("map: {:?}", self.node_map);
        // println!("revmap: {:?}", self.rev_node_map);
        println!("l_nodes: {:?}", self.l_nodes);
        println!("r_nodes: {:?}", self.r_nodes);
        println!("===============");
    }
}

pub fn bipartite_edge_color<G>(
    input_graph: G,
    input_l_nodes: &Vec<G::NodeId>,
    input_r_nodes: &Vec<G::NodeId>,
) -> DictMap<G::EdgeId, usize>
where
    G: GraphBase
        + NodeCount
        + NodeIndexable
        + IntoNodeIdentifiers
        + IntoEdges
        + EdgeCount
        + EdgeIndexable,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    let mut rbmg: RegularBipartiteMultiGraph = RegularBipartiteMultiGraph::new();

    let mut node_map: HashMap<G::NodeId, NodeIndex> =
        HashMap::with_capacity(input_graph.node_count());
    let mut rev_node_map: HashMap<NodeIndex, G::NodeId> =
        HashMap::with_capacity(input_graph.node_count());

    let input_graph_node_indices: Vec<G::NodeId> = input_graph.node_identifiers().collect();

    // Compute maximum degree of a node in input_graph
    let mut max_degree: usize = 0;
    for input_node_id in &input_graph_node_indices {
        let mut node_degree = 0;
        for edge in input_graph.edges(*input_node_id) {
            node_degree += 1;
        }
        max_degree = max(max_degree, node_degree);
    }
    // println!("max degree is {:?}", max_degree);
    // Add nodes to multi-graph
    // ToDo: add optimization to combine nodes
    for input_node_id in &input_graph_node_indices {
        let rbmg_node_id = rbmg.graph.add_node(());
        node_map.insert(*input_node_id, rbmg_node_id);
        rev_node_map.insert(rbmg_node_id, *input_node_id);
    }

    // Set l_nodes and r_nodes
    for input_node_id in input_l_nodes {
        rbmg.l_nodes.push(*node_map.get(input_node_id).unwrap());
    }
    for input_node_id in input_r_nodes {
        rbmg.r_nodes.push(*node_map.get(input_node_id).unwrap());
    }

    // Adds edges (note that input_graph may have multiple edges between the same
    // pair of vertices
    for edge in input_graph.edge_references() {
        // todo: endpoints
        let mapped_source = *node_map.get(&edge.source()).unwrap();
        let mapped_target = *node_map.get(&edge.target()).unwrap();
        rbmg.add_edge(mapped_source, mapped_target, 1, false);
    }

    // Create additional left or right vertices
    let n: usize = max(rbmg.l_nodes.len(), rbmg.r_nodes.len());

    while rbmg.l_nodes.len() < n {
        let new_node_index = rbmg.graph.add_node(());
        // todo: possibly update reverse node map
        rbmg.l_nodes.push(new_node_index);
    }

    while rbmg.r_nodes.len() < n {
        let new_node_index = rbmg.graph.add_node(());
        // todo: possibly update reverse node map
        rbmg.r_nodes.push(new_node_index);
    }

    let mut l_index = 0;
    let mut r_index = 0;

    while l_index < n {
        let l_degree: usize = rbmg
            .graph
            .edges(rbmg.l_nodes[l_index])
            .map(|e| e.weight().multiplicity)
            .sum();
        if l_degree == max_degree {
            l_index += 1;
            continue;
        }
        let r_degree: usize = rbmg
            .graph
            .edges(rbmg.r_nodes[r_index])
            .map(|e| e.weight().multiplicity)
            .sum();
        if r_degree == max_degree {
            r_index += 1;
            continue;
        }

        // println!("max_degree = {:?}, l_index = {:?}, l_degree = {:?}, r_index = {:?}, r_degree = {:?} ", max_degree, l_index, l_degree, r_index, r_degree);
        let multiplicity = min(max_degree - l_degree, max_degree - r_degree);
        rbmg.add_edge(
            rbmg.l_nodes[l_index],
            rbmg.r_nodes[r_index],
            multiplicity,
            false,
        );
    }
    rbmg.degree = max_degree;

    // rbmg.print();

    let coloring = rbmg_edge_color(&rbmg);
    // println!("Discovered coloring {:?}", coloring);

    // Let's build map (NodeIndex, NodeIndex) -> list of colors
    let mut endpoints_to_colors: DictMap<(NodeIndex, NodeIndex), Vec<usize>> =
        DictMap::with_capacity(rbmg.graph.edge_count());
    for (color, matching) in coloring.iter().enumerate() {
        for (a, b) in matching {
            let (a0, b0) = if a.index() < b.index() {
                (a, b)
            } else {
                (b, a)
            };
            // println!("a0 = {:?}, b0 = {:?}, colors = {}", a0, b0, color);
            if let Some(colors) = endpoints_to_colors.get_mut(&(*a0, *b0)) {
                colors.push(color);
            } else {
                endpoints_to_colors.insert((*a0, *b0), vec![color]);
            }
        }
    }
    // println!("and the hashmap is:");
    // println!("{:?}", endpoints_to_colors);

    // Iterate over all edges in the original graph...
    let mut edge_coloring: DictMap<G::EdgeId, usize> =
        DictMap::with_capacity(input_graph.edge_count());

    for edge in input_graph.edge_references() {
        let a = *node_map.get(&edge.source()).unwrap();
        let b = *node_map.get(&edge.target()).unwrap();
        let (a0, b0) = if a.index() < b.index() {
            (a, b)
        } else {
            (b, a)
        };
        let colors = endpoints_to_colors.get_mut(&(a0, b0)).unwrap();
        let last_color = colors.last().unwrap();
        //EdgeIndexable::to_index(&input_graph, edge.id())
        edge_coloring.insert(edge.id(), *last_color);
        colors.pop();
    }

    edge_coloring
}

pub fn if_bipartite_edge_color<G>(
    input_graph: G,
) -> Result<DictMap<G::EdgeId, usize>, GraphNotBipartite>
where
    G: GraphBase
        + NodeCount
        + NodeIndexable
        + IntoNodeIdentifiers
        + IntoEdges
        + EdgeCount
        + EdgeIndexable
        + IntoNeighborsDirected
        + GraphProp,
    G::NodeId: Eq + Hash + Debug,
    G::EdgeId: Eq + Hash,
{
    let two_color_result = two_color(&input_graph);
    if let Some(two_coloring) = two_color_result {
        let mut l_nodes: Vec<G::NodeId> = Vec::new();
        let mut r_nodes: Vec<G::NodeId> = Vec::new();
        for (node_id, color) in &two_coloring {
            // println!("HERE node_id = {:?} and color = {:?}", node_id, color);
            if *color == 0 {
                l_nodes.push(*node_id);
            } else {
                r_nodes.push(*node_id);
            }
        }
        return Ok(bipartite_edge_color(&input_graph, &l_nodes, &r_nodes));
    } else {
        return Err(GraphNotBipartite {});
    }
}

/// Returns a set of Euler cycles for a possibly disconnected graph.
///
///

fn other_node(graph: &EdgedGraph, edge_index: EdgeIndex, node_index: NodeIndex) -> NodeIndex {
    let (node1, node2) = graph.edge_endpoints(edge_index).unwrap();
    return if node_index == node1 { node2 } else { node1 };
}

fn euler_cycles(input_graph: &EdgedGraph) -> Vec<Vec<EdgeIndex>> {
    let mut cycles: Vec<Vec<EdgeIndex>> = Vec::new();

    let mut graph = input_graph.clone();
    let mut in_component = vec![false; graph.node_bound()];

    let node_indices: Vec<NodeIndex> = graph.node_identifiers().collect();

    for node_id in node_indices {
        if in_component[node_id.index()] {
            continue;
        }
        let mut stack: Vec<(NodeIndex, Option<EdgeIndex>)> = vec![(node_id, None)];
        let mut cycle: Vec<EdgeIndex> = vec![];
        while !stack.is_empty() {
            let (last_node_id, last_edge_id) = stack.last().unwrap();
            in_component[last_node_id.index()] = true;

            // println!("examining node = {:?}, edge = {:?}", last_node_id, last_edge_id);

            let new_edge = graph.edges(*last_node_id).next();
            // println!("=> new_edge = {:?}", new_edge);

            if new_edge.is_none() {
                // println!("adding {:?} to cycle", last_edge_id);
                if last_edge_id.is_some() {
                    cycle.push(last_edge_id.unwrap());
                }
                stack.pop();
            } else {
                let new_edge_id = new_edge.unwrap().id();
                let new_node_id = other_node(&graph, new_edge_id, *last_node_id);
                stack.push((new_node_id, Some(new_edge_id)));
                // println!("pushing {:?}, {:?} to stack", new_node_id, new_edge_id);

                // println!("BEFORE REMOVING");
                // print_graph(&graph);
                graph.remove_edge(new_edge_id);
                // println!("AFTER REMOVING");
                // print_graph(&graph);
            }
        }
        // println!("new cycle = {:?}", cycle);
        cycles.push(cycle);
    }

    cycles
}

fn rbmg_split_into_two(
    g: &RegularBipartiteMultiGraph,
) -> (RegularBipartiteMultiGraph, RegularBipartiteMultiGraph) {
    if g.degree % 2 == 1 {
        panic!("Cannot split multi-graph of odd degree.")
    }

    let mut h1: RegularBipartiteMultiGraph = RegularBipartiteMultiGraph::copy_without_edges(&g);
    h1.degree = g.degree.clone() / 2;

    let mut h2: RegularBipartiteMultiGraph = RegularBipartiteMultiGraph::copy_without_edges(&g);
    h2.degree = g.degree.clone() / 2;

    let mut r: EdgedGraph = g.graph.clone();
    r.clear_edges();

    for edge in g.graph.edge_references() {
        let multiplicity = edge.weight().multiplicity;
        let bad = edge.weight().bad;
        if multiplicity >= 2 {
            h1.add_edge(edge.source(), edge.target(), multiplicity / 2, bad);
            h2.add_edge(edge.source(), edge.target(), multiplicity / 2, bad);
        }
        if multiplicity % 2 == 1 {
            r.add_edge(
                edge.source(),
                edge.target(),
                EdgeData {
                    multiplicity: 1,
                    bad: bad,
                },
            );
        }
    }

    // println!("r:");
    // print_graph(&r);
    let cycles = euler_cycles(&r);
    // println!("detected cycles = {:?}", cycles);

    for cycle in cycles.into_iter() {
        for i in 0..cycle.len() {
            let (source, target) = r.edge_endpoints(cycle[i]).unwrap();
            let bad = r.edge_weight(cycle[i]).unwrap().bad;
            if i % 2 == 0 {
                h1.add_edge(source, target, 1, bad);
            } else {
                h2.add_edge(source, target, 1, bad);
            }
        }
    }

    (h1, h2)
}

fn rbmg_find_perfect_matching(g: &RegularBipartiteMultiGraph) -> Matching {
    let k = g.degree;
    let n = g.l_nodes.len();
    let m = k * n;
    let t = m.next_power_of_two();
    let t2 = t.trailing_zeros();
    let alpha = t / k;
    let beta = t - k * alpha;

    // println!("rbmg_find_perfect_matching: k = {}, n = {}, m = {}, t = {}, t2 = {}", k, n, m, t, t2);

    // println!("printing g");
    // g.print();

    let mut h1: RegularBipartiteMultiGraph = RegularBipartiteMultiGraph::copy_without_edges(&g);
    h1.degree = t;

    for edge in g.graph.edge_references() {
        h1.add_edge(
            edge.source(),
            edge.target(),
            alpha * edge.weight().multiplicity,
            false,
        );
    }

    // Choose an arbitrary matching left[i] -- right[i]
    for i in 0..n {
        h1.add_edge(g.l_nodes[i], g.r_nodes[i], beta, true);
    }

    // println!("printing h1 initially");
    // println!("============");
    // h1.print();

    // Recursively split until we find a matching
    for _ in 0..t2 {
        let (h2, h3): (RegularBipartiteMultiGraph, RegularBipartiteMultiGraph) =
            rbmg_split_into_two(&h1);
        let bad_h2: usize = h2
            .graph
            .edge_references()
            .map(|e| e.weight().bad as usize)
            .sum();
        let bad_h3: usize = h3
            .graph
            .edge_references()
            .map(|e| e.weight().bad as usize)
            .sum();
        h1 = if bad_h2 <= bad_h3 { h2 } else { h3 };
        // println!("printing h1 after split");
        // println!("============");
        // h1.print();
    }

    let mut matching: Matching = Vec::with_capacity(n);

    for edge in h1.graph.edge_references() {
        matching.push((edge.source(), edge.target()));
    }

    // println!("rbmg_find_perfect_matching: matching = {:?}", matching);
    matching
}

fn rbmg_edge_color_when_power_of_two(g: &RegularBipartiteMultiGraph) -> Vec<Matching> {
    if !g.degree.is_power_of_two() {
        panic!("degree is not power of 2");
    }

    let mut coloring: Vec<Matching> = Vec::with_capacity(g.degree);

    if g.degree == 1 {
        // println!("degree is 1 ");
        let mut matching: Matching = Vec::with_capacity(g.l_nodes.len());
        for edge in g.graph.edge_references() {
            matching.push((edge.source(), edge.target()));
        }
        coloring.push(matching);
        return coloring;
    }

    let (h1, h2) = rbmg_split_into_two(&g);
    let mut h1_coloring = rbmg_edge_color_when_power_of_two(&h1);
    let mut h2_coloring = rbmg_edge_color_when_power_of_two(&h2);
    coloring.append(&mut h1_coloring);
    coloring.append(&mut h2_coloring);

    if coloring.len() != g.degree {
        panic!("wrong len of coloring!!");
    }

    coloring
}

fn rbmg_edge_color(g0: &RegularBipartiteMultiGraph) -> Vec<Matching> {
    // todo: rename to clone, and maybe don't even need to implement clone()
    let mut g: RegularBipartiteMultiGraph = RegularBipartiteMultiGraph::copy(g0);

    // println!("Want to color rbmg");
    // g.print();

    let mut coloring: Vec<Matching> = Vec::with_capacity(g.degree);

    if g.degree == 0 {
        // println!("degree is 1 ");
        return coloring;
    }

    if g.degree == 1 {
        // println!("degree is 1 ");
        let mut matching: Matching = Vec::with_capacity(g.l_nodes.len());
        for edge in g.graph.edge_references() {
            matching.push((edge.source(), edge.target()));
        }
        coloring.push(matching);
        return coloring;
    }

    let mut odd_degree_matching: Option<Matching> = None;

    if g.degree % 2 == 1 {
        // println!("degree is odd {:?}", g.degree);
        let matching = rbmg_find_perfect_matching(&g);
        // println!("Ok, for odd degree have matching {:?}", matching);
        g.remove_matching(&matching);
        odd_degree_matching = Some(matching);
    }

    // println!("and now degree is {:?}", g.degree);

    // Split graph into two regular bipartite multigraphs of half-degree
    let (mut h1, mut h2) = rbmg_split_into_two(&g);

    // Recursively color h1
    let h1_coloring = rbmg_edge_color(&h1);

    // Transfer some matchings from H1 to H2 to make the degree of H2 to be a power of 2
    let r = h2.degree.next_power_of_two();
    let num_classes_to_move = r - h2.degree;
    // println!("r = {:?}, num_to_move = {:?}", r, num_classes_to_move);

    for i in 0..num_classes_to_move {
        h2.add_matching(&h1_coloring[i]);
    }
    // println!("check: g0.degree = {:?}, g.degree = {:?}, h2 degree {:?}", g0.degree, g.degree, h2.degree);

    // Edge-color h2 by recursive splitting
    let mut h2_coloring = rbmg_edge_color_when_power_of_two(&h2);

    // Now, create the matching

    coloring = h2_coloring;

    // Add remaining colors from h1
    for i in num_classes_to_move..h1_coloring.len() {
        coloring.push(h1_coloring[i].clone());
    }

    if odd_degree_matching.is_some() {
        coloring.push(odd_degree_matching.unwrap());
    }

    if coloring.len() != g0.degree {
        panic!("Wrong output coloring length");
    }
    coloring
}

fn print_graph(r: &EdgedGraph) {
    let nodes: Vec<_> = r.node_indices().collect();
    let edges: Vec<_> = r.edge_references().collect();
    println!("===============");
    println!("nodes:");
    for node in nodes {
        println!("  {:?}", node);
    }
    println!("edges:");
    for edge in edges {
        println!("  {:?}", edge);
    }
    println!("=======");
}

fn print_original_graph(r: &Graph<(), (), Undirected>) {
    let nodes: Vec<_> = r.node_indices().collect();
    let edges: Vec<_> = r.edge_references().collect();
    println!("===============");
    println!("nodes:");
    for node in nodes {
        println!("  {:?}", node);
    }
    println!("edges:");
    for edge in edges {
        println!("  {:?}", edge);
    }
    println!("=======");
}

#[cfg(test)]
mod test_bipartite_coloring {
    use petgraph::data::Element::Node;
    // use crate::bipartite_coloring::greedy_edge_color;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;

    // use crate::bipartite_coloring::{bipartite_edge_color, RegularBipartiteMultiGraph};

    use crate::bipartite_coloring::EdgeData;
    use crate::bipartite_coloring::{
        bipartite_edge_color, euler_cycles, if_bipartite_edge_color, print_graph,
        print_original_graph, EdgedGraph, RegularBipartiteMultiGraph,
    };
    use crate::generators::petersen_graph;
    use crate::generators::random_bipartite_graph;
    use hashbrown::HashSet;
    use petgraph::graph::{edge_index, EdgeIndex};
    use petgraph::prelude::StableGraph;
    use petgraph::stable_graph::NodeIndex;
    use petgraph::visit::{
        EdgeRef, IntoEdgeReferences, IntoEdges, IntoNodeIdentifiers, NodeIndexable,
    };
    use petgraph::Undirected;
    use std::fmt::Debug;
    use std::hash::Hash;

    fn check_valid_bipatite_edge_coloring<G>(graph: G, colors: &DictMap<G::EdgeId, usize>)
    where
        G: NodeIndexable + IntoEdges + IntoNodeIdentifiers,
        G::EdgeId: Eq + Hash + Debug,
    {
        // Check that every edge has valid color
        for edge in graph.edge_references() {
            if !colors.contains_key(&edge.id()) {
                panic!("Problem: edge {:?} has no color assigned.", &edge.id());
            }
        }

        // Check that for every node all of its edges have different colors
        // (i.e. the number of used colors is equal to the degree).
        // Also compute number of colors used and maximum node degree.
        let mut num_colors = 0;
        let mut max_node_degree = 0;
        let node_indices: Vec<G::NodeId> = graph.node_identifiers().collect();
        for node in node_indices {
            let mut cur_node_degree = 0;
            let mut used_colors: HashSet<usize> = HashSet::new();
            for edge in graph.edges(node) {
                let color = colors.get(&edge.id()).unwrap();
                used_colors.insert(*color);
                cur_node_degree += 1;
                if num_colors < *color + 1 {
                    num_colors = *color + 1;
                }
            }
            if used_colors.len() < cur_node_degree {
                panic!(
                    "Problem: node {:?} does not have enough colors.",
                    NodeIndexable::to_index(&graph, node)
                );
            }

            if cur_node_degree > max_node_degree {
                max_node_degree = cur_node_degree
            }
        }

        println!(
            "=> checking: max_degree = {:?}, num_colors = {:?}",
            max_node_degree, num_colors
        );
        // Check that number of colors used is at most max_node_degree + 1
        // (note that number of colors is max_color + 1).
        if max_node_degree < num_colors {
            panic!(
                "Problem: too many colors are used ({} colors used, {} max node degree)",
                num_colors, max_node_degree
            );
        }
    }

    #[test]
    fn test_bipartite_multiple_edges() {
        println!("Running test5");
        let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 1)];
        let graph = Graph::<(), (), Undirected>::from_edges(&edge_list);
        print_original_graph(&graph);

        let output = if_bipartite_edge_color(&graph);
        println!("FINAL FINAL {:?}", output);
    }

    #[test]
    fn test_bipartite_petersen_graphs() {
        for i in 3..30 {
            for j in 0..30 {
                let n = 2 * i;
                let k = 2 * j + 1;
                if n <= 2 * k {
                    continue;
                }
                let graph: petgraph::graph::UnGraph<(), ()> =
                    petersen_graph(n, k, || (), || ()).unwrap();
                match if_bipartite_edge_color(&graph) {
                    Ok(edge_coloring) => {
                        check_valid_bipatite_edge_coloring(&graph, &edge_coloring);
                    }
                    Err(_) => panic!("This should error"),
                }
            }
        }
    }

    #[test]
    fn test_not_bipartite_petersen_graphs() {
        for i in 3..30 {
            for j in 1..31 {
                let n = 2 * i;
                let k = 2 * j;
                if n > 2 * k {
                    let graph: petgraph::graph::UnGraph<(), ()> =
                        petersen_graph(n, k, || (), || ()).unwrap();
                    match if_bipartite_edge_color(&graph) {
                        Ok(edge_coloring) => panic!("This should error"),
                        Err(_) => (),
                    }
                }
            }
        }
    }

    #[test]
    fn test_bipartite_random_graphs() {
        for num_l_nodes in vec![5, 10, 15, 20] {
            for num_r_nodes in vec![5, 10, 15, 20] {
                for probability in vec![0.1, 0.3, 0.5, 0.7, 0.9] {
                    let graph: petgraph::graph::UnGraph<(), ()> = random_bipartite_graph(
                        num_l_nodes,
                        num_r_nodes,
                        probability,
                        Some(10),
                        || (),
                        || (),
                    )
                    .unwrap();
                    match if_bipartite_edge_color(&graph) {
                        Ok(edge_coloring) => {
                            check_valid_bipatite_edge_coloring(&graph, &edge_coloring);
                        }
                        Err(_) => panic!("This should error"),
                    }
                }
            }
        }
    }

    #[test]
    fn test_empty_graph() {
        let mut graph = Graph::<(), (), Undirected>::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let left = vec![a, b];
        let right = vec![c, d];
        let output = bipartite_edge_color(&graph, &left, &right);
        print_original_graph(&graph);
        println!("FINAL FINAL {:?}", output);
    }

    #[test]
    fn test2() {
        let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 1)];
        let original_graph = Graph::<(), (), Undirected>::from_edges(&edge_list);
        let left = vec![
            NodeIndex::new(0),
            NodeIndex::new(4),
            NodeIndex::new(5),
            NodeIndex::new(6),
        ];
        let right = vec![NodeIndex::new(1), NodeIndex::new(2), NodeIndex::new(3)];

        let output = bipartite_edge_color(&original_graph, &left, &right);
        // print_original_graph(&original_graph);
        println!("FINAL FINAL {:?}", output);
    }
}
