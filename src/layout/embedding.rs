use hashbrown::hash_map::HashMap;
use hashbrown::HashSet;
use petgraph::graph::Edge;
use petgraph::graph::Graph;
use petgraph::prelude::*;
use petgraph::visit::{GraphBase, NodeIndexable};
use petgraph::Directed;
use rayon::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

use crate::StablePyGraph;
use retworkx_core::connectivity::connected_components;
use retworkx_core::planar::lr_planar::{LRState, Sign};

pub type Point = [f64; 2];

#[derive(Debug)]
pub struct CwCcw<T> {
    cw: Option<T>,
    ccw: Option<T>,
}

impl<T> Default for CwCcw<T> {
    fn default() -> Self {
        CwCcw {
            cw: None,
            ccw: None,
        }
    }
}

impl<T> CwCcw<T> {
    fn new(cw: T, ccw: T) -> Self {
        CwCcw {
            cw: Some(cw),
            ccw: Some(ccw),
        }
    }
}

#[derive(Debug)]
pub struct FirstNbr<T> {
    first_nbr: Option<T>,
}

impl<T> Default for FirstNbr<T> {
    fn default() -> Self {
        FirstNbr { first_nbr: None }
    }
}

impl<T> FirstNbr<T> {
    fn new(first_nbr: T) -> Self {
        FirstNbr {
            first_nbr: Some(first_nbr),
        }
    }
}

pub struct PlanarEmbedding {
    pub embedding: Graph<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>,
}

impl Default for PlanarEmbedding {
    fn default() -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>::new(),
        }
    }
}
impl PlanarEmbedding {
    pub fn new() -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>::new(),
        }
    }

    fn neighbors_cw_order(&mut self, v: NodeIndex) -> Vec<NodeIndex> {
        let mut nbrs: Vec<NodeIndex> = vec![];
        let first_nbr = self.embedding[v].first_nbr;
        println!("first_nbr in nbrs_cw_order {:?}", first_nbr);
        if first_nbr.is_none() {
            // v has no neighbors
            return nbrs;
        }
        let start_node = first_nbr.unwrap();
        println!("start_node in nbrs_cw_order {:?}", start_node);
        nbrs.push(start_node);

        let mut node = self.get_edge_weight(v, start_node, true);
        if let Some(mut current_node) = node {
            println!("current_node 1 {:?}", current_node);
            let mut count = 0;
            while start_node != current_node && count < 20 {
                count += 1;
                nbrs.push(current_node);
                node = self.get_edge_weight(v, current_node, true);
                current_node = match node {
                    Some(node) => node,
                    None => break,
                };
                println!("current_node 2 {:?}", current_node);
            }
        }
        println!("nbrs 1 in nbrs_cw_order {:?}", nbrs);

        nbrs
    }

    fn add_half_edge_cw(
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_nbr: Option<NodeIndex>,
    ) {
        let cw_weight = CwCcw::<NodeIndex>::default();
        self.embedding.add_edge(start_node, end_node, cw_weight);
        if ref_nbr.is_none() {
            // The start node has no neighbors
            self.update_edge_weight(start_node, end_node, end_node, true);
            self.update_edge_weight(start_node, end_node, end_node, false);
            self.embedding[start_node].first_nbr = Some(end_node);
            return;
        }
        // if ref_nbr not in self[start_node] error
        let ref_nbr_node = ref_nbr.unwrap();
        // DEBUG
        if self.embedding.find_edge(start_node, ref_nbr_node).is_none() {
            println!("NO REF NBR in ADD CW {:?} {:?}", start_node, ref_nbr_node);
        }
        let cw_ref = self.get_edge_weight(start_node, ref_nbr_node, true);
        if let Some(cw_ref_node) = cw_ref {
            // Alter half-edge data structures
            self.update_edge_weight(start_node, ref_nbr_node, end_node, true);
            self.update_edge_weight(start_node, end_node, cw_ref_node, true);
            self.update_edge_weight(start_node, cw_ref_node, end_node, false);
            self.update_edge_weight(start_node, end_node, ref_nbr_node, false);
        }
    }

    fn add_half_edge_ccw(
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_nbr: Option<NodeIndex>,
    ) {
        if ref_nbr.is_none() {
            // Start node has no neighbors
            let cw_weight = CwCcw::<NodeIndex>::default();
            self.embedding.add_edge(start_node, end_node, cw_weight);
            self.update_edge_weight(start_node, end_node, end_node, true);
            self.update_edge_weight(start_node, end_node, end_node, false);
            self.embedding[start_node].first_nbr = Some(end_node);
        } else {
            let ref_nbr_node = ref_nbr.unwrap();
            let ccw_ref_node = self.get_edge_weight(start_node, ref_nbr_node, false);
            self.add_half_edge_cw(start_node, end_node, ccw_ref_node);
            if ref_nbr == self.embedding[start_node].first_nbr {
                // Update first neighbor
                self.embedding[start_node].first_nbr = Some(end_node);
            }
        }
    }

    fn add_half_edge_first(&mut self, start_node: NodeIndex, end_node: NodeIndex) {
        let ref_node: Option<NodeIndex> = if self.embedding.node_bound() >= start_node.index()
            && self.embedding[start_node].first_nbr.is_some()
        {
            self.embedding[start_node].first_nbr
        } else {
            None
        };
        self.add_half_edge_ccw(start_node, end_node, ref_node);
    }

    fn next_face_half_edge(&self, v: NodeIndex, w: NodeIndex) -> (NodeIndex, NodeIndex) {
        //println!("next v {:?} w {:?}", v, w);
        let new_node = self.get_edge_weight(w, v, false);
        //println!("new node {:?}", new_node);
        // FIX THIS
        //
        //
        if new_node.is_none() {
            println!("NEW NODE NONE in next_face {:?} {:?} {:?}", new_node, v, w);
            panic!("HELP!"); //return (w, v);
        }
        (w, new_node.unwrap())
    }

    fn update_edge_weight(&mut self, v: NodeIndex, w: NodeIndex, new_node: NodeIndex, cw: bool) {
        let found_edge = self.embedding.find_edge(v, w);
        if found_edge.is_none() {
            return;
        }
        let found_weight = self.embedding.edge_weight_mut(found_edge.unwrap()); //.unwrap();
        if found_weight.is_none() {
            return;
        }

        if cw {
            found_weight.unwrap().cw = Some(new_node);
        } else {
            found_weight.unwrap().ccw = Some(new_node);
        }
    }

    fn get_edge_weight(&self, v: NodeIndex, w: NodeIndex, cw: bool) -> Option<NodeIndex> {
        //println!("GET EDGE v w {:?} {:?}", v, w);
        let found_edge = self.embedding.find_edge(v, w);
        // FIX THIS
        //
        //
        if found_edge.is_none() {
            println!("GET EDGE find edge is none {:?}", found_edge);
            return None;
        }
        //println!("GET EDGE find edge{:?}", found_edge);
        let found_weight = self.embedding.edge_weight(found_edge.unwrap()); //.unwrap();
                                                                            //println!("after found weight {:?}", found_weight);
        if found_weight.is_none() {
            println!("GET EDGE Weight is none {:?}", found_weight);
            return None;
            // } else {
            //     let found_weight = found_weight.unwrap();
        }

        if cw {
            found_weight.unwrap().cw
        } else {
            found_weight.unwrap().ccw
        }
    }

    fn connect_components(&mut self, v: NodeIndex, w: NodeIndex) {
        self.add_half_edge_first(v, w);
        self.add_half_edge_first(w, v);
    }
}

pub fn create_embedding(
    planar_emb: &mut PlanarEmbedding,
    lr_state: &mut LRState<&StablePyGraph<Undirected>>,
) {
    // ********** DEBUG
    for node in lr_state.dir_graph.node_indices() {
        for edge in lr_state.dir_graph.edges(node) {
            println!("Edge {:?}, {:?}", edge.source(), edge.target());
        }
    }
    // ********** DEBUG END

    let mut ordered_adjs: Vec<Vec<NodeIndex>> = Vec::new();

    let mut nesting_depth: HashMap<(NodeIndex, NodeIndex), i32> =
        HashMap::with_capacity(lr_state.nesting_depth.len());

    // Create the adjacency list for each node
    for v in lr_state.dir_graph.node_indices() {
        ordered_adjs.push(lr_state.dir_graph.edges(v).map(|e| e.target()).collect());

        // Add empty FirstNbr to the embedding
        let first_nbr = FirstNbr::<NodeIndex>::default();
        planar_emb.embedding.add_node(first_nbr);

        // Change the sign for nesting_depth
        for e in lr_state.dir_graph.edges(v) {
            let edge: (NodeIndex, NodeIndex) = (e.source(), e.target());
            let signed_depth: i32 = lr_state.nesting_depth[&edge] as i32;
            let signed_side = if lr_state.side.contains_key(&edge)
                && sign(edge, &mut lr_state.eref, &mut lr_state.side) == Sign::Minus
            {
                -1
            } else {
                1
            };
            nesting_depth.insert(edge, signed_depth * signed_side);
        }
    }
    // ********** DEBUG
    for x in &ordered_adjs {
        println!("ordered {:?}", x);
    }
    // ********** DEBUG END

    // ********** DEBUG
    for x in &nesting_depth {
        println!("nesting {:?}", x);
    }
    // ********** DEBUG END

    // Sort the adjacency list using nesting depth as sort order
    for (v, adjs) in ordered_adjs.iter_mut().enumerate() {
        adjs.par_sort_by_key(|x| nesting_depth[&(NodeIndex::new(v), *x)]);
    }

    // ********** DEBUG
    for x in &ordered_adjs {
        println!("ordered 2222 {:?}", x);
    }
    // ********** DEBUG END

    // Add the initial half edge cw to the embedding using the ordered adjacency list
    for v in lr_state.dir_graph.node_indices() {
        let mut prev_node: Option<NodeIndex> = None;
        for w in &ordered_adjs[v.index()] {
            planar_emb.add_half_edge_cw(v, *w, prev_node);
            prev_node = Some(*w)
        }
    }

    // ********** DEBUG
    for v in planar_emb.embedding.node_indices() {
        println!("11111 embedding node {:?} {:?}", v, planar_emb.embedding[v]);
    }
    println!("lr eparent {:?}", lr_state.eparent);
    // ********** DEBUG END

    // Start the DFS traversal for the embedding
    let mut left_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut right_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut idx: Vec<usize> = vec![0; ordered_adjs.len()];

    for v in lr_state.roots.iter() {
        // Create the stack with an initial entry of v
        let mut dfs_stack: Vec<NodeIndex> = vec![*v];

        while dfs_stack.len() > 0 {
            let v = dfs_stack.pop().unwrap();
            let idx2 = idx[v.index()];

            // Iterate over the ordered_adjs starting at the saved index until the end
            for w in ordered_adjs[v.index()][idx2..].iter() {
                idx[v.index()] += 1;

                let ei = (v, *w);
                if lr_state.eparent.contains_key(&w) && ei == lr_state.eparent[&w] {
                    println!("in ei {:?}", ei);
                    planar_emb.add_half_edge_first(*w, v);
                    left_ref.entry(v).or_insert(*w);
                    right_ref.entry(v).or_insert(*w);
                    dfs_stack.push(v);
                    dfs_stack.push(*w);
                    break;
                } else {
                    if !lr_state.side.contains_key(&ei) || lr_state.side[&ei] == Sign::Plus {
                        planar_emb.add_half_edge_cw(*w, v, Some(right_ref[w]));
                    } else {
                        planar_emb.add_half_edge_ccw(*w, v, Some(left_ref[w]));
                        left_ref.entry(*w).or_insert(v);
                    }
                    println!("in else {:?}", ei);
                }
            }
        }
    }

    // ********** DEBUG
    for v in planar_emb.embedding.node_indices() {
        for w in &ordered_adjs[v.index()] {
            println!(
                "22222 embedding node {:?} {:?} {:?} {:?}",
                v,
                planar_emb.embedding[v],
                planar_emb.get_edge_weight(v, *w, true),
                planar_emb.get_edge_weight(v, *w, false)
            );
        }
    }
    // ********** DEBUG END

    pub fn sign(
        edge: (NodeIndex, NodeIndex),
        eref: &mut HashMap<(NodeIndex, NodeIndex), (NodeIndex, NodeIndex)>,
        side: &mut HashMap<(NodeIndex, NodeIndex), Sign>,
    ) -> Sign {
        // Resolve the relative side of an edge to the absolute side.

        if eref.contains_key(&edge) {
            if side[&edge].clone() == sign(eref[&edge].clone(), eref, side) {
                *side.get_mut(&edge).unwrap() = Sign::Plus;
            } else {
                *side.get_mut(&edge).unwrap() = Sign::Minus;
            }
            eref.remove(&edge);
        }
        side[&edge]
    }
}

pub fn embedding_to_pos(planar_emb: &mut PlanarEmbedding) -> Vec<Point> {
    let mut pos: Vec<Point> = Vec::with_capacity(planar_emb.embedding.node_count());
    if planar_emb.embedding.node_count() < 4 {
        let default_pos = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]].to_vec();
        return planar_emb
            .embedding
            .node_indices()
            .map(|n| default_pos[n.index()])
            .collect();
    }
    let outer_face = triangulate_embedding(planar_emb, true);

    let right_t_child = HashMap::<NodeIndex, usize>::new();
    let left_t_child = HashMap::<NodeIndex, usize>::new();
    let delta_x = HashMap::<NodeIndex, usize>::new();
    let y_coord = HashMap::<NodeIndex, usize>::new();

    let node_list = canonical_ordering(planar_emb, outer_face);

    pos
}

fn triangulate_embedding(
    planar_emb: &mut PlanarEmbedding,
    fully_triangulate: bool,
) -> Vec<NodeIndex> {
    let component_sets = connected_components(&planar_emb.embedding);
    println!("CONN COMP {:?}", component_sets);

    for i in 0..(component_sets.len() - 1) {
        let v1 = component_sets[i].iter().next().unwrap();
        let v2 = component_sets[i + 1].iter().next().unwrap();
        planar_emb.connect_components(*v1, *v2);
        println!("v1 v2 {:?} {:?}", *v1, *v2);
    }
    let mut outer_face = vec![];
    let mut face_list = vec![];
    let mut edges_counted: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
    println!(" in triangulate component sets{:?}", component_sets);

    for v in planar_emb.embedding.node_indices() {
        for w in planar_emb.neighbors_cw_order(v) {
            let new_face = make_bi_connected(planar_emb, v, w, &mut edges_counted);
            if new_face.len() > 0 {
                face_list.push(new_face.clone());
                if new_face.len() > outer_face.len() {
                    outer_face = new_face;
                }
            }
        }
    }

    println!("FINAL Face list {:?}", face_list);
    for face in face_list {
        println!("face {:?} outer_face {:?}", face, outer_face);
        if face != outer_face || fully_triangulate {
            triangulate_face(planar_emb, face[0], face[1]);
        }
    }
    if fully_triangulate {
        let v1 = outer_face[0];
        let v2 = outer_face[1];
        let v3 = planar_emb.get_edge_weight(v2, v1, false);
        outer_face = vec![v1, v2, v3.unwrap()];
    }
    println!("outer_face {:?}", outer_face);
    outer_face
}

fn make_bi_connected(
    planar_emb: &mut PlanarEmbedding,
    start_node: NodeIndex,
    out_node: NodeIndex,
    edges_counted: &mut HashSet<(NodeIndex, NodeIndex)>,
) -> Vec<NodeIndex> {
    if edges_counted.contains(&(start_node, out_node)) || start_node == out_node {
        println!(
            "biconnect already in start out {:?} {:?}",
            start_node, out_node
        );
        return vec![];
    }
    edges_counted.insert((start_node, out_node));
    //println!("edges counted {:?} {:?} {:?}", start_node, out_node, edges_counted);
    let mut v1 = start_node.clone();
    let mut v2 = out_node.clone();
    let mut face_list: Vec<NodeIndex> = vec![start_node];
    let (_, mut v3) = planar_emb.next_face_half_edge(v1, v2);

    let mut count = 0;
    while (v2 != start_node || v3 != out_node) && count < 20 {
        // && count < 300 {
        if v1 == v2 {
            println!("BICONNECT V1==V2 should raise");
        }
        println!("face_list in while {:?} {:?} {:?}", v2, v3, face_list);

        if face_list.contains(&v2) {
            println!(
                "face_list contains v2 {:?} {:?} {:?}",
                v2, v3, edges_counted
            );
            planar_emb.add_half_edge_cw(v1, v3, Some(v2));
            planar_emb.add_half_edge_ccw(v3, v1, Some(v2));
            edges_counted.insert((v2, v3));
            edges_counted.insert((v3, v1));
            v2 = v1.clone();
        } else {
            println!(
                "face_list not contain v2 {:?} {:?} {:?}",
                v2, v3, edges_counted
            );
            face_list.push(v2);
        }
        v1 = v2.clone();
        println!("2 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);
        (v2, v3) = planar_emb.next_face_half_edge(v2, v3);
        println!("3 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);

        edges_counted.insert((v1, v2));
        count += 1;
    }
    face_list
}

fn triangulate_face(planar_emb: &mut PlanarEmbedding, mut v1: NodeIndex, mut v2: NodeIndex) {
    let (_, mut v3) = planar_emb.next_face_half_edge(v1, v2);
    let (_, mut v4) = planar_emb.next_face_half_edge(v2, v3);
    if v1 == v2 || v1 == v3 {
        return;
    }
    let mut count = 0;
    while v1 != v4 && count < 20 {
        if planar_emb.embedding.contains_edge(v1, v3) {
            (v1, v2, v3) = (v2, v3, v4);
        } else {
            planar_emb.add_half_edge_cw(v1, v3, Some(v2));
            planar_emb.add_half_edge_ccw(v3, v1, Some(v2));
            (v1, v2, v3) = (v1, v3, v4);
        }
        (_, v4) = planar_emb.next_face_half_edge(v2, v3);
        count += 1;
    }
}

fn canonical_ordering(
    planar_emb: &mut PlanarEmbedding,
    outer_face: Vec<NodeIndex>,
) -> Vec<(NodeIndex, Vec<NodeIndex>)> {
    let v1 = outer_face[0];
    let v2 = outer_face[1];
    let mut chords: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(planar_emb.embedding.node_count());
    let mut marked_nodes: HashSet<NodeIndex> =
        HashSet::with_capacity(planar_emb.embedding.node_count());
    let mut ready_to_pick: HashSet<NodeIndex> = HashSet::with_capacity(outer_face.len());

    let mut outer_face_cw_nbr: HashMap<NodeIndex, NodeIndex> =
        HashMap::with_capacity(outer_face.len());
    let mut outer_face_ccw_nbr: HashMap<NodeIndex, NodeIndex> =
        HashMap::with_capacity(outer_face.len());
    let mut prev_nbr = v2.clone();

    for v in outer_face.iter().rev() {
        outer_face_ccw_nbr.entry(prev_nbr).or_insert(*v);
        prev_nbr = *v;
    }
    outer_face_ccw_nbr.entry(prev_nbr).or_insert(v1);

    for v in outer_face.iter().rev() {
        outer_face_cw_nbr.entry(prev_nbr).or_insert(*v);
        prev_nbr = *v;
    }

    fn is_outer_face_nbr(
        x: NodeIndex,
        y: NodeIndex,
        outer_face_cw_nbr: &HashMap<NodeIndex, NodeIndex>,
        outer_face_ccw_nbr: &HashMap<NodeIndex, NodeIndex>,
    ) -> bool {
        if !outer_face_ccw_nbr.contains_key(&x) {
            return outer_face_cw_nbr[&x] == y;
        }
        if !outer_face_ccw_nbr.contains_key(&x) {
            return outer_face_cw_nbr[&x] == y;
        }
        outer_face_cw_nbr[&x] == y || outer_face_ccw_nbr[&x] == y
    }

    fn is_on_outer_face(
        x: NodeIndex,
        v1: NodeIndex,
        marked_nodes: &HashSet<NodeIndex>,
        outer_face_ccw_nbr: &HashMap<NodeIndex, NodeIndex>,
    ) -> bool {
        !marked_nodes.contains(&x) && (outer_face_ccw_nbr.contains_key(&x) || x == v1)
    }

    for v in outer_face {
        for nbr in planar_emb.neighbors_cw_order(v) {
            if is_on_outer_face(nbr, v1, &marked_nodes, &outer_face_ccw_nbr)
                && is_outer_face_nbr(v, nbr, &outer_face_cw_nbr, &outer_face_ccw_nbr)
            {
                if chords.contains_key(&v) {
                    let chords_plus = chords[&v].clone();
                    chords.entry(v).or_insert(chords_plus + 1);
                } else {
                    chords.entry(v).or_insert(1);
                }
                ready_to_pick.remove(&v);
            }
        }
    }
    vec![(
        NodeIndex::new(0),
        vec![NodeIndex::new(1), NodeIndex::new(2)],
    )]
}
