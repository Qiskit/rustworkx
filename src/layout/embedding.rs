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

use retworkx_core::connectivity::connected_components;
use crate::StablePyGraph;
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
            //println!("current_node 1 {:?}", current_node);

            while start_node != current_node {
                nbrs.push(current_node);
                node = self.get_edge_weight(v, current_node, true);
                current_node = match node {
                    Some(node) => node,
                    None => break,
                };
                //println!("current_node 2 {:?}", current_node);
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
            return (w, v);
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
        println!("GET EDGE v w {:?} {:?}", v, w);
        let found_edge = self.embedding.find_edge(v, w);
        // FIX THIS
        //
        //
        if found_edge.is_none() {
            return None;
        }
        println!("GET EDGE find edge{:?}", found_edge);
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

fn id_to_index<G: GraphBase + NodeIndexable>(graph: G, node_id: G::NodeId) -> NodeIndex {
    NodeIndex::new(graph.to_index(node_id))
}

fn index_to_id<G: GraphBase + NodeIndexable>(graph: G, node_index: NodeIndex) -> G::NodeId {
    graph.from_index(node_index.index())
}

pub fn create_embedding(
    planar_emb: &mut PlanarEmbedding,
    lr_state: &mut LRState<&StablePyGraph<Undirected>>,
)
// where
//    <G as GraphBase>::NodeId: Hash + Eq,
//    <G as GraphBase>::NodeId: Debug,
//    // These are needed for par_sort
//    G: std::marker::Sync,
//    <G as GraphBase>::NodeId: Sync,
{
    for node in lr_state.dir_graph.node_indices() {
        for edge in lr_state.dir_graph.edges(node) {
            println!("Edge {:?}, {:?}", edge.source(), edge.target());
        }
    }

    let mut ordered_adjs: Vec<Vec<NodeIndex>> = Vec::new();

    for v in lr_state.dir_graph.node_indices() {
        ordered_adjs.push(lr_state.dir_graph.edges(v).map(|e| e.target()).collect());

        let first_nbr = FirstNbr::<NodeIndex>::default();
        planar_emb.embedding.add_node(first_nbr);
        let mut nesting_depth: HashMap<(NodeIndex, NodeIndex), i32> =
            HashMap::with_capacity(lr_state.nesting_depth.len());
        // Change the sign for nesting_depth
        for e in lr_state.dir_graph.edges(v) {
            let edge: (NodeIndex, NodeIndex) = (e.source(), e.target());
            //let edge_base = (index_to_id(&lr_state.graph, edge.source()), index_to_id(&lr_state.graph, edge.target()));
            let signed_depth: i32 = lr_state.nesting_depth[&edge] as i32;
            nesting_depth.insert(edge,
                sign(edge, &mut lr_state.eref, &mut lr_state.side) * signed_depth);
            //lr_state.nesting_depth[&edge];
        }
    }
    for x in &ordered_adjs {
        println!("ordered {:?}", x);
    }
    for x in &lr_state.nesting_depth {
        println!("nesting {:?}", x);
    }

    //lr_state.nesting_depth.iter().enumerate().map(|(e, val)| (e, val * sign(e)));

    for (v, adjs) in ordered_adjs.iter_mut().enumerate() {
        adjs.par_sort_by_key(|x| {
            lr_state.nesting_depth[&(
                index_to_id(&lr_state.graph, NodeIndex::new(v)),
                index_to_id(&lr_state.graph, *x),
            )]
        });
    }
    for x in &ordered_adjs {
        println!("ordered 2222 {:?}", x);
    }

    for v in lr_state.dir_graph.node_indices() {
        let mut prev_node: Option<NodeIndex> = None;
        for w in &ordered_adjs[v.index()] {
            planar_emb.add_half_edge_cw(v, *w, prev_node);
            prev_node = Some(*w)
        }
    }
    for v in planar_emb.embedding.node_indices() {
        println!("11111 embedding node {:?} {:?}", v, planar_emb.embedding[v]);
    }

    let mut left_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut right_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut idx: Vec<usize> = vec![0; ordered_adjs.len()];

    for v_id in lr_state.roots.iter() {
        let v = id_to_index(&lr_state.graph, *v_id);
        // println!(
        //     "second v {:?} v index {:?} ord {:?} idx {:?}",
        //     v,
        //     v.index(),
        //     ordered_adjs[v.index()],
        //     idx
        // );

        let mut dfs_stack: Vec<NodeIndex> = vec![v];

        // println!("lr eparent {:?}", lr_state.eparent);
        while dfs_stack.len() > 0 {
            let v = dfs_stack.pop().unwrap();
            let idx2 = idx[v.index()];
            for (w_pos, w) in ordered_adjs[v.index()][idx2..].iter().enumerate() {
                let w_id = index_to_id(&lr_state.graph, *w);
                // println!(
                //     "third v {:?} vindex {:?} w {:?} w_id {:?} w_pos {:?} idx {:?} ",
                //     v,
                //     v.index(),
                //     *w,
                //     w_id,
                //     w_pos,
                //     idx
                // );
                idx[v.index()] += 1;

                let ei = (v, w);
                let ei_id = (
                    index_to_id(&lr_state.graph, v),
                    index_to_id(&lr_state.graph, *w),
                );
                if lr_state.eparent.contains_key(&w_id) {
                    let parent_id = lr_state.eparent[&w_id];
                    let (v1, v2) = (
                        id_to_index(&lr_state.graph, parent_id.0),
                        id_to_index(&lr_state.graph, parent_id.1),
                    );

                    if ei == (v1, &v2) {
                        // println!("in ei {:?}", ei);
                        planar_emb.add_half_edge_first(*w, v);
                        left_ref.entry(v).or_insert(*w);
                        right_ref.entry(v).or_insert(*w);
                        dfs_stack.push(v);
                        dfs_stack.push(*w);
                        break;
                    } else {
                        if lr_state.side[&ei_id] == Sign::Plus {
                            planar_emb.add_half_edge_cw(*w, v, Some(right_ref[w]));
                        } else {
                            planar_emb.add_half_edge_ccw(*w, v, Some(left_ref[w]));
                            left_ref.entry(*w).or_insert(v);
                        }
                        // println!("in else {:?}", ei);
                    }
                }
            }
        }
    }

    for v in planar_emb.embedding.node_indices() {
        println!("22222 embedding node {:?} {:?}", v, planar_emb.embedding[v]);
    }
    pub fn sign(
        edge: (NodeIndex, NodeIndex),
        eref: &mut HashMap<(NodeIndex, NodeIndex), (NodeIndex, NodeIndex)>,
        side: &mut HashMap<(NodeIndex, NodeIndex), Sign>,
    ) -> i32 {
        let mut dfs_stack: Vec<(NodeIndex, NodeIndex)> = vec![edge];
        let mut old_ref: HashMap<(NodeIndex, NodeIndex), (NodeIndex, NodeIndex)> =
            HashMap::with_capacity(eref.len());

        // """Resolve the relative side of an edge to the absolute side."""

        let mut e: (NodeIndex, NodeIndex) = edge;
        let mut side_final: i32 = 1;
        // for e in side {
        //     println!("side {} {}", e.0.0.index(), e.0.1.index());
        // }
        //return side_final;
        while dfs_stack.len() > 0 {
            e = dfs_stack.pop().unwrap();

            if eref.contains_key(&e) {
                dfs_stack.push(e);
                dfs_stack.push(eref[&e]);
                *old_ref.get_mut(&e).unwrap() = eref[&e];
                eref.remove(&e);
            } else {
                if side.contains_key(&e) {
                if side[&e] == side[&old_ref[&e]] {
                        *side.get_mut(&e).unwrap() = Sign::Plus;
                    } else {
                        println!("side no find plus {} {}", e.0.index(), e.1.index());
                    }
                    side_final = 1;
                } else {
                    if side.contains_key(&e) {
                        *side.get_mut(&e).unwrap() = Sign::Minus;
                    } else {
                        println!("side no find minus {} {}", e.0.index(), e.1.index());
                    }
                    side_final = -1;
                }
            }
        }
        side_final
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
    let outer_face = triangulate_embedding(planar_emb);

    let right_t_child = HashMap::<NodeIndex, usize>::new();
    let left_t_child = HashMap::<NodeIndex, usize>::new();
    let delta_x = HashMap::<NodeIndex, usize>::new();
    let y_coord = HashMap::<NodeIndex, usize>::new();

    let node_list = canonical_ordering(planar_emb, outer_face);

    pos
}

fn triangulate_embedding(planar_emb: &mut PlanarEmbedding) -> Vec<NodeIndex> {
    let component_sets = connected_components(&planar_emb.embedding);
    println!("CONN COMP {:?}", component_sets);

    for i in 0..(component_sets.len() - 1) {
        let v1 = component_sets[i].iter().next().unwrap();
        let v2 = component_sets[i + 1].iter().next().unwrap();
        planar_emb.connect_components(*v1, *v2);
    }
    let mut outer_face = vec![];
    let mut face_list = vec![];
    let mut edges_counted: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
    println!(" in triangulate component sets{:?}", component_sets);
    return outer_face;

    for v in planar_emb.embedding.node_indices() {
        for w in planar_emb.neighbors_cw_order(v) {
            let new_face = make_bi_connected(planar_emb, v, w, &mut edges_counted);
            let new_len = new_face.len();
            if new_len > 0 {
                face_list.push(new_face.clone());
                if new_len > outer_face.len() {
                    outer_face = new_face;
                }
            }
        }
    }
    return outer_face;

    println!("FINAL Face list {:?}", face_list);
    for face in face_list {
        println!("face {:?} outer_face {:?}", face, outer_face);
        if face != outer_face {
            triangulate_face(planar_emb, face[0], face[1]);
        }
    }
    println!("outer_face {:?}", outer_face);
    outer_face
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

fn make_bi_connected(
    planar_emb: &mut PlanarEmbedding,
    start_node: NodeIndex,
    out_node: NodeIndex,
    edges_counted: &mut HashSet<(NodeIndex, NodeIndex)>,
) -> Vec<NodeIndex> {
    if edges_counted.contains(&(start_node, out_node)) || start_node == out_node {
        return vec![];
    }
    edges_counted.insert((start_node, out_node));
    //println!("edges counted {:?} {:?} {:?}", start_node, out_node, edges_counted);
    let mut v1 = start_node.clone();
    let mut v2 = out_node.clone();
    let mut face_list: Vec<NodeIndex> = vec![start_node];
    let (_, mut v3) = planar_emb.next_face_half_edge(v1, v2);

    let mut count = 0;
    while (v2 != start_node || v3 != out_node) && v2 != v3 && count < 20 {
        // && count < 300 {
        //println!("face_list in while {:?} {:?} {:?}", v2, v3, face_list);

        if face_list.contains(&v2) {
            //println!("face_list contains v2 {:?} {:?} {:?}", v2, v3, edges_counted);
            planar_emb.add_half_edge_cw(v1, v3, Some(v2));
            planar_emb.add_half_edge_ccw(v3, v1, Some(v2));
            edges_counted.insert((v2, v3));
            edges_counted.insert((v3, v1));
            v2 = v1.clone();
        } else {
            //println!("face_list not contain v2 {:?} {:?} {:?}", v2, v3, edges_counted);
            face_list.push(v2);
        }
        v1 = v2.clone();
        //println!("2 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);
        (v2, v3) = planar_emb.next_face_half_edge(v2, v3);
        //println!("3 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);

        edges_counted.insert((v1, v2));
        count += 1;
    }
    face_list
}

fn canonical_ordering(
    planar_emb: &mut PlanarEmbedding,
    outer_face: Vec<NodeIndex>,
) -> Vec<(NodeIndex, Vec<NodeIndex>)> {
    vec![(
        NodeIndex::new(0),
        vec![NodeIndex::new(1), NodeIndex::new(2)],
    )]
}
