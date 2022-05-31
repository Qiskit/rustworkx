use hashbrown::hash_map::HashMap;
use petgraph::graph::Edge;
use petgraph::graph::Graph;
use petgraph::prelude::*;
use petgraph::visit::{GraphBase, NodeIndexable};
use petgraph::Directed;
use rayon::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Mul;

use crate::planar::lr_planar::{LRState, Sign};

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
        let cw_ref_node = self.get_edge_weight(start_node, ref_nbr_node, true);
        // Alter half-edge data structures
        self.update_edge_weight(start_node, ref_nbr_node, end_node, true);
        self.update_edge_weight(start_node, end_node, cw_ref_node, true);
        self.update_edge_weight(start_node, cw_ref_node, end_node, false);
        self.update_edge_weight(start_node, end_node, ref_nbr_node, false);
    }

    fn add_half_edge_ccw(
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_nbr: Option<NodeIndex>,
    ) {
        if ref_nbr.is_none() {
            let cw_weight = CwCcw::<NodeIndex>::default();
            self.embedding.add_edge(start_node, end_node, cw_weight);
            self.update_edge_weight(start_node, end_node, end_node, true);
            self.update_edge_weight(start_node, end_node, end_node, false);
            self.embedding[start_node].first_nbr = Some(end_node);
        } else {
            let ref_nbr_node = ref_nbr.unwrap();
            let ccw_ref_node = Some(self.get_edge_weight(start_node, ref_nbr_node, false));
            self.add_half_edge_cw(start_node, end_node, ccw_ref_node);
            if ref_nbr == self.embedding[start_node].first_nbr {
                self.embedding[start_node].first_nbr = Some(end_node);
            }
        }
    }

    fn add_half_edge_first(&mut self, start_node: NodeIndex, end_node: NodeIndex) {
        let ref_node: Option<NodeIndex> = if self.embedding.node_bound() >= start_node.index()
            && !self.embedding[start_node].first_nbr.is_none()
        {
            self.embedding[start_node].first_nbr
        } else {
            None
        };
        self.add_half_edge_ccw(start_node, end_node, ref_node);
    }

    fn next_face_half_edge(&self, v: NodeIndex, w: NodeIndex) -> (NodeIndex, NodeIndex) {
        let new_node = self.get_edge_weight(v, w, false);
        (w, new_node)
    }

    fn update_edge_weight(&mut self, v: NodeIndex, w: NodeIndex, new_node: NodeIndex, cw: bool) {
        let found_edge = self.embedding.find_edge(v, w);
        let mut found_weight = self.embedding.edge_weight_mut(found_edge.unwrap()).unwrap();

        if cw {
            found_weight.cw = Some(new_node);
        } else {
            found_weight.ccw = Some(new_node);
        }
    }

    fn get_edge_weight(&self, v: NodeIndex, w: NodeIndex, cw: bool) -> NodeIndex {
        let found_edge = self.embedding.find_edge(v, w);
        let found_weight = self.embedding.edge_weight(found_edge.unwrap()).unwrap();

        if cw {
            found_weight.cw.unwrap()
        } else {
            found_weight.ccw.unwrap()
        }
    }
}

fn id_to_index<G: GraphBase + NodeIndexable>(graph: G, node_id: G::NodeId) -> NodeIndex {
    NodeIndex::new(graph.to_index(node_id))
}

fn index_to_id<G: GraphBase + NodeIndexable>(graph: G, node_index: NodeIndex) -> G::NodeId {
    graph.from_index(node_index.index())
}

pub fn create_embedding<G: GraphBase + NodeIndexable>(
    planar_emb: &mut PlanarEmbedding,
    lr_state: &LRState<G>,
) where
    <G as GraphBase>::NodeId: Hash + Eq,
    <G as GraphBase>::NodeId: Debug,
    // These are needed for par_sort
    G: std::marker::Sync,
    <G as GraphBase>::NodeId: Sync,
{
    // for node in lr_state.dir_graph.node_indices() {
    //     for edge in lr_state.dir_graph.edges(node) {
    //         println!("Edge {:?}, {:?}", edge.source(), edge.target());
    //     }
    // }

    let mut ordered_adjs: Vec<Vec<NodeIndex>> = Vec::new();

    for v in lr_state.dir_graph.node_indices() {
        ordered_adjs.push(lr_state.dir_graph.edges(v).map(|e| e.target()).collect());

        let first_nbr = FirstNbr::<NodeIndex>::default();
        planar_emb.embedding.add_node(first_nbr);
        // Change the sign for nesting_depth
        // for e in lr_state.dir_graph.edges(v) {
        //     lr_state.nesting_depth[e] = sign(e, &lr_state.eref, &lr_state.side) * lr_state.nesting_depth[&e];
        // }
    }
    // for x in &ordered_adjs {
    //     println!("ordered {:?}", x);
    // }
    // for x in &lr_state.nesting_depth {
    //     println!("nesting {:?}", x);
    // }

    //lr_state.nesting_depth.iter().enumerate().map(|(e, val)| (e, val * lr_state.sign(e))

    for (v, adjs) in ordered_adjs.iter_mut().enumerate() {
        adjs.par_sort_by_key(|x| {
            lr_state.nesting_depth[&(
                index_to_id(&lr_state.graph, NodeIndex::new(v)),
                index_to_id(&lr_state.graph, *x),
            )]
        });
    }
    // for x in &ordered_adjs {
    //     println!("ordered 2222 {:?}", x);
    // }

    for v in lr_state.dir_graph.node_indices() {
        let mut prev_node: Option<NodeIndex> = None;
        for w in &ordered_adjs[v.index()] {
            planar_emb.add_half_edge_cw(v, *w, prev_node);
            prev_node = Some(*w)
        }
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
    pub fn sign<G>(
        edge: Edge<G>,
        eref: &mut HashMap<Edge<G>, Edge<G>>,
        side: &mut HashMap<Edge<G>, Sign>,
    ) -> i32
    where
        G: GraphBase,
        Edge<G>: Hash + Eq + Copy,
    {
        let mut dfs_stack: Vec<Edge<G>> = vec![edge];
        let mut old_ref: HashMap<Edge<G>, Edge<G>> =
            HashMap::with_capacity(eref.len());

        // """Resolve the relative side of an edge to the absolute side."""

        let mut e: Edge<G> = edge;
        let mut side_final: i32 = 1;
        while dfs_stack.len() > 0 {
            e = dfs_stack.pop().unwrap();

            if eref.contains_key(&e) {
                dfs_stack.push(e);
                dfs_stack.push(eref[&e]);
                *old_ref.get_mut(&e).unwrap() = eref[&e];
                eref.remove(&e);
            } else {
                if side[&e] == side[&old_ref[&e]] {
                    *side.get_mut(&e).unwrap() = Sign::Plus;
                    side_final = 1;
                } else {
                    *side.get_mut(&e).unwrap() = Sign::Minus;
                    side_final = -1;
                }
            }
        }
        side_final
    }
}

pub fn embedding_to_pos(planar_emb: &PlanarEmbedding) -> Vec<Point> {
    let mut pos: Vec<Point> = Vec::with_capacity(planar_emb.embedding.node_count());
    if planar_emb.embedding.node_count() < 4 {
        let default_pos = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]].to_vec();
        pos = planar_emb
            .embedding
            .node_indices()
            .map(|n| default_pos[n.index()])
            .collect();
    }
    let outer_face = triangulate_embedding(&planar_emb, true);

    let right_t_child = HashMap::<NodeIndex, usize>::new();
    let left_t_child = HashMap::<NodeIndex, usize>::new();
    let delta_x = HashMap::<NodeIndex, usize>::new();
    let y_coord = HashMap::<NodeIndex, usize>::new();

    let node_list = canonical_ordering(&planar_emb, outer_face);

    pos
}

fn triangulate_embedding(planar_emb: &PlanarEmbedding, fully_triangulate: bool) -> Vec<NodeIndex> {
    if planar_emb.embedding.node_count() <= 1 {
        return planar_emb
            .embedding
            .node_indices()
            .map(|n| n)
            .collect::<Vec<_>>();
    }
    //let component_nodes = connected_components(embedding);
    let outer_face = planar_emb
        .embedding
        .node_indices()
        .map(|n| n)
        .collect::<Vec<_>>();
    // println!("DFLT {:?}", outer_face);
    outer_face
}

fn canonical_ordering(planar_emb: &PlanarEmbedding, outer_face: Vec<NodeIndex>) -> Vec<NodeIndex> {
    if planar_emb.embedding.node_count() <= 1 {
        return planar_emb
            .embedding
            .node_indices()
            .map(|n| n)
            .collect::<Vec<_>>();
    }
    //let component_nodes = connected_components(embedding);
    let outer_face = planar_emb
        .embedding
        .node_indices()
        .map(|n| n)
        .collect::<Vec<_>>();
    // println!("DFLT {:?}", outer_face);
    outer_face
}
