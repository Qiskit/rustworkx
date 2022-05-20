use petgraph::prelude::*;
use petgraph::Directed;
use petgraph::visit::{GraphBase, NodeIndexable};
use petgraph::graph::Graph;
use std::hash::Hash;
use hashbrown::hash_map::HashMap;
use std::fmt::Debug;

use crate::dictmap::*;
use crate::planar::is_planar;
use crate::planar::lr_planar::LRState;

pub type Point = [f64; 2];

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
    fn new() -> Self {
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
        let new_edge = self.embedding.add_edge(start_node, end_node, cw_weight);
        if ref_nbr.is_none() {
            // The start node has no neighbors
            let first_nbr = FirstNbr::<NodeIndex>::default();
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
            let new_edge = self.embedding.add_edge(start_node, end_node, cw_weight);
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

pub fn create_embedding<G: GraphBase>(
    planar_emb: &PlanarEmbedding,
    lr_state: &LRState<G>,
) where <G as GraphBase>::NodeId: Hash + Eq, <G as GraphBase>::NodeId: Debug {
    println!("ROOTS {:?}", lr_state.roots);

    //let mut planar_emb = PlanarEmbedding::default();

}

pub fn combinatorial_embedding_to_pos(
    planar_emb: &PlanarEmbedding,
) -> Vec<Point> {
    let mut pos: Vec<Point> = Vec::with_capacity(planar_emb.embedding.node_count());
    if planar_emb.embedding.node_count() < 4 {
        let default_pos = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]].to_vec();
        pos = planar_emb.embedding
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

fn triangulate_embedding(
    planar_emb: &PlanarEmbedding,
    fully_triangulate: bool,
) -> Vec<NodeIndex> {
    if planar_emb.embedding.node_count() <= 1 {
        return planar_emb.embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    }
    //let component_nodes = connected_components(embedding);
    let outer_face = planar_emb.embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    println!("DFLT {:?}", outer_face);
    outer_face
}

fn canonical_ordering(
    planar_emb: &PlanarEmbedding,
    outer_face: Vec<NodeIndex>,
) -> Vec<NodeIndex> {
    if planar_emb.embedding.node_count() <= 1 {
        return planar_emb.embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    }
    //let component_nodes = connected_components(embedding);
    let outer_face = planar_emb.embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    println!("DFLT {:?}", outer_face);
    outer_face
}
