use petgraph::{EdgeType, Directed};
use petgraph::prelude::*;
use pyo3::prelude::*;

use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;
use crate::Graph;
use crate::connected_components;
use retworkx_core::dictmap::*;

pub struct CwCcw<T>  {
    cw: Option<T>,
    ccw: Option<T>,
}

impl<T> Default for CwCcw<T>  {
    fn default() -> Self {
        CwCcw { cw: None, ccw: None }
    }
}

impl<T> CwCcw<T> {
    fn new(cw: T, ccw: T) -> Self {
        CwCcw {
            cw: Some(cw),
            ccw: Some(ccw),
        }
    }

    fn cw_is_empty(&self) -> bool {
        self.cw.is_none()
    }

    fn ccw_is_empty(&self) -> bool {
        self.ccw.is_none()
    }

    fn cw_unwrap(self) -> T {
        self.cw.unwrap()
    }

    fn ccw_unwrap(self) -> T {
        self.ccw.unwrap()
    }

    fn cw_as_ref(&mut self) -> Option<&T> {
        self.cw.as_ref()
    }

    fn ccw_as_ref(&mut self) -> Option<&T> {
        self.ccw.as_ref()
    }

    fn cw_as_mut(&mut self) -> Option<&mut T> {
        self.cw.as_mut()
    }

    fn cc_as_mut(&mut self) -> Option<&mut T> {
        self.ccw.as_mut()
    }
}

pub struct FirstNbr {
    first_nbr: Option<NodeIndex>,
}

impl Default for FirstNbr {
    fn default() -> Self {
        FirstNbr { first_nbr: None }
    }
}

pub fn planar_layout<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {

    let node_num = graph.node_count();
    if node_num == 0 {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    }
    let mut pos: Vec<Point> = Vec::with_capacity(node_num);
    let mut planar = PlanarEmbedding::new();

    if !is_planar(graph) {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    } else {
        create_embedding(graph, &mut planar.embedding);
        combinitorial_embedding_to_pos(&planar.embedding, &mut pos);

        if let Some(scale) = scale {
            rescale(&mut pos, scale, (0..node_num).collect());
        }
        if let Some(center) = center {
            recenter(&mut pos, center);
        }
        Pos2DMapping {
            pos_map: planar.embedding.node_indices().map(|n| n.index()).zip(pos).collect(),
        }
    }
}

pub fn is_planar<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
) -> bool {
    true
}

pub fn create_embedding<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    embedding: &mut Graph<FirstNbr, CwCcw<NodeIndex>, Directed>,
) -> bool {
    // DEBUG CODE FOR TESTING BASIC EMBEDDING
    // for v in graph.node_indices() {
    //     if v.index() < 5 {
    //         println!("GRAPH {:?} {}", v, graph[v].clone());
    //         embedding.add_node(graph[v].clone());
    //     } else {
    //         break;
    //     }
    // }
    true
}

pub fn combinitorial_embedding_to_pos (
    embedding: &Graph<FirstNbr, CwCcw<NodeIndex>, Directed>,
    pos: &mut Vec<Point>,
){
    if embedding.node_count() < 4 {
        *pos = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]].to_vec()
    }
    let outer_face = triangulate_embedding(&embedding, true);

    // DEBUG: Need proper value for pos
    *pos = [[0.4, 0.5]].to_vec()
}

pub fn triangulate_embedding (
    embedding: &Graph<FirstNbr, CwCcw<NodeIndex>, Directed>,
    fully_triangulate: bool,
) -> Vec<NodeIndex> {
    if embedding.node_count() <= 1 {
        return embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    }
    //let component_nodes = connected_components(embedding);
    let outer_face = embedding.node_indices().map(|n| n).collect::<Vec<_>>();
    println!("DFLT {:?}", outer_face);
    outer_face
}

struct PlanarEmbedding {
    embedding: Graph::<FirstNbr, CwCcw<NodeIndex>, Directed>,
}

impl Default for PlanarEmbedding {
    fn default () -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr, CwCcw<NodeIndex>, Directed>::new(),
        }
    }
}
impl PlanarEmbedding {
    pub fn new () -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr, CwCcw<NodeIndex>, Directed>::new(),
        }
    }

    fn neighbors_cw_order (&self, v: NodeIndex)
    {
    }

    fn check_structure (&self)
    {
    }

    fn add_half_edge_ccw (
        &self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_neighbor: Option<NodeIndex>
    ){
    }

    fn add_half_edge_cw (
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_neighbor: Option<NodeIndex>
    ){
        let mut cw_weight = CwCcw::<NodeIndex>::default();
        let new_edge = self.embedding.add_edge(start_node, end_node, cw_weight);
        if !ref_neighbor.is_none() {
            self.embedding[new_edge].cw = Some(end_node.clone());
            //self.embedding.update_edge(start_node, end_node, upd_weight);
        }
    }

    fn add_half_edge_first (&self, start_node: NodeIndex, end_node: NodeIndex)
    {
    }

    fn next_face_half_edge (&self, v: NodeIndex, w: NodeIndex)
    {
    }

    fn traverse_face (&self, v: NodeIndex, w: NodeIndex, mark_half_edges: bool)
    {
    }

}
