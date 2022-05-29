use petgraph::prelude::*;

use super::spring::{recenter, rescale, Point};
use crate::connected_components;
use crate::iterators::Pos2DMapping;
use crate::Graph;
use crate::StablePyGraph;
use retworkx_core::dictmap::*;
use retworkx_core::planar::{
    combinatorial_embedding_to_pos, create_embedding, is_planar, LRState, PlanarEmbedding,
};

pub fn planar_layout(
    graph: &StablePyGraph<Undirected>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    if node_num == 0 {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    }

    let (its_planar, lr_state) = is_planar(graph);

    if !its_planar {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    } else {
        let mut planar_emb = PlanarEmbedding::default();
        planar_emb.embedding = Graph::with_capacity(node_num, 0);

        create_embedding(&mut planar_emb, &lr_state);

        let mut pos = combinatorial_embedding_to_pos(&planar_emb);

        if let Some(scale) = scale {
            rescale(&mut pos, scale, (0..node_num).collect());
        }
        if let Some(center) = center {
            recenter(&mut pos, center);
        }
        Pos2DMapping {
            pos_map: planar_emb
                .embedding
                .node_indices()
                .map(|n| n.index())
                .zip(pos)
                .collect(),
        }
    }
}
