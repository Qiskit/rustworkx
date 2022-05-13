use petgraph::prelude::*;

use super::spring::{recenter, rescale, Point};
use crate::connected_components;
use crate::iterators::Pos2DMapping;
use crate::Graph;
use crate::StablePyGraph;
use retworkx_core::dictmap::*;
use retworkx_core::planar::{PlanarEmbedding, is_planar, create_embedding};
// use retworkx_core::planar::is_planar;
// use retworkx_core::planar::create_embedding;

pub fn planar_layout(
    graph: &StablePyGraph<Undirected>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {

    let node_num = graph.node_count();
    println!("NODE NUM {}", node_num);
    if node_num == 0 {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    }

    println!("before is");
    let (its_planar, lr_state) = is_planar(graph);

    if !its_planar {
        println!("is false");
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };
    } else {
        println!("is true");

        let mut planar_emb = PlanarEmbedding::default();
        planar_emb.embedding = Graph::with_capacity(node_num, 0);
        println!("ROOTS {:?}", lr_state.roots);
        let mut pos = create_embedding(&planar_emb, &lr_state);

        println!("after emb to pos {:?}", pos);

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
