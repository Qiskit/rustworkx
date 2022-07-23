use petgraph::prelude::*;

use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;
use crate::layout::embedding::{create_embedding, embedding_to_pos, PlanarEmbedding};
use crate::Graph;
use crate::StablePyGraph;
use retworkx_core::dictmap::*;
use retworkx_core::planar::{is_planar, LRState};

/// If a graph is planar, create a set of position coordinates for a planar
/// layout that can be passed to a drawer.
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

    // First determine if the graph is planar.
    let mut lr_state = LRState::new(graph);
    let its_planar = is_planar(graph, Some(&mut lr_state));

    // If not planar, return an empty pos_map
    if !its_planar {
        return Pos2DMapping {
            pos_map: DictMap::new(),
        };

    // If planar, create the position coordinates.
    } else {
        let mut planar_emb = PlanarEmbedding::new();
        planar_emb.embedding = Graph::with_capacity(node_num, 0);

        // First create the graph embedding
        create_embedding(&mut planar_emb, &mut lr_state);

        for node in planar_emb.embedding.node_indices() {
            println!("node {:?} data {:?}", node, planar_emb.embedding[node]);
        }
        // Then convert the embedding to position coordinates.
        let mut pos = embedding_to_pos(&mut planar_emb);

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
