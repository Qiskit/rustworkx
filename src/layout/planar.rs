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

use petgraph::prelude::*;
use petgraph::visit::NodeIndexable;
use pyo3::prelude::*;

use super::super::GraphNotPlanar;
use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;
use crate::layout::embedding::{create_embedding, embedding_to_pos, PlanarEmbedding};
use crate::StablePyGraph;
use rustworkx_core::dictmap::*;
use rustworkx_core::planar::{is_planar_for_layout, LRState};

/// If a graph is planar, create a set of position coordinates for a planar
/// layout that can be passed to a drawer.
pub fn planar_layout(
    graph: &StablePyGraph<Undirected>,
    scale: Option<f64>,
    center: Option<Point>,
) -> PyResult<Pos2DMapping> {
    let node_num = graph.node_bound();
    if node_num == 0 {
        return Ok(Pos2DMapping {
            pos_map: DictMap::new(),
        });
    }

    // First determine if the graph is planar.
    let mut lr_state = LRState::new(graph);
    if !is_planar_for_layout(graph, Some(&mut lr_state)) {
        Err(GraphNotPlanar::new_err("The input graph is not planar."))

    // If planar, create the position coordinates.
    } else {
        let mut planar_emb = PlanarEmbedding::new();
        planar_emb.embedding = StableGraph::with_capacity(node_num, 0);

        // First create the graph embedding
        create_embedding(&mut planar_emb, &mut lr_state);

        // Then convert the embedding to position coordinates.
        let mut pos = embedding_to_pos(&mut planar_emb);

        if let Some(scale) = scale {
            rescale(
                &mut pos,
                scale,
                graph.node_indices().map(|n| n.index()).collect(),
            );
        }
        if let Some(center) = center {
            recenter(&mut pos, center);
        }
        Ok(Pos2DMapping {
            pos_map: graph
                .node_indices()
                .map(|n| {
                    let n = n.index();
                    (n, pos[n])
                })
                .collect(),
        })
    }
}
