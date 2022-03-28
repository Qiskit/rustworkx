use petgraph::EdgeType;

use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;

pub fn planar_layout<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    scale: Option<f64>,
    center: Option<[f64; 2]>,
) -> Pos2DMapping {

    match scale {
        Some(scale) => scale,
        None => 1.0,
    };
    println!("HELLO WORLD!");
    Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let random_tuple: [f64; 2] = [5.0, 5.0];
                match center {
                    Some(center) => (
                        n.index(),
                        [random_tuple[0] + center[0], random_tuple[1] + center[1]],
                    ),
                    None => (n.index(), random_tuple),
                }
            })
            .collect(),
    }
}

//     import numpy as np
// 
//     if dim != 2:
//         raise ValueError("can only handle 2 dimensions")
// 
//     G, center = _process_params(G, center, dim)
// 
//     if len(G) == 0:
//         return {}
// 
//     if isinstance(G, nx.PlanarEmbedding):
//         embedding = G
//     else:
//         is_planar, embedding = nx.check_planarity(G)
//         if not is_planar:
//             raise nx.NetworkXException("G is not planar.")
//     pos = nx.combinatorial_embedding_to_pos(embedding)
//     node_list = list(embedding)
//     pos = np.row_stack([pos[x] for x in node_list])
//     pos = pos.astype(np.float64)
//     pos = rescale_layout(pos, scale=scale) + center
//     return dict(zip(node_list, pos))
// 