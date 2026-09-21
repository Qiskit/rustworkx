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

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, NodeIndexable};

use super::InvalidInputError;

// Helper to convert flat index -> coordinates
fn index_to_coords(index: usize, dim: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(dim.len());
    let mut rem = index;
    for &d in dim {
        coords.push(rem % d);
        rem /= d;
    }
    coords
}

// Helper to convert coordinates -> flat index
fn coords_to_index(coords: &[usize], dim: &[usize]) -> usize {
    let mut idx = 0;
    let mut mult = 1;
    for (i, &c) in coords.iter().enumerate() {
        idx += c * mult;
        mult *= dim[i];
    }
    idx
}

/// Generate an n-dimensional grid graph.
///
/// Nodes are placed at integer coordinates in an n-dimensional box, and edges
/// connect nodes that differ by 1 in exactly one coordinate.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::nd_grid_graph;
///
/// let g: petgraph::graph::UnGraph<(), ()> = nd_grid_graph(
///     &[2, 2, 2], || (), || (), false, None
/// ).unwrap();
/// assert_eq!(g.node_count(), 8);
/// assert_eq!(g.edge_count(), 12);
/// ```
pub fn nd_grid_graph<G, T, F, H, M>(
    dim: &[usize],
    mut default_node_weight: F,
    mut default_edge_weight: H,
    bidirectional: bool,
    periodic: Option<&[bool]>,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
{
    if dim.is_empty() || dim.iter().any(|&d| d == 0) {
        return Err(InvalidInputError {});
    }

    if let Some(p) = periodic {
        if p.len() != dim.len() {
            return Err(InvalidInputError {});
        }
        // periodic dim needs at least 2 nodes (can't wrap to self)
        for (i, &is_per) in p.iter().enumerate() {
            if is_per && dim[i] < 2 {
                return Err(InvalidInputError {});
            }
        }
    }

    let num_nodes: usize = dim.iter().product();

    // count edges: each dimension contributes (size-1) edges per line, or size if periodic
    let mut num_edges = 0;
    for (i, &sz) in dim.iter().enumerate() {
        let per_line = if periodic.map_or(false, |p| p[i]) { sz } else { sz - 1 };
        num_edges += per_line * (num_nodes / sz);
    }
    if bidirectional {
        num_edges *= 2;
    }

    let mut graph = G::with_capacity(num_nodes, num_edges);

    for _ in 0..num_nodes {
        graph.add_node(default_node_weight());
    }

    // add edges: connect each node to its neighbor in positive direction per dimension
    for node_idx in 0..num_nodes {
        let coords = index_to_coords(node_idx, dim);

        for (d, &sz) in dim.iter().enumerate() {
            let is_per = periodic.map_or(false, |p| p[d]);

            if coords[d] + 1 < sz {
                let mut nb = coords.clone();
                nb[d] += 1;
                let nb_idx = coords_to_index(&nb, dim);

                let a = graph.from_index(node_idx);
                let b = graph.from_index(nb_idx);
                graph.add_edge(a, b, default_edge_weight());
                if bidirectional {
                    graph.add_edge(b, a, default_edge_weight());
                }
            } else if is_per {
                // wrap around
                let mut nb = coords.clone();
                nb[d] = 0;
                let nb_idx = coords_to_index(&nb, dim);

                let a = graph.from_index(node_idx);
                let b = graph.from_index(nb_idx);
                graph.add_edge(a, b, default_edge_weight());
                if bidirectional {
                    graph.add_edge(b, a, default_edge_weight());
                }
            }
        }
    }

    Ok(graph)
}

/// Like `nd_grid_graph` but assigns node weights based on coordinates.
pub fn nd_grid_graph_weighted<G, T, F, H, M>(
    dim: &[usize],
    mut node_weight_fn: F,
    mut default_edge_weight: H,
    bidirectional: bool,
    periodic: Option<&[bool]>,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut(&[usize]) -> T,
    H: FnMut() -> M,
{
    if dim.is_empty() || dim.iter().any(|&d| d == 0) {
        return Err(InvalidInputError {});
    }

    if let Some(p) = periodic {
        if p.len() != dim.len() {
            return Err(InvalidInputError {});
        }
        for (i, &is_per) in p.iter().enumerate() {
            if is_per && dim[i] < 2 {
                return Err(InvalidInputError {});
            }
        }
    }

    let num_nodes: usize = dim.iter().product();

    let mut num_edges = 0;
    for (i, &sz) in dim.iter().enumerate() {
        let per_line = if periodic.map_or(false, |p| p[i]) { sz } else { sz - 1 };
        num_edges += per_line * (num_nodes / sz);
    }
    if bidirectional {
        num_edges *= 2;
    }

    let mut graph = G::with_capacity(num_nodes, num_edges);

    for i in 0..num_nodes {
        let coords = index_to_coords(i, dim);
        graph.add_node(node_weight_fn(&coords));
    }

    for node_idx in 0..num_nodes {
        let coords = index_to_coords(node_idx, dim);

        for (d, &sz) in dim.iter().enumerate() {
            let is_per = periodic.map_or(false, |p| p[d]);

            if coords[d] + 1 < sz {
                let mut nb = coords.clone();
                nb[d] += 1;
                let nb_idx = coords_to_index(&nb, dim);

                let a = graph.from_index(node_idx);
                let b = graph.from_index(nb_idx);
                graph.add_edge(a, b, default_edge_weight());
                if bidirectional {
                    graph.add_edge(b, a, default_edge_weight());
                }
            } else if is_per {
                let mut nb = coords.clone();
                nb[d] = 0;
                let nb_idx = coords_to_index(&nb, dim);

                let a = graph.from_index(node_idx);
                let b = graph.from_index(nb_idx);
                graph.add_edge(a, b, default_edge_weight());
                if bidirectional {
                    graph.add_edge(b, a, default_edge_weight());
                }
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petgraph;
    use petgraph::visit::EdgeRef;

    #[test]
    fn test_line_graph() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[5], || (), || (), false, None).unwrap();
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_grid_2d() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[3, 4], || (), || (), false, None).unwrap();
        assert_eq!(g.node_count(), 12);
        assert_eq!(g.edge_count(), 17);
    }

    #[test]
    fn test_cube() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[2, 2, 2], || (), || (), false, None).unwrap();
        assert_eq!(g.node_count(), 8);
        assert_eq!(g.edge_count(), 12);
    }

    #[test]
    fn test_hypercube_4d() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[2, 2, 2, 2], || (), || (), false, None).unwrap();
        assert_eq!(g.node_count(), 16);
        assert_eq!(g.edge_count(), 32);
    }

    #[test]
    fn test_bidir() {
        let g: petgraph::graph::DiGraph<(), ()> =
            nd_grid_graph(&[2, 2], || (), || (), true, None).unwrap();
        assert_eq!(g.edge_count(), 8);
    }

    #[test]
    fn test_cycle() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[5], || (), || (), false, Some(&[true])).unwrap();
        assert_eq!(g.edge_count(), 5);
    }

    #[test]
    fn test_torus() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[3, 3], || (), || (), false, Some(&[true, true])).unwrap();
        assert_eq!(g.edge_count(), 18);
    }

    #[test]
    fn test_coords() {
        let g: petgraph::graph::UnGraph<Vec<usize>, ()> =
            nd_grid_graph_weighted(&[2, 3], |c| c.to_vec(), || (), false, None).unwrap();
        assert_eq!(g[petgraph::graph::NodeIndex::new(0)], vec![0, 0]);
        assert_eq!(g[petgraph::graph::NodeIndex::new(1)], vec![1, 0]);
        assert_eq!(g[petgraph::graph::NodeIndex::new(3)], vec![1, 1]);
    }

    #[test]
    fn test_empty_err() {
        let r: Result<petgraph::graph::UnGraph<(), ()>, _> =
            nd_grid_graph(&[], || (), || (), false, None);
        assert!(r.is_err());
    }

    #[test]
    fn test_zero_err() {
        let r: Result<petgraph::graph::UnGraph<(), ()>, _> =
            nd_grid_graph(&[2, 0], || (), || (), false, None);
        assert!(r.is_err());
    }

    #[test]
    fn test_cube_edges() {
        let g: petgraph::graph::UnGraph<(), ()> =
            nd_grid_graph(&[2, 2, 2], || (), || (), false, None).unwrap();

        let edges: Vec<(usize, usize)> = g
            .edge_references()
            .map(|e| {
                let (a, b) = (e.source().index(), e.target().index());
                (a.min(b), a.max(b))
            })
            .collect();

        // check some expected edges
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(0, 4)));
    }
}
