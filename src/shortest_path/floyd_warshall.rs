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

use hashbrown::HashMap;
use rustworkx_core::dictmap::*;

use crate::{get_edge_iter_with_weights, weight_callable};

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, NodeIndexable};
use petgraph::EdgeType;

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::iterators::{AllPairsPathLengthMapping, PathLengthMapping};
use crate::StablePyGraph;

pub fn floyd_warshall<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: DictMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathLengthMapping {
                            path_lengths: DictMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let n = graph.node_bound();

    // Allocate empty matrix
    let mut mat: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];

    // Set diagonal to 0
    for i in 0..n {
        if let Some(row_i) = mat.get_mut(i) {
            row_i.entry(i).or_insert(0.0);
        }
    }

    // Utility to set row_i[j] = min(row_i[j], m_ij)
    macro_rules! insert_or_minimize {
        ($row_i: expr, $j: expr, $m_ij: expr) => {{
            $row_i
                .entry($j)
                .and_modify(|e| {
                    if $m_ij < *e {
                        *e = $m_ij;
                    }
                })
                .or_insert($m_ij);
        }};
    }

    // Build adjacency matrix
    for edge in graph.edge_references() {
        let i = NodeIndexable::to_index(&graph, edge.source());
        let j = NodeIndexable::to_index(&graph, edge.target());
        let weight = edge.weight().clone();

        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        if let Some(row_i) = mat.get_mut(i) {
            insert_or_minimize!(row_i, j, edge_weight);
        }
        if as_undirected {
            if let Some(row_j) = mat.get_mut(j) {
                insert_or_minimize!(row_j, i, edge_weight);
            }
        }
    }

    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            let row_k = mat.get(k).cloned().unwrap_or_default();
            mat.iter_mut().for_each(|row_i| {
                if let Some(m_ik) = row_i.get(&k).cloned() {
                    for (j, m_kj) in row_k.iter() {
                        let m_ikj = m_ik + *m_kj;
                        insert_or_minimize!(row_i, *j, m_ikj);
                    }
                }
            })
        }
    } else {
        for k in 0..n {
            let row_k = mat.get(k).cloned().unwrap_or_default();
            mat.par_iter_mut().for_each(|row_i| {
                if let Some(m_ik) = row_i.get(&k).cloned() {
                    for (j, m_kj) in row_k.iter() {
                        let m_ikj = m_ik + *m_kj;
                        insert_or_minimize!(row_i, *j, m_ikj);
                    }
                }
            })
        }
    }

    // Convert to return format
    let out_map: DictMap<usize, PathLengthMapping> = graph
        .node_indices()
        .map(|i| {
            let out_map = PathLengthMapping {
                path_lengths: mat[i.index()].iter().map(|(k, v)| (*k, *v)).collect(),
            };
            (i.index(), out_map)
        })
        .collect();
    Ok(AllPairsPathLengthMapping {
        path_lengths: out_map,
    })
}

pub fn floyd_warshall_numpy<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<Array2<f64>> {
    let n = graph.node_count();
    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        if as_undirected {
            mat[[j, i]] = mat[[j, i]].min(edge_weight);
        }
    }
    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d_ijk = mat[[i, k]] + mat[[k, j]];
                    if d_ijk < mat[[i, j]] {
                        mat[[i, j]] = d_ijk;
                    }
                }
            }
        }
    } else {
        for k in 0..n {
            let row_k = mat.slice(s![k, ..]).to_owned();
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row_i| {
                    let m_ik = row_i[k];
                    row_i.iter_mut().zip(row_k.iter()).for_each(|(m_ij, m_kj)| {
                        let d_ijk = m_ik + *m_kj;
                        if d_ijk < *m_ij {
                            *m_ij = d_ijk;
                        }
                    })
                })
        }
    }
    Ok(mat)
}
