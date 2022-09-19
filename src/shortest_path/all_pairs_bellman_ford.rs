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

use rustworkx_core::dictmap::*;
use rustworkx_core::shortest_path::bellman_ford;

use std::sync::RwLock;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::EdgeType;

use rayon::prelude::*;

use crate::iterators::{
    AllPairsPathLengthMapping, AllPairsPathMapping, PathLengthMapping, PathMapping,
};
use crate::{edge_weights_from_callable, NegativeCycle, StablePyGraph};

pub fn all_pairs_bellman_ford_path_lengths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    edge_cost_fn: PyObject,
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
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let negative_cycle = RwLock::new(false);

    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let out_map: DictMap<usize, PathLengthMapping> = node_indices
        .into_par_iter()
        .map(|x| {
            if *negative_cycle.read().unwrap() {
                return (
                    x.index(),
                    PathLengthMapping {
                        path_lengths: DictMap::new(),
                    },
                );
            }

            let path_lengths: Option<Vec<Option<f64>>> =
                bellman_ford(graph, x, |e| edge_cost(e.id()), None).unwrap();

            if path_lengths.is_none() {
                let mut cycle = negative_cycle.write().unwrap();
                *cycle = true;
                return (
                    x.index(),
                    PathLengthMapping {
                        path_lengths: DictMap::new(),
                    },
                );
            }

            let out_map = PathLengthMapping {
                path_lengths: path_lengths
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, opt_cost)| {
                        if index != x.index() {
                            opt_cost.map(|cost| (index, cost))
                        } else {
                            None
                        }
                    })
                    .collect(),
            };
            (x.index(), out_map)
        })
        .collect();

    if *negative_cycle.read().unwrap() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    Ok(AllPairsPathLengthMapping {
        path_lengths: out_map,
    })
}

pub fn all_pairs_bellman_ford_shortest_paths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: DictMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathMapping {
                            paths: DictMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();

    let negative_cycle = RwLock::new(false);

    let out_map = AllPairsPathMapping {
        paths: node_indices
            .into_par_iter()
            .map(|x| {
                if *negative_cycle.read().unwrap() {
                    return (
                        x.index(),
                        PathMapping {
                            paths: DictMap::new(),
                        },
                    );
                }

                let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> =
                    DictMap::with_capacity(graph.node_count());
                let path_lengths: Option<Vec<Option<f64>>> =
                    bellman_ford(graph, x, |e| edge_cost(e.id()), Some(&mut paths)).unwrap();

                if path_lengths.is_none() {
                    let mut cycle = negative_cycle.write().unwrap();
                    *cycle = true;
                    return (
                        x.index(),
                        PathMapping {
                            paths: DictMap::new(),
                        },
                    );
                }

                let index = x.index();

                let out_paths = PathMapping {
                    paths: paths
                        .iter()
                        .filter_map(|path_mapping| {
                            let path_index = path_mapping.0.index();
                            if index != path_index {
                                Some((
                                    path_index,
                                    path_mapping.1.iter().map(|x| x.index()).collect(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect(),
                };
                (index, out_paths)
            })
            .collect(),
    };

    if *negative_cycle.read().unwrap() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    Ok(out_map)
}
