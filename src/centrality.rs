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

use crate::digraph;
use crate::graph;

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use hashbrown::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    GraphBase,
    GraphProp, // allows is_directed
    IntoNeighborsDirected,
    IntoNodeIdentifiers,
    NodeCount,
    NodeIndexable,
};

// Correspondence to notation in Brandes 2001
//
// P -- predecessors
// S -- sorted_by_distance,
//      vertices in order of non-decreasing distance from s
// Q -- Q
// sigma -- sigma
// delta -- delta
// d -- distance

pub fn betweenness_centrality<G>(
    graph: G,
    endpoints: bool,
    normalized: bool,
) -> PyResult<Vec<f64>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp
        + GraphBase<NodeId = NodeIndex>,
    // rustfmt deletes the following comments if placed inline above
    // + IntoNodeIdentifiers // for node_identifiers()
    // + IntoNeighborsDirected // for neighbors()
    // + NodeCount // for node_count
    // + GraphProp // for is_directed
{
    let mut betweenness = vec![0.0; graph.node_count()];
    for node_s in graph.node_identifiers() {
        let is = graph.to_index(node_s);
        let (mut sorted_by_distance, predecessors, sigma) =
            shortest_path_for_centrality(&graph, &node_s);
        sorted_by_distance.reverse(); // will be effectively popping from the stack
        if endpoints {
            _accumulate_endpoints(
                &mut betweenness,
                graph.node_count(),
                &sorted_by_distance,
                predecessors,
                sigma,
                is,
            );
        } else {
            _accumulate_basic(
                &mut betweenness,
                graph.node_count(),
                &sorted_by_distance,
                predecessors,
                sigma,
                is,
            );
        }
    }
    _rescale(
        &mut betweenness,
        graph.node_count(),
        normalized,
        graph.is_directed(),
        endpoints,
    );

    Ok(betweenness)
}

fn _rescale(
    betweenness: &mut Vec<f64>,
    node_count: usize,
    normalized: bool,
    directed: bool,
    endpoints: bool,
) {
    let mut do_scale = true;
    let mut scale = 1.0;
    if normalized {
        if endpoints {
            if node_count < 2 {
                do_scale = false;
            } else {
                scale = 1.0 / (node_count * (node_count - 1)) as f64;
            }
        } else if node_count <= 2 {
            do_scale = false;
        } else {
            scale = 1.0 / ((node_count - 1) * (node_count - 2)) as f64;
        }
    } else if !directed {
        scale = 0.5;
    } else {
        do_scale = false;
    }
    if do_scale {
        for i in 0..betweenness.len() as usize {
            betweenness[i] *= scale;
        }
    }
}

fn _accumulate_basic(
    betweenness: &mut Vec<f64>,
    node_count: usize,
    sorted_by_distance: &Vec<NodeIndex>,
    mut predecessors: HashMap<NodeIndex, Vec<NodeIndex>>,
    sigma: HashMap<usize, i64>,
    is: usize,
) {
    let mut delta = vec![0.0; node_count];
    for w in sorted_by_distance {
        let iw = (*w).index();
        let coeff = (1.0 + delta[iw]) / (sigma[&iw] as f64);
        let p_w = predecessors.get_mut(&w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += (sigma[&iv] as f64) * coeff;
        }
        if iw != is {
            betweenness[iw] += delta[iw];
        }
    }
}

fn _accumulate_endpoints(
    betweenness: &mut Vec<f64>,
    node_count: usize,
    sorted_by_distance: &Vec<NodeIndex>,
    mut predecessors: HashMap<NodeIndex, Vec<NodeIndex>>,
    sigma: HashMap<usize, i64>,
    is: usize,
) {
    betweenness[is] += (sorted_by_distance.len() - 1) as f64;
    let mut delta = vec![0.0; node_count];
    for w in sorted_by_distance {
        let iw = (*w).index();
        let coeff = (1.0 + delta[iw]) / (sigma[&iw] as f64);
        let p_w = predecessors.get_mut(&w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += (sigma[&iv] as f64) * coeff;
        }
        if iw != is {
            betweenness[iw] += delta[iw] + 1.0;
        }
    }
}

fn shortest_path_for_centrality<G>(
    graph: G,
    node_s: &G::NodeId,
) -> (
    Vec<NodeIndex>,
    HashMap<G::NodeId, Vec<G::NodeId>>,
    HashMap<usize, i64>,
)
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + GraphBase<NodeId = NodeIndex>, // for get() and get_mut()
{
    let mut sorted_by_distance: Vec<NodeIndex> = Vec::new(); // a stack
    #[allow(non_snake_case)]
    let mut Q: Vec<NodeIndex> = Vec::new(); // a queue
    let mut predecessors = HashMap::<G::NodeId, Vec<G::NodeId>>::new();
    let mut sigma = HashMap::<usize, i64>::new();
    let mut distance = HashMap::<G::NodeId, i64>::new();

    let i_s = graph.to_index(*node_s);
    let index_s = NodeIndex::new(i_s);

    for node in graph.node_identifiers() {
        let i = graph.to_index(node);
        let index = NodeIndex::new(i);
        predecessors.insert(index, Vec::new());
        sigma.insert(i, 0);
        distance.insert(index, -1);
    }
    sigma.insert(index_s.index(), 1);
    distance.insert(index_s, 0);
    Q.push(index_s);
    while !Q.is_empty() {
        let v = Q.remove(0);
        sorted_by_distance.push(v);
        for w in graph.neighbors(v) {
            if *(distance.get(&w).unwrap()) < 0 {
                Q.push(w);
                distance.insert(w, *(distance.get(&v).unwrap()) + 1);
            }
            if *(distance.get(&w).unwrap()) == *(distance.get(&v).unwrap()) + 1
            {
                sigma.insert(
                    graph.to_index(w),
                    *(sigma.get(&graph.to_index(w)).unwrap())
                        + *(sigma.get(&graph.to_index(v)).unwrap()),
                );
                let e_p = predecessors.get_mut(&w).unwrap();
                e_p.push(v);
            }
        }
    }
    (sorted_by_distance, predecessors, sigma)
}

#[pyfunction(normalized = "true", endpoints = "false")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False)")]
pub fn graph_betweenness_centrality(
    py: Python,
    graph: &graph::PyGraph,
    normalized: bool,
    endpoints: bool,
) -> PyResult<PyObject> {
    let betweenness = betweenness_centrality(&graph, endpoints, normalized);
    Ok(PyList::new(py, betweenness).into())
}

#[pyfunction(normalized = "true", endpoints = "false")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False)")]
pub fn digraph_betweenness_centrality(
    py: Python,
    graph: &digraph::PyDiGraph,
    normalized: bool,
    endpoints: bool,
) -> PyResult<PyObject> {
    let betweenness = betweenness_centrality(&graph, endpoints, normalized);
    Ok(PyList::new(py, betweenness).into())
}
