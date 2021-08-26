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

use crate::iterators::CentralityMapping;

use crate::digraph;
use crate::graph;

use pyo3::prelude::*;
use pyo3::Python;
use std::collections::VecDeque;

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

// The algorithm here is taken from:
// Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
// Journal of Mathematical Sociology 25(2):163-177, 2001.
//
// Correspondence of variable names to quantities in the paper is as follows:
//
// P -- predecessors
// S -- verts_sorted_by_distance,
//      vertices in order of non-decreasing distance from s
// Q -- Q
// sigma -- sigma
// delta -- delta
// d -- distance

pub fn betweenness_centrality<G>(
    graph: G,
    endpoints: bool,
    normalized: bool,
) -> Vec<Option<f64>>
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
    let max_index = graph.node_bound();

    let mut betweenness: Vec<Option<f64>> = vec![None; max_index];
    for node_s in graph.node_identifiers() {
        let is: usize = graph.to_index(node_s);
        betweenness[is] = Some(0.0);
    }
    for node_s in graph.node_identifiers() {
        let is: usize = graph.to_index(node_s);
        let mut shortest_path_calc =
            shortest_path_for_centrality(&graph, &node_s);
        if endpoints {
            _accumulate_endpoints(
                &mut betweenness,
                max_index,
                &mut shortest_path_calc,
                is,
            );
        } else {
            _accumulate_basic(
                &mut betweenness,
                max_index,
                &mut shortest_path_calc,
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

    betweenness
}

fn _rescale(
    betweenness: &mut Vec<Option<f64>>,
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
        for x in betweenness.iter_mut() {
            *x = x.map(|y| y * scale);
        }
    }
}

fn _accumulate_basic(
    betweenness: &mut Vec<Option<f64>>,
    max_index: usize,
    path_calc: &mut ShortestPathData,
    is: usize,
) {
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += path_calc.sigma[v] * coeff;
        }
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw]);
        }
    }
}

fn _accumulate_endpoints(
    betweenness: &mut Vec<Option<f64>>,
    max_index: usize,
    path_calc: &mut ShortestPathData,
    is: usize,
) {
    betweenness[is] = betweenness[is]
        .map(|x| x + ((path_calc.verts_sorted_by_distance.len() - 1) as f64));
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += path_calc.sigma[v] * coeff;
        }
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw] + 1.0);
        }
    }
}

struct ShortestPathData {
    verts_sorted_by_distance: Vec<NodeIndex>,
    predecessors: HashMap<NodeIndex, Vec<NodeIndex>>,
    sigma: HashMap<NodeIndex, f64>,
}

fn shortest_path_for_centrality<G>(
    graph: G,
    node_s: &G::NodeId,
) -> ShortestPathData
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphBase<NodeId = NodeIndex>, // for get() and get_mut()
{
    let mut verts_sorted_by_distance: Vec<NodeIndex> = Vec::new(); // a stack
    let c = graph.node_count();
    let mut predecessors =
        HashMap::<G::NodeId, Vec<G::NodeId>>::with_capacity(c);
    let mut sigma = HashMap::<G::NodeId, f64>::with_capacity(c);
    let mut distance = HashMap::<G::NodeId, i64>::with_capacity(c);
    #[allow(non_snake_case)]
    let mut Q: VecDeque<NodeIndex> = VecDeque::with_capacity(c);

    let i_s = graph.to_index(*node_s);
    let index_s = NodeIndex::new(i_s);

    for node in graph.node_identifiers() {
        predecessors.insert(node, Vec::new());
        sigma.insert(node, 0.0);
        distance.insert(node, -1);
    }
    sigma.insert(index_s, 1.0);
    distance.insert(index_s, 0);
    Q.push_back(index_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let distance_v = distance[&v];
        for w in graph.neighbors(v) {
            if distance[&w] < 0 {
                Q.push_back(w);
                distance.insert(w, distance_v + 1);
            }
            if distance[&w] == distance_v + 1 {
                sigma.insert(w, sigma[&w] + sigma[&v]);
                let e_p = predecessors.get_mut(&w).unwrap();
                e_p.push(v);
            }
        }
    }
    verts_sorted_by_distance.reverse(); // will be effectively popping from the stack
    ShortestPathData {
        verts_sorted_by_distance,
        predecessors,
        sigma,
    }
}

/// Compute the betweenness centrality of all nodes in a PyGraph.
///
/// :param PyGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param bool endpoints: Whether to include the endpoints of paths in pathlengths used to
///    compute the betweenness.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      betweenness score for each node.
/// :rtype: CentralityMapping
#[pyfunction(normalized = "true", endpoints = "false")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False)")]
pub fn graph_betweenness_centrality(
    _py: Python,
    graph: &graph::PyGraph,
    normalized: bool,
    endpoints: bool,
) -> PyResult<CentralityMapping> {
    let betweenness = betweenness_centrality(&graph, endpoints, normalized);
    let out_map = CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    };
    Ok(out_map)
}

/// Compute the betweenness centrality of all nodes in a PyDiGraph.
///
/// :param PyDiGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param bool endpoints: Whether to include the endpoints of paths in pathlengths used to
///    compute the betweenness.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      betweenness score for each node.
/// :rtype: CentralityMapping
#[pyfunction(normalized = "true", endpoints = "false")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False)")]
pub fn digraph_betweenness_centrality(
    _py: Python,
    graph: &digraph::PyDiGraph,
    normalized: bool,
    endpoints: bool,
) -> PyResult<CentralityMapping> {
    let betweenness = betweenness_centrality(&graph, endpoints, normalized);
    let out_map = CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    };
    Ok(out_map)
}
