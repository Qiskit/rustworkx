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

#![allow(clippy::float_cmp)]

use crate::{digraph, DAGHasCycle};

use hashbrown::HashMap;

use pyo3::prelude::*;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;

use num_traits::{Num, Zero};

pub fn longest_path<F, T>(
    graph: &digraph::PyDiGraph,
    mut weight_fn: F,
) -> PyResult<(Vec<usize>, T)>
where
    F: FnMut(usize, usize, &PyObject) -> PyResult<T>,
    T: Num + Zero + PartialOrd + Copy,
{
    let dag = &graph.graph;
    let mut path: Vec<usize> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    if nodes.is_empty() {
        return Ok((path, T::zero()));
    }
    let mut dist: HashMap<NodeIndex, (T, NodeIndex)> = HashMap::new();
    for node in nodes {
        let parents = dag.edges_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(T, NodeIndex)> = Vec::new();
        for p_edge in parents {
            let p_node = p_edge.source();
            let weight: T = weight_fn(
                p_node.index(),
                p_edge.target().index(),
                p_edge.weight(),
            )?;
            let length = dist[&p_node].0 + weight;
            us.push((length, p_node));
        }
        let maxu: (T, NodeIndex) = if !us.is_empty() {
            *us.iter()
                .max_by(|a, b| {
                    let weight_a = a.0;
                    let weight_b = b.0;
                    weight_a.partial_cmp(&weight_b).unwrap()
                })
                .unwrap()
        } else {
            (T::zero(), node)
        };
        dist.insert(node, maxu);
    }
    let first = dist
        .keys()
        .max_by(|a, b| dist[a].partial_cmp(&dist[b]).unwrap())
        .unwrap();
    let mut v = *first;
    let mut u: Option<NodeIndex> = None;
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v.index());
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    let path_weight = dist[first].0;
    Ok((path, path_weight))
}
