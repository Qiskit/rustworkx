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

use super::{digraph, graph};

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::EdgeType;

use rayon::prelude::*;

pub fn _core_number<Ty>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
) -> PyResult<PyObject>
where
    Ty: EdgeType,
{
    let node_num = graph.node_count();
    if node_num == 0 {
        return Ok(PyDict::new(py).into());
    }

    let mut cores: HashMap<NodeIndex, usize> = HashMap::with_capacity(node_num);
    let mut node_vec: Vec<NodeIndex> = graph.node_indices().collect();
    let mut degree_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_num);
    let mut nbrs: HashMap<NodeIndex, HashSet<NodeIndex>> =
        HashMap::with_capacity(node_num);
    let mut node_pos: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_num);

    for k in node_vec.iter() {
        let k_nbrs: HashSet<NodeIndex> =
            graph.neighbors_undirected(*k).collect();
        let k_deg = k_nbrs.len();

        nbrs.insert(*k, k_nbrs);
        cores.insert(*k, k_deg);
        degree_map.insert(*k, k_deg);
    }
    node_vec.par_sort_by_key(|k| degree_map.get(k));

    let mut bin_boundaries: Vec<usize> =
        Vec::with_capacity(degree_map[&node_vec[node_num - 1]] + 1);
    bin_boundaries.push(0);
    let mut curr_degree = 0;
    for (i, v) in node_vec.iter().enumerate() {
        node_pos.insert(*v, i);
        let v_degree = degree_map[v];
        if v_degree > curr_degree {
            for _ in 0..v_degree - curr_degree {
                bin_boundaries.push(i);
            }
            curr_degree = v_degree;
        }
    }

    for v_ind in 0..node_vec.len() {
        let v = node_vec[v_ind];
        let v_nbrs = nbrs[&v].clone();
        for u in v_nbrs {
            if cores[&u] > cores[&v] {
                nbrs.get_mut(&u).unwrap().remove(&v);
                let pos = node_pos[&u];
                let bin_start = bin_boundaries[cores[&u]];
                *node_pos.get_mut(&u).unwrap() = bin_start;
                *node_pos.get_mut(&node_vec[bin_start]).unwrap() = pos;
                node_vec.swap(bin_start, pos);
                bin_boundaries[cores[&u]] += 1;
                *cores.get_mut(&u).unwrap() -= 1;
            }
        }
    }

    let out_dict = PyDict::new(py);
    for (v_index, core) in cores {
        out_dict.set_item(v_index.index(), core)?;
    }
    Ok(out_dict.into())
}

/// Return the core number for each node in the graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The graph to get core numbers
///
/// :returns: A dictionary keyed by node index to the core number
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_core_number(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<PyObject> {
    _core_number(py, &graph.graph)
}

/// Return the core number for each node in the directed graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
/// For directed graphs, the degree is calculated as in_degree + out_degree.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyDiGraph: The directed graph to get core numbers
///
/// :returns: A dictionary keyed by node index to the core number
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn digraph_core_number(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> PyResult<PyObject> {
    _core_number(py, &graph.graph)
}
