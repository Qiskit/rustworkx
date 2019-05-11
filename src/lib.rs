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

extern crate pyo3;

extern crate daggy;
extern crate petgraph;

use std::collections::HashMap;
use std::iter;

use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use daggy::Dag;
use daggy::NodeIndex;
use daggy::Walker;
use petgraph::algo;
use petgraph::visit::IntoNeighbors;
use petgraph::visit::IntoNeighborsDirected;
//use petgraph::visit::WalkerIter;

#[pyclass]
pub struct PyDAG {
    graph: Dag<PyObject, PyObject>,
}

#[pymethods]
impl PyDAG {
    #[new]
    fn new(obj: &PyRawObject) {
        obj.init(PyDAG {
            graph: Dag::<PyObject, PyObject>::new(),
        });
    }
    //   pub fn edges(&self) -> PyResult<()> {
    //
    //   }
    //   pub fn nodes(&self) -> PyResult<()> {
    //
    //   }
    //   pub fn successors(&self, node: usize) -> Iter<PyObject> {
    //       let index = NodeIndex::new(node);
    //       let c_walker = self.graph.children(index);
    //       c_walker.iter(&self.graph) as Iter<PyObject>
    //   }
    //   pub fn predecessors(&self, node: usize) -> Iter<PyObject> {
    //       let index = NodeIndex::new(node);
    //       let p_walker = self.graph.parents(index);
    //       p_walker.iter(&self.graph) as Iter<PyObject>
    //
    //   }
    //   pub fn get_edge_data(&self) -> PyResult<()> {
    //
    //   }
    pub fn remove_node(&mut self, node: usize) -> PyResult<()> {
        let index = NodeIndex::new(node);
        self.graph.remove_node(index);

        Ok(())
    }
    pub fn add_edge(
        &mut self,
        parent: usize,
        child: usize,
        edge: PyObject,
    ) -> PyResult<()> {
        let p_index = NodeIndex::new(parent);
        let c_index = NodeIndex::new(child);
        match self.graph.update_edge(p_index, c_index, edge) {
            Err(_err) => {
                Err(DAGWouldCycle::py_err("Adding an edge would cycle"))
            }
            Ok(_result) => Ok(()),
        }
    }
    pub fn add_node(&mut self, obj: PyObject) -> usize {
        let index = self.graph.add_node(obj);
        index.index()
    }
    pub fn add_child(
        &mut self,
        parent: usize,
        obj: PyObject,
        edge: PyObject,
    ) -> usize {
        let index = NodeIndex::new(parent);
        let (_, index) = self.graph.add_child(index, edge, obj);
        index.index()
    }
    pub fn add_parent(
        &mut self,
        child: usize,
        obj: PyObject,
        edge: PyObject,
    ) -> usize {
        let index = NodeIndex::new(child);
        let (_, index) = self.graph.add_parent(index, edge, obj);
        index.index()
    }
    pub fn adj(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.neighbors(index);
        let out_dict = PyDict::new(py);
        let graph = self.graph.graph();
        for neighbor in neighbors {
            let edge = graph.find_edge(index, neighbor);
            let edge_w = graph.edge_weight(edge.unwrap());
            out_dict.set_item(neighbor.index(), edge_w)?;
        }
        Ok(out_dict.into())
    }

    pub fn adj_direction(
        &mut self,
        py: Python,
        node: usize,
        direction: bool,
    ) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let dir;
        if direction {
            dir = petgraph::Direction::Incoming;
        } else {
            dir = petgraph::Direction::Outgoing;
        }
        let graph = self.graph.graph();
        let neighbors = self.graph.neighbors_directed(index, dir);
        let out_dict = PyDict::new(py);
        for neighbor in neighbors {
            let edge = graph.find_edge(index, neighbor);
            let edge_w = graph.edge_weight(edge.unwrap());
            out_dict.set_item(neighbor.index(), edge_w)?;
        }
        Ok(out_dict.into())
    }
    //   pub fn add_nodes_from(&self) -> PyResult<()> {
    //
    //   }
    //   pub fn add_edges_from(&self) -> PyResult<()> {
    //
    //   }
    //   pub fn number_of_edges(&self) -> PyResult<()> {
    //
    //   }
    //   pub fn in_degree(&self) -> PyResult<()> {
    //
    //   }
}

fn pairwise<'a, I>(
    xs: I,
) -> Box<Iterator<Item = (Option<I::Item>, I::Item)> + 'a>
where
    I: 'a + IntoIterator + Clone,
{
    let left = iter::once(None).chain(xs.clone().into_iter().map(Some));
    let right = xs.into_iter();
    Box::new(left.zip(right))
}

#[pyfunction]
fn dag_longest_path_length(graph: &PyDAG) -> usize {
    let dag = &graph.graph;
    let nodes = match algo::toposort(dag.graph(), None) {
        Ok(nodes) => nodes,
        Err(_err) => panic!("DAG has a cycle, something is really wrong"),
    };
    let mut dist: HashMap<usize, (usize, usize)> = HashMap::new();
    for node in nodes {
        // Iterator that yields (EdgeIndex, NodeIndex)
        let parents = dag.parents(node).iter(&dag);
        let mut us: Vec<(usize, usize)> = Vec::new();
        for (_, p_node) in parents {
            let p_index = p_node.index();
            let length = dist[&p_index].0 + 1;
            let u = p_index;
            us.push((length, u));
        }
        let maxu: (usize, usize);
        if us.len() > 0 {
            maxu = *us.iter().max_by_key(|x| x.0).unwrap();
        } else {
            maxu = (0, node.index());
        };
        dist.insert(node.index(), maxu);
    }
    let mut u: Option<usize> = None;
    let first = dist.iter().max_by_key(|(_, v)| v.0).unwrap();
    let first_v = *first.1;
    let mut v = first_v.0;
    let mut path: Vec<usize> = Vec::new();
    while !u.is_some() || u.unwrap() != v {
        path.push(v);
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    let mut path_length: usize = 0;
    for (_, _) in pairwise(path) {
        path_length += 1
    }
    path_length
}
//
//#[pyfunction]
//fn number_weakly_connected_components(graph: PyDAG) -> u64 {
//
//}

#[pyfunction]
fn is_isomorphic(first: &PyDAG, second: &PyDAG) -> bool {
    algo::is_isomorphic(first.graph.graph(), second.graph.graph())
}

//#[pyfunction]
//fn lexicographical_topological_sort(graph: PyDAG,
//                                    key: &PyObject) {
//
//}

#[pymodule]
fn retworkx(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    //   m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic))?;
    //    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_class::<PyDAG>()?;
    Ok(())
}

create_exception!(retworkx, DAGWouldCycle, Exception);
create_exception!(retworkx, DAGIsCycle, Exception);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
