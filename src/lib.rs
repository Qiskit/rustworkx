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
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use daggy::Dag;
use daggy::EdgeIndex;
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

    pub fn edges(&self, py: Python) -> PyObject {
        let raw_edges = self.graph.raw_edges();
        let mut out: Vec<&PyObject> = Vec::new();
        for edge in raw_edges {
            out.push(&edge.weight);
        }
        PyList::new(py, out).into()
    }

    pub fn nodes(&self, py: Python) -> PyObject {
        let raw_nodes = self.graph.raw_nodes();
        let mut out: Vec<&PyObject> = Vec::new();
        for node in raw_nodes {
            out.push(&node.weight);
        }
        PyList::new(py, out).into()
    }

    pub fn successors(&self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let c_walker = self.graph.children(index);
        let mut succesors: Vec<&PyObject> = Vec::new();
        for succ in c_walker.iter(&self.graph) {
            succesors.push(self.graph.node_weight(succ.1).unwrap());
        }
        Ok(PyList::new(py, succesors).into())
    }

    pub fn predecessors(&self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let p_walker = self.graph.parents(index);
        let mut predec: Vec<&PyObject> = Vec::new();
        for pred in p_walker.iter(&self.graph) {
            predec.push(self.graph.node_weight(pred.1).unwrap());
        }
        Ok(PyList::new(py, predec).into())
    }

    pub fn get_edge_data(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<(&PyObject)> {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        let edge_index = match self.graph.find_edge(index_a, index_b) {
            Some(edge_index) => edge_index,
            None => {
                return Err(NoEdgeBetweenNodes::py_err(
                    "No edge found between nodes",
                ))
            }
        };

        let data = self.graph.edge_weight(edge_index).unwrap();
        Ok(data)
    }

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
        match self.graph.add_edge(p_index, c_index, edge) {
            Err(_err) => {
                Err(DAGWouldCycle::py_err("Adding an edge would cycle"))
            }
            Ok(_result) => Ok(()),
        }
    }

    pub fn remove_edge(&mut self, parent: usize, child: usize) -> PyResult<()> {
        let p_index = NodeIndex::new(parent);
        let c_index = NodeIndex::new(child);
        let edge_index = match self.graph.find_edge(p_index, c_index) {
            Some(edge_index) => edge_index,
            None => {
                return Err(NoEdgeBetweenNodes::py_err(
                    "No edge found between nodes",
                ))
            }
        };
        self.graph.remove_edge(edge_index);
        Ok(())
    }

    pub fn remove_edge_from_index(&mut self, edge: usize) -> PyResult<()> {
        let edge_index = EdgeIndex::new(edge);
        self.graph.remove_edge(edge_index);
        Ok(())
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
            let mut edge = graph.find_edge(index, neighbor);
            // If there is no edge then it must be a parent neighbor
            if edge.is_none() {
                edge = graph.find_edge(neighbor, index);
            }
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
            let edge;
            if direction {
                edge = match graph.find_edge(neighbor, index) {
                    Some(edge) => edge,
                    None => {
                        return Err(NoEdgeBetweenNodes::py_err(
                            "No edge found between nodes",
                        ))
                    }
                };
            } else {
                edge = match graph.find_edge(index, neighbor) {
                    Some(edge) => edge,
                    None => {
                        return Err(NoEdgeBetweenNodes::py_err(
                            "No edge found between nodes",
                        ))
                    }
                };
            }
            let edge_w = graph.edge_weight(edge);
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
    pub fn in_degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Incoming;
        let neighbors = self.graph.neighbors_directed(index, dir);
        neighbors.count()
    }
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
fn dag_longest_path_length(graph: &PyDAG) -> PyResult<usize> {
    let dag = &graph.graph;
    let nodes = match algo::toposort(dag.graph(), None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    if nodes.len() == 0 {
        return Ok(0);
    }
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
        if !us.is_empty() {
            maxu = *us.iter().max_by_key(|x| x.0).unwrap();
        } else {
            maxu = (0, node.index());
        };
        dist.insert(node.index(), maxu);
    }
    let mut u: Option<usize> = None;
    let first = match dist.iter().max_by_key(|(_, v)| v.0) {
        Some(first) => first,
        None => {
            return Err(Exception::py_err("Encountered something unexpected"))
        }
    };
    let first_v = *first.1;
    let mut v = first_v.0;
    let mut path: Vec<usize> = Vec::new();
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v);
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    let mut path_length: usize = 0;
    for (_, _) in pairwise(path) {
        path_length += 1
    }
    Ok(path_length)
}

#[pyfunction]
fn number_weakly_connected_components(graph: &PyDAG) -> usize {
    let dag = graph.graph.graph();
    algo::connected_components(dag)
}

#[pyfunction]
fn is_isomorphic(first: &PyDAG, second: &PyDAG) -> bool {
    algo::is_isomorphic(first.graph.graph(), second.graph.graph())
}

#[pyfunction]
fn is_isomorphic_node_match(
    py: Python,
    first: &PyDAG,
    second: &PyDAG,
    matcher: PyObject,
) -> bool {
    let compare_nodes = |a: &PyObject, b: &PyObject| -> bool {
        let res = matcher.call1(py, (a, b)).unwrap();
        res.is_true(py).unwrap()
    };

    fn compare_edges(_a: &PyObject, _b: &PyObject) -> bool {
        true
    }
    algo::is_isomorphic_matching(
        first.graph.graph(),
        second.graph.graph(),
        compare_nodes,
        compare_edges,
    )
}

#[pyfunction]
fn topological_sort(py: Python, graph: &PyDAG) -> PyResult<PyObject> {
    let nodes = match algo::toposort(graph.graph.graph(), None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    let mut out: Vec<usize> = Vec::new();
    for node in nodes {
        out.push(node.index());
    }
    Ok(PyList::new(py, out).into())
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
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic_node_match))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    //    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_class::<PyDAG>()?;
    Ok(())
}

create_exception!(retworkx, DAGWouldCycle, Exception);
create_exception!(retworkx, NoEdgeBetweenNodes, Exception);
create_exception!(retworkx, DAGHasCycle, Exception);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
