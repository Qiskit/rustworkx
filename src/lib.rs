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

use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use daggy::Dag;
use daggy::NodeIndex;
use daggy::Walker;
use petgraph::algo;
//use petgraph::visit::WalkerIter;

struct EdgeWeight;

#[pyclass]
pub struct PyDAG {
    graph: Dag<PyObject, EdgeWeight>,
}

#[pymethods]
impl PyDAG {
    #[new]
    fn new(obj: &PyRawObject) {
        obj.init(PyDAG {
            graph: Dag::<PyObject, EdgeWeight>::new(),
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
    pub fn add_edge(&mut self, parent: usize, child: usize) -> PyResult<()> {
        let p_index = NodeIndex::new(parent);
        let c_index = NodeIndex::new(child);
        match self.graph.update_edge(p_index, c_index, EdgeWeight) {
            Err(_err) => Err(DAGWouldCycle::py_err("Adding an edge would cycle")),
            Ok(_result) => Ok(()),
        }
    }
    pub fn add_node(&mut self, obj: PyObject) -> usize {
        let index = self.graph.add_node(obj);
        index.index()
    }
    pub fn add_child(&mut self, parent: usize, obj: PyObject) -> usize {
        let index = NodeIndex::new(parent);
        let (_, index) = self.graph.add_child(index, EdgeWeight, obj);
        index.index()
    }
    pub fn add_parent(&mut self, child: usize, obj: PyObject) -> usize {
        let index = NodeIndex::new(child);
        let (_, index) = self.graph.add_parent(index, EdgeWeight, obj);
        index.index()
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

// Not finished yet, always returns 0 now
#[pyfunction]
fn dag_longest_path_length(graph: &PyDAG) -> u64 {
    let dag = &graph.graph;
    let nodes = match algo::toposort(dag.graph(), None) {
        Ok(nodes) => nodes,
        Err(_err) => panic!("DAG has a cycle, something is really wrong"),
    };
    for node in nodes {
        let parents = dag.parents(node).iter(&dag);
        let mut length: usize;
        for p_node in parents {
            println!("{}", p_node.1.index())
        }
    }
    0
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
