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

extern crate fixedbitset;
extern crate hashbrown;
extern crate ndarray;
extern crate numpy;
extern crate petgraph;
extern crate pyo3;

mod dag_isomorphism;
mod graph;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::ops::{Index, IndexMut};

use hashbrown::HashMap;

use pyo3::class::PyMappingProtocol;
use pyo3::create_exception;
use pyo3::exceptions::{Exception, IndexError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyLong, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{
    Bfs, GetAdjacencyMatrix, GraphBase, GraphProp, IntoEdgeReferences,
    IntoEdges, IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected,
    IntoNodeIdentifiers, IntoNodeReferences, NodeCompactIndexable, NodeCount,
    NodeIndexable, Reversed, Visitable,
};

use ndarray::prelude::*;
use numpy::IntoPyArray;

#[pyclass(module = "retworkx")]
pub struct PyDAG {
    pub graph: StableDiGraph<PyObject, PyObject>,
    cycle_state: algo::DfsSpace<
        NodeIndex,
        <StableDiGraph<PyObject, PyObject> as Visitable>::Map,
    >,
    pub check_cycle: bool,
    pub node_removed: bool,
}

pub type Edges<'a, E> =
    petgraph::stable_graph::Edges<'a, E, petgraph::Directed>;

impl GraphBase for PyDAG {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
}

impl NodeCount for PyDAG {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

impl GraphProp for PyDAG {
    type EdgeType = petgraph::Directed;
    fn is_directed(&self) -> bool {
        true
    }
}

impl petgraph::visit::Visitable for PyDAG {
    type Map = <StableDiGraph<PyObject, PyObject> as Visitable>::Map;
    fn visit_map(&self) -> Self::Map {
        self.graph.visit_map()
    }
    fn reset_map(&self, map: &mut Self::Map) {
        self.graph.reset_map(map)
    }
}

impl petgraph::visit::Data for PyDAG {
    type NodeWeight = PyObject;
    type EdgeWeight = PyObject;
}

impl petgraph::data::DataMap for PyDAG {
    fn node_weight(&self, id: Self::NodeId) -> Option<&Self::NodeWeight> {
        self.graph.node_weight(id)
    }
    fn edge_weight(&self, id: Self::EdgeId) -> Option<&Self::EdgeWeight> {
        self.graph.edge_weight(id)
    }
}

impl petgraph::data::DataMapMut for PyDAG {
    fn node_weight_mut(
        &mut self,
        id: Self::NodeId,
    ) -> Option<&mut Self::NodeWeight> {
        self.graph.node_weight_mut(id)
    }
    fn edge_weight_mut(
        &mut self,
        id: Self::EdgeId,
    ) -> Option<&mut Self::EdgeWeight> {
        self.graph.edge_weight_mut(id)
    }
}

impl<'a> IntoNeighbors for &'a PyDAG {
    type Neighbors = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors(self, n: NodeIndex) -> Self::Neighbors {
        self.graph.neighbors(n)
    }
}

impl<'a> IntoNeighborsDirected for &'a PyDAG {
    type NeighborsDirected = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors_directed(
        self,
        n: NodeIndex,
        d: petgraph::Direction,
    ) -> Self::Neighbors {
        self.graph.neighbors_directed(n, d)
    }
}

impl<'a> IntoEdgeReferences for &'a PyDAG {
    type EdgeRef = petgraph::stable_graph::EdgeReference<'a, PyObject>;
    type EdgeReferences = petgraph::stable_graph::EdgeReferences<'a, PyObject>;
    fn edge_references(self) -> Self::EdgeReferences {
        self.graph.edge_references()
    }
}

impl<'a> IntoEdges for &'a PyDAG {
    type Edges = Edges<'a, PyObject>;
    fn edges(self, a: Self::NodeId) -> Self::Edges {
        self.graph.edges(a)
    }
}

impl<'a> IntoEdgesDirected for &'a PyDAG {
    type EdgesDirected = Edges<'a, PyObject>;
    fn edges_directed(
        self,
        a: Self::NodeId,
        dir: petgraph::Direction,
    ) -> Self::EdgesDirected {
        self.graph.edges_directed(a, dir)
    }
}

impl<'a> IntoNodeIdentifiers for &'a PyDAG {
    type NodeIdentifiers = petgraph::stable_graph::NodeIndices<'a, PyObject>;
    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.graph.node_identifiers()
    }
}

impl<'a> IntoNodeReferences for &'a PyDAG {
    type NodeRef = (NodeIndex, &'a PyObject);
    type NodeReferences = petgraph::stable_graph::NodeReferences<'a, PyObject>;
    fn node_references(self) -> Self::NodeReferences {
        self.graph.node_references()
    }
}

impl NodeIndexable for PyDAG {
    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }
    fn to_index(&self, ix: NodeIndex) -> usize {
        self.graph.to_index(ix)
    }
    fn from_index(&self, ix: usize) -> Self::NodeId {
        self.graph.from_index(ix)
    }
}

impl NodeCompactIndexable for PyDAG {}

impl Index<NodeIndex> for PyDAG {
    type Output = PyObject;
    fn index(&self, index: NodeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<NodeIndex> for PyDAG {
    fn index_mut(&mut self, index: NodeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl Index<EdgeIndex> for PyDAG {
    type Output = PyObject;
    fn index(&self, index: EdgeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<EdgeIndex> for PyDAG {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl GetAdjacencyMatrix for PyDAG {
    type AdjMatrix =
        <StableDiGraph<PyObject, PyObject> as GetAdjacencyMatrix>::AdjMatrix;
    fn adjacency_matrix(&self) -> Self::AdjMatrix {
        self.graph.adjacency_matrix()
    }
    fn is_adjacent(
        &self,
        matrix: &Self::AdjMatrix,
        a: NodeIndex,
        b: NodeIndex,
    ) -> bool {
        self.graph.is_adjacent(matrix, a, b)
    }
}

#[pymethods]
impl PyDAG {
    #[new]
    #[args(check_cycle = "false")]
    fn new(check_cycle: bool) -> Self {
        PyDAG {
            graph: StableDiGraph::<PyObject, PyObject>::new(),
            cycle_state: algo::DfsSpace::default(),
            check_cycle,
            node_removed: false,
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let out_dict = PyDict::new(py);
        let node_dict = PyDict::new(py);
        let mut out_list: Vec<PyObject> = Vec::new();
        out_dict.set_item("nodes", node_dict)?;

        let dir = petgraph::Direction::Incoming;
        for node_index in self.graph.node_indices() {
            let node_data = self.graph.node_weight(node_index).unwrap();
            node_dict.set_item(node_index.index(), node_data)?;
            for edge in self.graph.edges_directed(node_index, dir) {
                let edge_w = edge.weight();
                let triplet =
                    (edge.source().index(), edge.target().index(), edge_w)
                        .to_object(py);
                out_list.push(triplet);
            }
        }
        let py_out_list: PyObject = PyList::new(py, out_list).into();
        out_dict.set_item("edges", py_out_list)?;
        Ok(out_dict.into())
    }

    fn __setstate__(&mut self, state: PyObject) -> PyResult<()> {
        let mut node_mapping: HashMap<usize, NodeIndex> = HashMap::new();
        self.graph = StableDiGraph::<PyObject, PyObject>::new();
        let gil = Python::acquire_gil();
        let py = gil.python();
        let dict_state = state.cast_as::<PyDict>(py)?;

        let nodes_dict =
            dict_state.get_item("nodes").unwrap().downcast::<PyDict>()?;
        let edges_list =
            dict_state.get_item("edges").unwrap().downcast::<PyList>()?;
        for raw_index in nodes_dict.keys().iter() {
            let tmp_index = raw_index.downcast::<PyLong>()?;
            let index: usize = tmp_index.extract()?;
            let raw_data = nodes_dict.get_item(index).unwrap();
            let node_index = self.graph.add_node(raw_data.into());
            node_mapping.insert(index, node_index);
        }
        for raw_edge in edges_list.iter() {
            let edge = raw_edge.downcast::<PyTuple>()?;
            let raw_p_index = edge.get_item(0).downcast::<PyLong>()?;
            let tmp_p_index: usize = raw_p_index.extract()?;
            let raw_c_index = edge.get_item(1).downcast::<PyLong>()?;
            let tmp_c_index: usize = raw_c_index.extract()?;
            let edge_data = edge.get_item(2);

            let p_index = node_mapping.get(&tmp_p_index).unwrap();
            let c_index = node_mapping.get(&tmp_c_index).unwrap();
            self.graph.add_edge(*p_index, *c_index, edge_data.into());
        }
        Ok(())
    }

    #[getter]
    fn get_check_cycle(&self) -> PyResult<bool> {
        Ok(self.check_cycle)
    }

    #[setter]
    fn set_check_cycle(&mut self, value: bool) -> PyResult<()> {
        if !self.check_cycle && value && !is_directed_acyclic_graph(self) {
            return Err(DAGHasCycle::py_err("PyDAG object has a cycle"));
        }
        self.check_cycle = value;
        Ok(())
    }

    pub fn edges(&self, py: Python) -> PyObject {
        let raw_edges = self.graph.edge_indices();
        let mut out: Vec<&PyObject> = Vec::new();
        for edge in raw_edges {
            out.push(self.graph.edge_weight(edge).unwrap());
        }
        PyList::new(py, out).into()
    }

    pub fn nodes(&self, py: Python) -> PyObject {
        let raw_nodes = self.graph.node_indices();
        let mut out: Vec<&PyObject> = Vec::new();
        for node in raw_nodes {
            out.push(self.graph.node_weight(node).unwrap());
        }
        PyList::new(py, out).into()
    }

    pub fn node_indexes(&self, py: Python) -> PyObject {
        let mut out_list: Vec<usize> = Vec::new();
        for node_index in self.graph.node_indices() {
            out_list.push(node_index.index());
        }
        PyList::new(py, out_list).into()
    }

    pub fn has_edge(&self, node_a: usize, node_b: usize) -> bool {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        self.graph.find_edge(index_a, index_b).is_some()
    }

    pub fn successors(&self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let children = self
            .graph
            .neighbors_directed(index, petgraph::Direction::Outgoing);
        let mut succesors: Vec<&PyObject> = Vec::new();
        let mut used_indexes: HashSet<NodeIndex> = HashSet::new();
        for succ in children {
            if !used_indexes.contains(&succ) {
                succesors.push(self.graph.node_weight(succ).unwrap());
                used_indexes.insert(succ);
            }
        }
        Ok(PyList::new(py, succesors).into())
    }

    pub fn predecessors(&self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let parents = self
            .graph
            .neighbors_directed(index, petgraph::Direction::Incoming);
        let mut predec: Vec<&PyObject> = Vec::new();
        let mut used_indexes: HashSet<NodeIndex> = HashSet::new();
        for pred in parents {
            if !used_indexes.contains(&pred) {
                predec.push(self.graph.node_weight(pred).unwrap());
                used_indexes.insert(pred);
            }
        }
        Ok(PyList::new(py, predec).into())
    }

    pub fn get_edge_data(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<&PyObject> {
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

    pub fn get_node_data(&self, node: usize) -> PyResult<&PyObject> {
        let index = NodeIndex::new(node);
        let node = match self.graph.node_weight(index) {
            Some(node) => node,
            None => return Err(IndexError::py_err("No node found for index")),
        };
        Ok(node)
    }

    pub fn get_all_edge_data(
        &self,
        py: Python,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<PyObject> {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        let raw_edges = self
            .graph
            .edges_directed(index_a, petgraph::Direction::Outgoing);
        let mut out: Vec<&PyObject> = Vec::new();
        for edge in raw_edges {
            if edge.target() == index_b {
                out.push(edge.weight());
            }
        }
        if out.is_empty() {
            Err(NoEdgeBetweenNodes::py_err("No edge found between nodes"))
        } else {
            Ok(PyList::new(py, out).into())
        }
    }

    pub fn remove_node(&mut self, node: usize) -> PyResult<()> {
        let index = NodeIndex::new(node);
        self.graph.remove_node(index);
        self.node_removed = true;
        Ok(())
    }

    pub fn add_edge(
        &mut self,
        parent: usize,
        child: usize,
        edge: PyObject,
    ) -> PyResult<usize> {
        let p_index = NodeIndex::new(parent);
        let c_index = NodeIndex::new(child);
        if self.check_cycle {
            let should_check_for_cycle =
                must_check_for_cycle(self, p_index, c_index);
            let state = Some(&mut self.cycle_state);
            if should_check_for_cycle
                && algo::has_path_connecting(
                    &self.graph,
                    c_index,
                    p_index,
                    state,
                )
            {
                Err(DAGWouldCycle::py_err("Adding an edge would cycle"))
            } else {
                let edge = self.graph.add_edge(p_index, c_index, edge);
                Ok(edge.index())
            }
        } else {
            let edge = self.graph.add_edge(p_index, c_index, edge);
            Ok(edge.index())
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

    pub fn add_node(&mut self, obj: PyObject) -> PyResult<usize> {
        let index = self.graph.add_node(obj);
        Ok(index.index())
    }

    pub fn add_child(
        &mut self,
        parent: usize,
        obj: PyObject,
        edge: PyObject,
    ) -> PyResult<usize> {
        let index = NodeIndex::new(parent);
        let child_node = self.graph.add_node(obj);
        self.graph.add_edge(index, child_node, edge);
        Ok(child_node.index())
    }

    pub fn add_parent(
        &mut self,
        child: usize,
        obj: PyObject,
        edge: PyObject,
    ) -> PyResult<usize> {
        let index = NodeIndex::new(child);
        let parent_node = self.graph.add_node(obj);
        self.graph.add_edge(parent_node, index, edge);
        Ok(parent_node.index())
    }

    pub fn adj(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.neighbors(index);
        let out_dict = PyDict::new(py);
        for neighbor in neighbors {
            let mut edge = self.graph.find_edge(index, neighbor);
            // If there is no edge then it must be a parent neighbor
            if edge.is_none() {
                edge = self.graph.find_edge(neighbor, index);
            }
            let edge_w = self.graph.edge_weight(edge.unwrap());
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
        let dir = if direction {
            petgraph::Direction::Incoming
        } else {
            petgraph::Direction::Outgoing
        };
        let neighbors = self.graph.neighbors_directed(index, dir);
        let out_dict = PyDict::new(py);
        for neighbor in neighbors {
            let edge = if direction {
                match self.graph.find_edge(neighbor, index) {
                    Some(edge) => edge,
                    None => {
                        return Err(NoEdgeBetweenNodes::py_err(
                            "No edge found between nodes",
                        ))
                    }
                }
            } else {
                match self.graph.find_edge(index, neighbor) {
                    Some(edge) => edge,
                    None => {
                        return Err(NoEdgeBetweenNodes::py_err(
                            "No edge found between nodes",
                        ))
                    }
                }
            };
            let edge_w = self.graph.edge_weight(edge);
            out_dict.set_item(neighbor.index(), edge_w)?;
        }
        Ok(out_dict.into())
    }

    pub fn in_edges(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Incoming;
        let mut out_list: Vec<PyObject> = Vec::new();
        let raw_edges = self.graph.edges_directed(index, dir);
        for edge in raw_edges {
            let edge_w = edge.weight();
            let triplet = (edge.source().index(), node, edge_w).to_object(py);
            out_list.push(triplet)
        }
        Ok(PyList::new(py, out_list).into())
    }

    pub fn out_edges(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let mut out_list: Vec<PyObject> = Vec::new();
        let raw_edges = self.graph.edges_directed(index, dir);
        for edge in raw_edges {
            let edge_w = edge.weight();
            let triplet = (node, edge.target().index(), edge_w).to_object(py);
            out_list.push(triplet)
        }
        Ok(PyList::new(py, out_list).into())
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
        let neighbors = self.graph.edges_directed(index, dir);
        neighbors.count()
    }

    pub fn out_degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let neighbors = self.graph.edges_directed(index, dir);
        neighbors.count()
    }

    pub fn find_adjacent_node_by_edge(
        &self,
        py: Python,
        node: usize,
        predicate: PyObject,
    ) -> PyResult<&PyObject> {
        let predicate_callable = |a: &PyObject| -> PyResult<PyObject> {
            let res = predicate.call1(py, (a,))?;
            Ok(res.to_object(py))
        };
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let edges = self.graph.edges_directed(index, dir);
        for edge in edges {
            let edge_predicate_raw = predicate_callable(&edge.weight())?;
            let edge_predicate: bool = edge_predicate_raw.extract(py)?;
            if edge_predicate {
                return Ok(self.graph.node_weight(edge.target()).unwrap());
            }
        }
        return Err(NoSuitableNeighbors::py_err("No suitable neighbor"));
    }
}

#[pyproto]
impl PyMappingProtocol for PyDAG {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }
}

fn must_check_for_cycle(dag: &PyDAG, a: NodeIndex, b: NodeIndex) -> bool {
    let mut parents_a = dag
        .graph
        .neighbors_directed(a, petgraph::Direction::Incoming);
    let mut children_b = dag
        .graph
        .neighbors_directed(b, petgraph::Direction::Outgoing);
    parents_a.next().is_some()
        && children_b.next().is_some()
        && dag.graph.find_edge(a, b).is_none()
}

fn longest_path(graph: &PyDAG) -> PyResult<Vec<usize>> {
    let dag = &graph.graph;
    let mut path: Vec<usize> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::py_err("Sort encountered a cycle"))
        }
    };
    if nodes.is_empty() {
        return Ok(path);
    }
    let mut dist: HashMap<NodeIndex, (usize, NodeIndex)> = HashMap::new();
    for node in nodes {
        let parents =
            dag.neighbors_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(usize, NodeIndex)> = Vec::new();
        for p_node in parents {
            let length = dist[&p_node].0 + 1;
            us.push((length, p_node));
        }
        let maxu: (usize, NodeIndex);
        if !us.is_empty() {
            maxu = *us.iter().max_by_key(|x| x.0).unwrap();
        } else {
            maxu = (0, node);
        };
        dist.insert(node, maxu);
    }
    let first = match dist.keys().max_by_key(|index| dist[index]) {
        Some(first) => first,
        None => {
            return Err(Exception::py_err("Encountered something unexpected"))
        }
    };
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
    Ok(path)
}

#[pyfunction]
fn dag_longest_path(py: Python, graph: &PyDAG) -> PyResult<PyObject> {
    let path = longest_path(graph)?;
    Ok(PyList::new(py, path).into())
}

#[pyfunction]
fn dag_longest_path_length(graph: &PyDAG) -> PyResult<usize> {
    let path = longest_path(graph)?;
    if path.is_empty() {
        return Ok(0);
    }
    let path_length: usize = path.len() - 1;
    Ok(path_length)
}

#[pyfunction]
fn number_weakly_connected_components(graph: &PyDAG) -> usize {
    algo::connected_components(graph)
}

#[pyfunction]
fn is_directed_acyclic_graph(graph: &PyDAG) -> bool {
    let cycle_detected = algo::is_cyclic_directed(graph);
    !cycle_detected
}

#[pyfunction]
fn is_isomorphic(first: &PyDAG, second: &PyDAG) -> PyResult<bool> {
    let res = dag_isomorphism::is_isomorphic(first, second)?;
    Ok(res)
}

#[pyfunction]
fn is_isomorphic_node_match(
    py: Python,
    first: &PyDAG,
    second: &PyDAG,
    matcher: PyObject,
) -> PyResult<bool> {
    let compare_nodes = |a: &PyObject, b: &PyObject| -> PyResult<bool> {
        let res = matcher.call1(py, (a, b))?;
        Ok(res.is_true(py).unwrap())
    };

    fn compare_edges(_a: &PyObject, _b: &PyObject) -> PyResult<bool> {
        Ok(true)
    }
    let res = dag_isomorphism::is_isomorphic_matching(
        first,
        second,
        compare_nodes,
        compare_edges,
    )?;
    Ok(res)
}

#[pyfunction]
fn topological_sort(py: Python, graph: &PyDAG) -> PyResult<PyObject> {
    let nodes = match algo::toposort(graph, None) {
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

#[pyfunction]
fn bfs_successors(
    py: Python,
    graph: &PyDAG,
    node: usize,
) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut bfs = Bfs::new(graph, index);
    let mut out_list: Vec<(&PyObject, Vec<&PyObject>)> = Vec::new();
    while let Some(nx) = bfs.next(graph) {
        let children = graph
            .graph
            .neighbors_directed(nx, petgraph::Direction::Outgoing);
        let mut succesors: Vec<&PyObject> = Vec::new();
        for succ in children {
            succesors.push(graph.graph.node_weight(succ).unwrap());
        }
        if !succesors.is_empty() {
            out_list.push((graph.graph.node_weight(nx).unwrap(), succesors));
        }
    }
    Ok(PyList::new(py, out_list).into())
}

#[pyfunction]
fn ancestors(py: Python, graph: &PyDAG, node: usize) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let reverse_graph = Reversed(graph);
    let res = algo::dijkstra(reverse_graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    Ok(out_set.to_object(py))
}

#[pyfunction]
fn descendants(py: Python, graph: &PyDAG, node: usize) -> PyResult<PyObject> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let res = algo::dijkstra(graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    Ok(out_set.to_object(py))
}

#[pyfunction]
fn lexicographical_topological_sort(
    py: Python,
    dag: &PyDAG,
    key: PyObject,
) -> PyResult<PyObject> {
    let key_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = key.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    // HashMap of node_index indegree
    let mut in_degree_map: HashMap<NodeIndex, usize> = HashMap::new();
    for node in dag.graph.node_indices() {
        in_degree_map.insert(node, dag.in_degree(node.index()));
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        key: String,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &State) -> Ordering {
            // Notice that the we flip the ordering on costs.
            // In case of a tie we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.index().cmp(&self.node.index()))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &State) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut zero_indegree = BinaryHeap::new();
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key_raw = key_callable(&dag.graph[*node])?;
            let map_key: String = map_key_raw.extract(py)?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<&PyObject> = Vec::new();
    let dir = petgraph::Direction::Outgoing;
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.graph.neighbors_directed(node, dir);
        for child in neighbors {
            let child_degree = in_degree_map.get_mut(&child).unwrap();
            *child_degree -= 1;
            if *child_degree == 0 {
                let map_key_raw = key_callable(&dag.graph[child])?;
                let map_key: String = map_key_raw.extract(py)?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            }
        }
        out_list.push(&dag.graph[node])
    }
    Ok(PyList::new(py, out_list).into())
}

// Find the shortest path lengths between all pairs of nodes using Floyd's
// algorithm
// Note: Edge weights are assumed to be 1
#[pyfunction]
fn floyd_warshall(py: Python, dag: &PyDAG) -> PyResult<PyObject> {
    let mut dist: HashMap<(usize, usize), usize> = HashMap::new();
    for node in dag.graph.node_indices() {
        // Distance from a node to itself is zero
        dist.insert((node.index(), node.index()), 0);
    }
    for edge in dag.graph.edge_indices() {
        // Distance between nodes that share an edge is 1
        let source_target = dag.graph.edge_endpoints(edge).unwrap();
        let u = source_target.0.index();
        let v = source_target.1.index();
        // Update dist only if the key hasn't been set to 0 already
        // (i.e. in case edge is a self edge). Assumes edge weight = 1.
        dist.entry((u, v)).or_insert(1);
    }
    // The shortest distance between any pair of nodes u, v is the min of the
    // distance tracked so far from u->v and the distance from u to v thorough
    // another node w, for any w.
    for w in dag.graph.node_indices() {
        for u in dag.graph.node_indices() {
            for v in dag.graph.node_indices() {
                let u_v_dist = match dist.get(&(u.index(), v.index())) {
                    Some(u_v_dist) => *u_v_dist,
                    None => std::usize::MAX,
                };
                let u_w_dist = match dist.get(&(u.index(), w.index())) {
                    Some(u_w_dist) => *u_w_dist,
                    None => std::usize::MAX,
                };
                let w_v_dist = match dist.get(&(w.index(), v.index())) {
                    Some(w_v_dist) => *w_v_dist,
                    None => std::usize::MAX,
                };
                if u_w_dist == std::usize::MAX || w_v_dist == std::usize::MAX {
                    // Avoid overflow!
                    continue;
                }
                if u_v_dist > u_w_dist + w_v_dist {
                    dist.insert((u.index(), v.index()), u_w_dist + w_v_dist);
                }
            }
        }
    }

    // Some re-formatting for Python: Dict[int, Dict[int, int]]
    let out_dict = PyDict::new(py);
    for (nodes, distance) in dist {
        let u_index = nodes.0;
        let v_index = nodes.1;
        if out_dict.contains(u_index)? {
            let u_dict =
                out_dict.get_item(u_index).unwrap().downcast::<PyDict>()?;
            u_dict.set_item(v_index, distance)?;
            out_dict.set_item(u_index, u_dict)?;
        } else {
            let u_dict = PyDict::new(py);
            u_dict.set_item(v_index, distance)?;
            out_dict.set_item(u_index, u_dict)?;
        }
    }
    Ok(out_dict.into())
}

#[pyfunction]
fn layers(
    py: Python,
    dag: &PyDAG,
    first_layer: Vec<usize>,
) -> PyResult<PyObject> {
    let mut output: Vec<Vec<&PyObject>> = Vec::new();
    // Convert usize to NodeIndex
    let mut first_layer_index: Vec<NodeIndex> = Vec::new();
    for index in first_layer {
        first_layer_index.push(NodeIndex::new(index));
    }

    let mut cur_layer = first_layer_index;
    let mut next_layer: Vec<NodeIndex> = Vec::new();
    let mut predecessor_count: HashMap<NodeIndex, usize> = HashMap::new();

    let mut layer_node_data: Vec<&PyObject> = Vec::new();
    for layer_node in &cur_layer {
        layer_node_data.push(&dag[*layer_node]);
    }
    output.push(layer_node_data);

    // Iterate until there are no more
    while !cur_layer.is_empty() {
        for node in &cur_layer {
            let children = dag
                .graph
                .neighbors_directed(*node, petgraph::Direction::Outgoing);
            let mut used_indexes: HashSet<NodeIndex> = HashSet::new();
            for succ in children {
                // Skip duplicate successors
                if used_indexes.contains(&succ) {
                    continue;
                }
                used_indexes.insert(succ);
                let mut multiplicity: usize = 0;
                let raw_edges = dag
                    .graph
                    .edges_directed(*node, petgraph::Direction::Outgoing);
                for edge in raw_edges {
                    if edge.target() == succ {
                        multiplicity += 1;
                    }
                }
                predecessor_count
                    .entry(succ)
                    .and_modify(|e| *e -= multiplicity)
                    .or_insert(dag.in_degree(succ.index()) - multiplicity);
                if *predecessor_count.get(&succ).unwrap() == 0 {
                    next_layer.push(succ);
                    predecessor_count.remove(&succ);
                }
            }
        }
        let mut layer_node_data: Vec<&PyObject> = Vec::new();
        for layer_node in &next_layer {
            layer_node_data.push(&dag[*layer_node]);
        }
        if !layer_node_data.is_empty() {
            output.push(layer_node_data);
        }
        cur_layer = next_layer;
        next_layer = Vec::new();
    }
    Ok(PyList::new(py, output).into())
}

#[pyfunction]
fn dag_adjacency_matrix(
    py: Python,
    graph: &PyDAG,
    weight_fn: PyObject,
) -> PyResult<PyObject> {
    let node_map: Option<HashMap<NodeIndex, usize>>;
    let n: usize;
    if graph.node_removed {
        let mut node_hash_map: HashMap<NodeIndex, usize> = HashMap::new();
        let mut count = 0;
        for node in graph.graph.node_indices() {
            node_hash_map.insert(node, count);
            count += 1;
        }
        n = count;
        node_map = Some(node_hash_map);
    } else {
        n = graph.graph.node_bound();
        node_map = None;
    }
    let mut matrix = Array::<f64, _>::zeros((n, n).f());

    let weight_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = weight_fn.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    for edge in graph.graph.edge_references() {
        let edge_weight_raw = weight_callable(&edge.weight())?;
        let edge_weight: f64 = edge_weight_raw.extract(py)?;
        let source = edge.source();
        let target = edge.target();
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                i = *map.get(&source).unwrap();
                j = *map.get(&target).unwrap();
            }
            None => {
                i = source.index();
                j = target.index();
            }
        }
        matrix[[i, j]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

#[pyfunction]
fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
) -> PyResult<PyObject> {
    let node_map: Option<HashMap<NodeIndex, usize>>;
    let n: usize;
    if graph.node_removed {
        let mut node_hash_map: HashMap<NodeIndex, usize> = HashMap::new();
        let mut count = 0;
        for node in graph.graph.node_indices() {
            node_hash_map.insert(node, count);
            count += 1;
        }
        n = count;
        node_map = Some(node_hash_map);
    } else {
        n = graph.graph.node_bound();
        node_map = None;
    }
    let mut matrix = Array::<f64, _>::zeros((n, n).f());

    let weight_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = weight_fn.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    for edge in graph.graph.edge_references() {
        let edge_weight_raw = weight_callable(&edge.weight())?;
        let edge_weight: f64 = edge_weight_raw.extract(py)?;
        let source = edge.source();
        let target = edge.target();
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                i = *map.get(&source).unwrap();
                j = *map.get(&target).unwrap();
            }
            None => {
                i = source.index();
                j = target.index();
            }
        }
        matrix[[i, j]] += edge_weight;
        matrix[[j, i]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

#[pymodule]
fn retworkx(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pyfunction!(bfs_successors))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_directed_acyclic_graph))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(is_isomorphic_node_match))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(layers))?;
    m.add_wrapped(wrap_pyfunction!(dag_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_adjacency_matrix))?;
    m.add_class::<PyDAG>()?;
    m.add_class::<graph::PyGraph>()?;
    Ok(())
}

create_exception!(retworkx, DAGWouldCycle, Exception);
create_exception!(retworkx, NoEdgeBetweenNodes, Exception);
create_exception!(retworkx, DAGHasCycle, Exception);
create_exception!(retworkx, NoSuitableNeighbors, Exception);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
