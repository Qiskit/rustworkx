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

use std::collections::HashSet;
use std::ops::{Index, IndexMut};

use hashbrown::HashMap;

use pyo3::class::PyMappingProtocol;
use pyo3::exceptions::IndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyLong, PyTuple};
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{
    GetAdjacencyMatrix, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected,
    IntoNodeIdentifiers, IntoNodeReferences, NodeCompactIndexable, NodeCount,
    NodeIndexable, Visitable,
};

use super::{
    is_directed_acyclic_graph, DAGHasCycle, DAGWouldCycle, NoEdgeBetweenNodes,
    NoSuitableNeighbors,
};

#[pyclass(module = "retworkx", subclass)]
pub struct PyDiGraph {
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

impl GraphBase for PyDiGraph {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
}

impl NodeCount for PyDiGraph {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

impl GraphProp for PyDiGraph {
    type EdgeType = petgraph::Directed;
    fn is_directed(&self) -> bool {
        true
    }
}

impl petgraph::visit::Visitable for PyDiGraph {
    type Map = <StableDiGraph<PyObject, PyObject> as Visitable>::Map;
    fn visit_map(&self) -> Self::Map {
        self.graph.visit_map()
    }
    fn reset_map(&self, map: &mut Self::Map) {
        self.graph.reset_map(map)
    }
}

impl petgraph::visit::Data for PyDiGraph {
    type NodeWeight = PyObject;
    type EdgeWeight = PyObject;
}

impl petgraph::data::DataMap for PyDiGraph {
    fn node_weight(&self, id: Self::NodeId) -> Option<&Self::NodeWeight> {
        self.graph.node_weight(id)
    }
    fn edge_weight(&self, id: Self::EdgeId) -> Option<&Self::EdgeWeight> {
        self.graph.edge_weight(id)
    }
}

impl petgraph::data::DataMapMut for PyDiGraph {
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

impl<'a> IntoNeighbors for &'a PyDiGraph {
    type Neighbors = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors(self, n: NodeIndex) -> Self::Neighbors {
        self.graph.neighbors(n)
    }
}

impl<'a> IntoNeighborsDirected for &'a PyDiGraph {
    type NeighborsDirected = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors_directed(
        self,
        n: NodeIndex,
        d: petgraph::Direction,
    ) -> Self::Neighbors {
        self.graph.neighbors_directed(n, d)
    }
}

impl<'a> IntoEdgeReferences for &'a PyDiGraph {
    type EdgeRef = petgraph::stable_graph::EdgeReference<'a, PyObject>;
    type EdgeReferences = petgraph::stable_graph::EdgeReferences<'a, PyObject>;
    fn edge_references(self) -> Self::EdgeReferences {
        self.graph.edge_references()
    }
}

impl<'a> IntoEdges for &'a PyDiGraph {
    type Edges = Edges<'a, PyObject>;
    fn edges(self, a: Self::NodeId) -> Self::Edges {
        self.graph.edges(a)
    }
}

impl<'a> IntoEdgesDirected for &'a PyDiGraph {
    type EdgesDirected = Edges<'a, PyObject>;
    fn edges_directed(
        self,
        a: Self::NodeId,
        dir: petgraph::Direction,
    ) -> Self::EdgesDirected {
        self.graph.edges_directed(a, dir)
    }
}

impl<'a> IntoNodeIdentifiers for &'a PyDiGraph {
    type NodeIdentifiers = petgraph::stable_graph::NodeIndices<'a, PyObject>;
    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.graph.node_identifiers()
    }
}

impl<'a> IntoNodeReferences for &'a PyDiGraph {
    type NodeRef = (NodeIndex, &'a PyObject);
    type NodeReferences = petgraph::stable_graph::NodeReferences<'a, PyObject>;
    fn node_references(self) -> Self::NodeReferences {
        self.graph.node_references()
    }
}

impl NodeIndexable for PyDiGraph {
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

impl NodeCompactIndexable for PyDiGraph {}

impl Index<NodeIndex> for PyDiGraph {
    type Output = PyObject;
    fn index(&self, index: NodeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<NodeIndex> for PyDiGraph {
    fn index_mut(&mut self, index: NodeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl Index<EdgeIndex> for PyDiGraph {
    type Output = PyObject;
    fn index(&self, index: EdgeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<EdgeIndex> for PyDiGraph {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl GetAdjacencyMatrix for PyDiGraph {
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

// Rust side only PyDiGraph methods
impl PyDiGraph {
    fn _add_edge(
        &mut self,
        p_index: NodeIndex,
        c_index: NodeIndex,
        edge: PyObject,
    ) -> PyResult<usize> {
        // Only check for cycles if instance attribute is set to true
        if self.check_cycle {
            // Only check for a cycle (by running has_path_connecting) if
            // the new edge could potentially add a cycle
            let cycle_check_required =
                is_cycle_check_required(self, p_index, c_index);
            let state = Some(&mut self.cycle_state);
            if cycle_check_required
                && algo::has_path_connecting(
                    &self.graph,
                    c_index,
                    p_index,
                    state,
                )
            {
                return Err(DAGWouldCycle::py_err(
                    "Adding an edge would cycle",
                ));
            }
        }
        let edge = self.graph.add_edge(p_index, c_index, edge);
        Ok(edge.index())
    }
}

#[pymethods]
impl PyDiGraph {
    #[new]
    #[args(check_cycle = "false")]
    fn new(check_cycle: bool) -> Self {
        PyDiGraph {
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
            return Err(DAGHasCycle::py_err("PyDiGraph object has a cycle"));
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
        let out_index = self._add_edge(p_index, c_index, edge)?;
        Ok(out_index)
    }

    pub fn add_edges_from(
        &mut self,
        obj_list: Vec<(usize, usize, PyObject)>,
    ) -> PyResult<Vec<usize>> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            let edge = self._add_edge(p_index, c_index, obj.2)?;
            out_list.push(edge);
        }
        Ok(out_list)
    }

    pub fn add_edges_from_no_data(
        &mut self,
        py: Python,
        obj_list: Vec<(usize, usize)>,
    ) -> PyResult<Vec<usize>> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            let edge = self._add_edge(p_index, c_index, py.None())?;
            out_list.push(edge);
        }
        Ok(out_list)
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

    pub fn add_nodes_from(
        &mut self,
        obj_list: Vec<PyObject>,
    ) -> PyResult<Vec<usize>> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let node_index = self.graph.add_node(obj);
            out_list.push(node_index.index());
        }
        Ok(out_list)
    }

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
        Err(NoSuitableNeighbors::py_err("No suitable neighbor"))
    }
}

#[pyproto]
impl PyMappingProtocol for PyDiGraph {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }
}

fn is_cycle_check_required(
    dag: &PyDiGraph,
    a: NodeIndex,
    b: NodeIndex,
) -> bool {
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
