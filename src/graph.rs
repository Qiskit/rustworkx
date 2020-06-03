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

use std::ops::{Index, IndexMut};

use pyo3::class::PyMappingProtocol;
use pyo3::exceptions::IndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Python;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::{
    GetAdjacencyMatrix, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoNeighbors, IntoNodeIdentifiers, IntoNodeReferences,
    NodeCompactIndexable, NodeCount, NodeIndexable, Visitable,
};

use super::NoEdgeBetweenNodes;

#[pyclass(module = "retworkx")]
pub struct PyGraph {
    pub graph: StableUnGraph<PyObject, PyObject>,
}

pub type Edges<'a, E> =
    petgraph::stable_graph::Edges<'a, E, petgraph::Undirected>;

impl GraphBase for PyGraph {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
}

impl NodeCount for PyGraph {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

impl GraphProp for PyGraph {
    type EdgeType = petgraph::Undirected;
    fn is_directed(&self) -> bool {
        false
    }
}

impl petgraph::visit::Visitable for PyGraph {
    type Map = <StableUnGraph<PyObject, PyObject> as Visitable>::Map;
    fn visit_map(&self) -> Self::Map {
        self.graph.visit_map()
    }
    fn reset_map(&self, map: &mut Self::Map) {
        self.graph.reset_map(map)
    }
}

impl petgraph::visit::Data for PyGraph {
    type NodeWeight = PyObject;
    type EdgeWeight = PyObject;
}

impl petgraph::data::DataMap for PyGraph {
    fn node_weight(&self, id: Self::NodeId) -> Option<&Self::NodeWeight> {
        self.graph.node_weight(id)
    }
    fn edge_weight(&self, id: Self::EdgeId) -> Option<&Self::EdgeWeight> {
        self.graph.edge_weight(id)
    }
}

impl petgraph::data::DataMapMut for PyGraph {
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

impl<'a> IntoNeighbors for &'a PyGraph {
    type Neighbors = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors(self, n: NodeIndex) -> Self::Neighbors {
        self.graph.neighbors(n)
    }
}

impl<'a> IntoEdgeReferences for &'a PyGraph {
    type EdgeRef = petgraph::stable_graph::EdgeReference<'a, PyObject>;
    type EdgeReferences = petgraph::stable_graph::EdgeReferences<'a, PyObject>;
    fn edge_references(self) -> Self::EdgeReferences {
        self.graph.edge_references()
    }
}

impl<'a> IntoEdges for &'a PyGraph {
    type Edges = Edges<'a, PyObject>;
    fn edges(self, a: Self::NodeId) -> Self::Edges {
        self.graph.edges(a)
    }
}

impl<'a> IntoNodeIdentifiers for &'a PyGraph {
    type NodeIdentifiers = petgraph::stable_graph::NodeIndices<'a, PyObject>;
    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.graph.node_identifiers()
    }
}

impl<'a> IntoNodeReferences for &'a PyGraph {
    type NodeRef = (NodeIndex, &'a PyObject);
    type NodeReferences = petgraph::stable_graph::NodeReferences<'a, PyObject>;
    fn node_references(self) -> Self::NodeReferences {
        self.graph.node_references()
    }
}

impl NodeIndexable for PyGraph {
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

impl NodeCompactIndexable for PyGraph {}

impl Index<NodeIndex> for PyGraph {
    type Output = PyObject;
    fn index(&self, index: NodeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<NodeIndex> for PyGraph {
    fn index_mut(&mut self, index: NodeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl Index<EdgeIndex> for PyGraph {
    type Output = PyObject;
    fn index(&self, index: EdgeIndex) -> &PyObject {
        &self.graph[index]
    }
}

impl IndexMut<EdgeIndex> for PyGraph {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut PyObject {
        &mut self.graph[index]
    }
}

impl GetAdjacencyMatrix for PyGraph {
    type AdjMatrix =
        <StableUnGraph<PyObject, PyObject> as GetAdjacencyMatrix>::AdjMatrix;
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
impl PyGraph {
    #[new]
    fn new() -> Self {
        PyGraph {
            graph: StableUnGraph::<PyObject, PyObject>::default(),
        }
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
        let raw_edges = self.graph.edges(index_a);
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

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        node_a: usize,
        node_b: usize,
        edge: PyObject,
    ) -> PyResult<usize> {
        let p_index = NodeIndex::new(node_a);
        let c_index = NodeIndex::new(node_b);
        let edge = self.graph.add_edge(p_index, c_index, edge);
        Ok(edge.index())
    }

    pub fn remove_edge(
        &mut self,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<()> {
        let p_index = NodeIndex::new(node_a);
        let c_index = NodeIndex::new(node_b);
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

    pub fn adj(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.neighbors(index);
        let out_dict = PyDict::new(py);
        for neighbor in neighbors {
            let edge = self.graph.find_edge(index, neighbor);
            let edge_w = self.graph.edge_weight(edge.unwrap());
            out_dict.set_item(neighbor.index(), edge_w)?;
        }
        Ok(out_dict.into())
    }

    pub fn degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.edges(index);
        neighbors.count()
    }
}

#[pyproto]
impl PyMappingProtocol for PyGraph {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }
}
