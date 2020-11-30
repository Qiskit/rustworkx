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

use std::cmp;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::ops::{Index, IndexMut};
use std::str;

use hashbrown::{HashMap, HashSet};

use pyo3::class::PyMappingProtocol;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyLong, PyString, PyTuple};
use pyo3::Python;

use super::dot_utils::build_dot;
use super::iterators::{EdgeList, NodeIndices, WeightedEdgeList};
use super::{NoEdgeBetweenNodes, NodesRemoved};

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::{
    GetAdjacencyMatrix, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers,
    IntoNodeReferences, NodeCompactIndexable, NodeCount, NodeFiltered,
    NodeIndexable, Visitable,
};

/// A class for creating undirected graphs
///
/// The PyGraph class is used to create an undirected graph. It can be a
/// multigraph (have multiple edges between nodes). Each node and edge
/// (although rarely used for edges) is indexed by an integer id. Additionally,
/// each node and edge contains an arbitrary Python object as a weight/data
/// payload. You can use the index for access to the data payload as in the
/// following example:
///
/// .. jupyter-execute::
///
///     import retworkx
///
///     graph = retworkx.PyGraph()
///     data_payload = "An arbitrary Python object"
///     node_index = graph.add_node(data_payload)
///     print("Node Index: %s" % node_index)
///     print(graph[node_index])
///
/// The PyDiGraph implements the Python mapping protocol for nodes so in
/// addition to access you can also update the data payload with:
///
/// .. jupyter-execute::
///
///     import retworkx
///
///     graph = retworkx.PyGraph()
///     data_payload = "An arbitrary Python object"
///     node_index = graph.add_node(data_payload)
///     graph[node_index] = "New Payload"
///     print("Node Index: %s" % node_index)
///     print(graph[node_index])
///
#[pyclass(module = "retworkx")]
#[text_signature = "()"]
pub struct PyGraph {
    pub graph: StableUnGraph<PyObject, PyObject>,
    pub node_removed: bool,
}

pub type Edges<'a, E> =
    petgraph::stable_graph::Edges<'a, E, petgraph::Undirected>;

impl GraphBase for PyGraph {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
}

impl<'a> NodesRemoved for &'a PyGraph {
    fn nodes_removed(&self) -> bool {
        self.node_removed
    }
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

impl<'a> IntoNeighborsDirected for &'a PyGraph {
    type NeighborsDirected = petgraph::stable_graph::Neighbors<'a, PyObject>;
    fn neighbors_directed(
        self,
        n: NodeIndex,
        d: petgraph::Direction,
    ) -> Self::Neighbors {
        self.graph.neighbors_directed(n, d)
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
            node_removed: false,
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let out_dict = PyDict::new(py);
        let node_dict = PyDict::new(py);
        let mut out_list: Vec<PyObject> = Vec::new();
        out_dict.set_item("nodes", node_dict)?;
        for node_index in self.graph.node_indices() {
            let node_data = self.graph.node_weight(node_index).unwrap();
            node_dict.set_item(node_index.index(), node_data)?;
        }
        for edge in self.graph.edge_indices() {
            let edge_w = self.graph.edge_weight(edge);
            let endpoints = self.graph.edge_endpoints(edge).unwrap();

            let triplet = (endpoints.0.index(), endpoints.1.index(), edge_w)
                .to_object(py);
            out_list.push(triplet);
        }
        let py_out_list: PyObject = PyList::new(py, out_list).into();
        out_dict.set_item("edges", py_out_list)?;
        Ok(out_dict.into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        self.graph = StableUnGraph::<PyObject, PyObject>::default();
        let dict_state = state.cast_as::<PyDict>(py)?;
        let nodes_dict =
            dict_state.get_item("nodes").unwrap().downcast::<PyDict>()?;
        let edges_list =
            dict_state.get_item("edges").unwrap().downcast::<PyList>()?;
        let mut node_indices: Vec<usize> = Vec::new();
        for raw_index in nodes_dict.keys() {
            let tmp_index = raw_index.downcast::<PyLong>()?;
            node_indices.push(tmp_index.extract()?);
        }
        if node_indices.is_empty() {
            return Ok(());
        }
        let max_index: usize = *node_indices.iter().max().unwrap();
        let mut tmp_nodes: Vec<NodeIndex> = Vec::new();
        let mut node_count: usize = 0;
        while max_index >= self.graph.node_bound() {
            match nodes_dict.get_item(node_count) {
                Some(raw_data) => {
                    self.graph.add_node(raw_data.into());
                }
                None => {
                    let tmp_node = self.graph.add_node(py.None());
                    tmp_nodes.push(tmp_node);
                }
            };
            node_count += 1;
        }
        for tmp_node in tmp_nodes {
            self.graph.remove_node(tmp_node);
        }
        for raw_edge in edges_list.iter() {
            let edge = raw_edge.downcast::<PyTuple>()?;
            let raw_p_index = edge.get_item(0).downcast::<PyLong>()?;
            let parent: usize = raw_p_index.extract()?;
            let p_index = NodeIndex::new(parent);
            let raw_c_index = edge.get_item(1).downcast::<PyLong>()?;
            let child: usize = raw_c_index.extract()?;
            let c_index = NodeIndex::new(child);
            let edge_data = edge.get_item(2);

            self.graph.add_edge(p_index, c_index, edge_data.into());
        }
        Ok(())
    }

    /// Return a list of all edge data.
    ///
    /// :returns: A list of all the edge data objects in the graph
    /// :rtype: list
    #[text_signature = "(self)"]
    pub fn edges(&self) -> Vec<&PyObject> {
        self.graph
            .edge_indices()
            .map(|edge| self.graph.edge_weight(edge).unwrap())
            .collect()
    }

    /// Return a list of all node data.
    ///
    /// :returns: A list of all the node data objects in the graph
    /// :rtype: list
    #[text_signature = "(self)"]
    pub fn nodes(&self) -> Vec<&PyObject> {
        self.graph
            .node_indices()
            .map(|node| self.graph.node_weight(node).unwrap())
            .collect()
    }

    /// Return a list of all node indexes.
    ///
    /// :returns: A list of all the node indexes in the graph
    /// :rtype: NodeIndices
    #[text_signature = "(self)"]
    pub fn node_indexes(&self) -> NodeIndices {
        NodeIndices {
            nodes: self.graph.node_indices().map(|node| node.index()).collect(),
        }
    }

    /// Return True if there is an edge between node_a to node_b.
    ///
    /// :param int node_a: The node index to check for an edge between
    /// :param int node_b: The node index to check for an edge between
    ///
    /// :returns: True if there is an edge false if there is no edge
    /// :rtype: bool
    #[text_signature = "(self, node_a, node_b, /)"]
    pub fn has_edge(&self, node_a: usize, node_b: usize) -> bool {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        self.graph.find_edge(index_a, index_b).is_some()
    }

    ///  Return the edge data for the edge between 2 nodes.
    ///
    ///  Note if there are multiple edges between the nodes only one will be
    ///  returned. To get all edge data objects use
    ///  :meth:`~retworkx.PyGraph.get_all_edge_data`
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node
    ///
    /// :returns: The data object set for the edge
    /// :raises NoEdgeBetweenNodes: when there is no edge between the provided
    ///     nodes
    #[text_signature = "(self, node_a, node_b, /)"]
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
                return Err(NoEdgeBetweenNodes::new_err(
                    "No edge found between nodes",
                ))
            }
        };

        let data = self.graph.edge_weight(edge_index).unwrap();
        Ok(data)
    }

    /// Return the node data for a given node index
    ///
    /// :param int node: The index for the node
    ///
    /// :returns: The data object set for that node
    /// :raises IndexError: when an invalid node index is provided
    #[text_signature = "(self, node, /)"]
    pub fn get_node_data(&self, node: usize) -> PyResult<&PyObject> {
        let index = NodeIndex::new(node);
        let node = match self.graph.node_weight(index) {
            Some(node) => node,
            None => {
                return Err(PyIndexError::new_err("No node found for index"))
            }
        };
        Ok(node)
    }

    /// Return the edge data for all the edges between 2 nodes.
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node
    ///
    /// :returns: A list with all the data objects for the edges between nodes
    /// :rtype: list
    /// :raises NoEdgeBetweenNodes: When there is no edge between nodes
    #[text_signature = "(self, node_a, node_b, /)"]
    pub fn get_all_edge_data(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<Vec<&PyObject>> {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        let out: Vec<&PyObject> = self
            .graph
            .edges(index_a)
            .filter(|edge| edge.target() == index_b)
            .map(|edge| edge.weight())
            .collect();
        if out.is_empty() {
            Err(NoEdgeBetweenNodes::new_err("No edge found between nodes"))
        } else {
            Ok(out)
        }
    }

    /// Get edge list
    ///
    /// Returns a list of tuples of the form ``(source, target)`` where
    /// ``source`` and ``target`` are the node indices.
    ///
    /// :returns: An edge list with weights
    /// :rtype: EdgeList
    #[text_signature = "(self)"]
    pub fn edge_list(&self) -> EdgeList {
        EdgeList {
            edges: self
                .edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect(),
        }
    }

    /// Get edge list with weights
    ///
    /// Returns a list of tuples of the form ``(source, target, weight)`` where
    /// ``source`` and ``target`` are the node indices and ``weight`` is the
    /// payload of the edge.
    ///
    /// :returns: An edge list with weights
    /// :rtype: WeightedEdgeList
    #[text_signature = "(self)"]
    pub fn weighted_edge_list(&self, py: Python) -> WeightedEdgeList {
        WeightedEdgeList {
            edges: self
                .edge_references()
                .map(|edge| {
                    (
                        edge.source().index(),
                        edge.target().index(),
                        edge.weight().clone_ref(py),
                    )
                })
                .collect(),
        }
    }

    /// Remove a node from the graph.
    ///
    /// :param int node: The index of the node to remove. If the index is not
    ///     present in the graph it will be ignored and this function will
    ///     have no effect.
    #[text_signature = "(self, node, /)"]
    pub fn remove_node(&mut self, node: usize) -> PyResult<()> {
        let index = NodeIndex::new(node);
        self.graph.remove_node(index);
        self.node_removed = true;
        Ok(())
    }

    /// Add an edge between 2 nodes.
    ///
    /// :param int parent: Index of the parent node
    /// :param int child: Index of the child node
    /// :param edge: The object to set as the data for the edge. It can be any
    ///     python object.
    /// :param int parent: Index of the parent node
    /// :param int child: Index of the child node
    /// :param edge: The object to set as the data for the edge. It can be any
    ///     python object.
    #[text_signature = "(self, node_a, node_b, edge, /)"]
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

    /// Add new edges to the graph.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(node_a, node_b, obj)`` to attach to the graph. ``node_a`` and
    ///     ``node_b`` are integer indexes describing where an edge should be
    ///     added, and ``obj`` is the python object for the edge data.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[text_signature = "(self, obj_list, /)"]
    pub fn add_edges_from(
        &mut self,
        obj_list: Vec<(usize, usize, PyObject)>,
    ) -> PyResult<Vec<usize>> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            let edge = self.graph.add_edge(p_index, c_index, obj.2);
            out_list.push(edge.index());
        }
        Ok(out_list)
    }

    /// Add new edges to the graph without python data.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(parent, child)`` to attach to the graph. ``parent`` and
    ///     ``child`` are integer indexes describing where an edge should be
    ///     added. Unlike :meth:`add_edges_from` there is no data payload and
    ///     when the edge is created None will be used.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[text_signature = "(self, obj_list, /)"]
    pub fn add_edges_from_no_data(
        &mut self,
        py: Python,
        obj_list: Vec<(usize, usize)>,
    ) -> PyResult<Vec<usize>> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            let edge = self.graph.add_edge(p_index, c_index, py.None());
            out_list.push(edge.index());
        }
        Ok(out_list)
    }

    /// Extend graph from an edge list
    ///
    /// This method differs from :meth:`add_edges_from_no_data` in that it will
    /// add nodes if a node index is not present in the edge list.
    ///
    /// :param list edge_list: A list of tuples of the form ``(source, target)``
    ///     where source and target are integer node indices. If the node index
    ///     is not present in the graph, nodes will be added (with a node
    ///     weight of ``None``) to that index.
    #[text_signature = "(self, edge_list, /)"]
    pub fn extend_from_edge_list(
        &mut self,
        py: Python,
        edge_list: Vec<(usize, usize)>,
    ) {
        for (source, target) in edge_list {
            let max_index = cmp::max(source, target);
            while max_index >= self.node_count() {
                self.graph.add_node(py.None());
            }
            self.graph.add_edge(
                NodeIndex::new(source),
                NodeIndex::new(target),
                py.None(),
            );
        }
    }

    /// Extend graph from a weighted edge list
    ///
    /// This method differs from :meth:`add_edges_from` in that it will
    /// add nodes if a node index is not present in the edge list.
    ///
    /// :param list edge_list: A list of tuples of the form
    ///     ``(source, target, weight)`` where source and target are integer
    ///     node indices. If the node index is not present in the graph,
    ///     nodes will be added (with a node weight of ``None``) to that index.
    #[text_signature = "(self, edge_lsit, /)"]
    pub fn extend_from_weighted_edge_list(
        &mut self,
        py: Python,
        edge_list: Vec<(usize, usize, PyObject)>,
    ) {
        for (source, target, weight) in edge_list {
            let max_index = cmp::max(source, target);
            while max_index >= self.node_count() {
                self.graph.add_node(py.None());
            }
            self.graph.add_edge(
                NodeIndex::new(source),
                NodeIndex::new(target),
                weight,
            );
        }
    }

    /// Remove an edge between 2 nodes.
    ///
    /// Note if there are multiple edges between the specified nodes only one
    /// will be removed.
    ///
    /// :param int parent: The index for the parent node.
    /// :param int child: The index of the child node.
    ///
    /// :raises NoEdgeBetweenNodes: If there are no edges between the nodes
    ///     specified
    #[text_signature = "(self, node_a, node_b, /)"]
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
                return Err(NoEdgeBetweenNodes::new_err(
                    "No edge found between nodes",
                ))
            }
        };
        self.graph.remove_edge(edge_index);
        Ok(())
    }

    /// Remove an edge identified by the provided index
    ///
    /// :param int edge: The index of the edge to remove
    #[text_signature = "(self, edge, /)"]
    pub fn remove_edge_from_index(&mut self, edge: usize) -> PyResult<()> {
        let edge_index = EdgeIndex::new(edge);
        self.graph.remove_edge(edge_index);
        Ok(())
    }

    /// Remove edges from the graph.
    ///
    /// Note if there are multiple edges between the specified nodes only one
    /// will be removed.
    ///
    /// :param list index_list: A list of node index pairs to remove from
    ///     the graph
    #[text_signature = "(self, index_list, /)"]
    pub fn remove_edges_from(
        &mut self,
        index_list: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        for (p_index, c_index) in index_list
            .iter()
            .map(|(x, y)| (NodeIndex::new(*x), NodeIndex::new(*y)))
        {
            let edge_index = match self.graph.find_edge(p_index, c_index) {
                Some(edge_index) => edge_index,
                None => {
                    return Err(NoEdgeBetweenNodes::new_err(
                        "No edge found between nodes",
                    ))
                }
            };
            self.graph.remove_edge(edge_index);
        }
        Ok(())
    }

    /// Add a new node to the graph.
    ///
    /// :param obj: The python object to attach to the node
    ///
    /// :returns: The index of the newly created node
    /// :rtype: int
    #[text_signature = "(self, obj, /)"]
    pub fn add_node(&mut self, obj: PyObject) -> PyResult<usize> {
        let index = self.graph.add_node(obj);
        Ok(index.index())
    }

    /// Add new nodes to the graph.
    ///
    /// :param list obj_list: A list of python object to attach to the graph.
    ///
    /// :returns indices: A list of int indices of the newly created nodes
    /// :rtype: NodeIndices
    #[text_signature = "(self, obj_list, /)"]
    pub fn add_nodes_from(&mut self, obj_list: Vec<PyObject>) -> NodeIndices {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let node_index = self.graph.add_node(obj);
            out_list.push(node_index.index());
        }
        NodeIndices { nodes: out_list }
    }

    /// Remove nodes from the graph.
    ///
    /// If a node index in the list is not present in the graph it will be
    /// ignored.
    ///
    /// :param list index_list: A list of node indicies to remove from the
    ///     the graph
    #[text_signature = "(self, index_list, /)"]
    pub fn remove_nodes_from(
        &mut self,
        index_list: Vec<usize>,
    ) -> PyResult<()> {
        for node in index_list.iter().map(|x| NodeIndex::new(*x)) {
            self.graph.remove_node(node);
        }
        Ok(())
    }

    /// Get the index and data for the neighbors of a node.
    ///
    /// This will return a dictionary where the keys are the node indexes of
    /// the adjacent nodes (inbound or outbound) and the value is the edge data
    /// objects between that adjacent node and the provided node. Note, that
    /// in the case of multigraphs only a single edge data object will be
    /// returned
    ///
    /// :param int node: The index of the node to get the neighbors
    ///
    /// :returns neighbors: A dictionary where the keys are node indexes and
    ///     the value is the edge data object for all nodes that share an
    ///     edge with the specified node.
    /// :rtype: dict
    #[text_signature = "(self, node, /)"]
    pub fn adj(&mut self, node: usize) -> PyResult<HashMap<usize, &PyObject>> {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.neighbors(index);
        let mut out_map: HashMap<usize, &PyObject> = HashMap::new();

        for neighbor in neighbors {
            let edge = self.graph.find_edge(index, neighbor);
            let edge_w = self.graph.edge_weight(edge.unwrap());
            out_map.insert(neighbor.index(), edge_w.unwrap());
        }
        Ok(out_map)
    }

    /// Get the neighbors of a node.
    ///
    /// This with return a list of neighbor node indices
    ///
    /// :param int node: The index of the node to get the neibhors of
    ///
    /// :returns: A list of the neighbor node indicies
    /// :rtype: NodeIndices
    #[text_signature = "(self, node, /)"]
    pub fn neighbors(&self, node: usize) -> NodeIndices {
        NodeIndices {
            nodes: self
                .graph
                .neighbors(NodeIndex::new(node))
                .map(|node| node.index())
                .collect(),
        }
    }

    /// Get the degree for a node
    ///
    /// :param int node: The index of the  node to find the inbound degree of
    ///
    /// :returns degree: The inbound degree for the specified node
    /// :rtype: int
    #[text_signature = "(self, node, /)"]
    pub fn degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.edges(index);
        neighbors.count()
    }

    /// Generate a dot file from the graph
    ///
    /// :param node_attr: A callable that will take in a node data object
    ///     and return a dictionary of attributes to be associated with the
    ///     node in the dot file. The key and value of this dictionary **must**
    ///     be a string. If they're not strings retworkx will raise TypeError
    ///     (unfortunately without an error message because of current
    ///     limitations in the PyO3 type checking)
    /// :param edge_attr: A callable that will take in an edge data object
    ///     and return a dictionary of attributes to be associated with the
    ///     node in the dot file. The key and value of this dictionary **must**
    ///     be a string. If they're not strings retworkx will raise TypeError
    ///     (unfortunately without an error message because of current
    ///     limitations in the PyO3 type checking)
    /// :param dict graph_attr: An optional dictionary that specifies any graph
    ///     attributes for the output dot file. The key and value of this
    ///     dictionary **must** be a string. If they're not strings retworkx
    ///     will raise TypeError (unfortunately without an error message
    ///     because of current limitations in the PyO3 type checking)
    /// :param str filename: An optional path to write the dot file to
    ///     if specified there is no return from the function
    ///
    /// :returns: A string with the dot file contents if filename is not
    ///     specified.
    /// :rtype: str
    ///
    /// Using this method enables you to leverage graphviz to visualize a
    /// :class:`retworkx.PyGraph` object. For example:
    ///
    /// .. jupyter-execute::
    ///
    ///   import os
    ///   import tempfile
    ///
    ///   import pydot
    ///   from PIL import Image
    ///
    ///   import retworkx
    ///
    ///   graph = retworkx.undirected_gnp_random_graph(15, .25)
    ///   dot_str = graph.to_dot(
    ///       lambda node: dict(
    ///           color='black', fillcolor='lightblue', style='filled'))
    ///   dot = pydot.graph_from_dot_data(dot_str)[0]
    ///
    ///   with tempfile.TemporaryDirectory() as tmpdirname:
    ///       tmp_path = os.path.join(tmpdirname, 'dag.png')
    ///       dot.write_png(tmp_path)
    ///       image = Image.open(tmp_path)
    ///       os.remove(tmp_path)
    ///   image
    ///
    #[text_signature = "(self, /, node_attr=None, edge_attr=None, graph_attr=None, filename=None)"]
    pub fn to_dot(
        &self,
        py: Python,
        node_attr: Option<PyObject>,
        edge_attr: Option<PyObject>,
        graph_attr: Option<BTreeMap<String, String>>,
        filename: Option<String>,
    ) -> PyResult<Option<PyObject>> {
        if filename.is_some() {
            let mut file = File::create(filename.unwrap())?;
            build_dot(py, self, &mut file, graph_attr, node_attr, edge_attr)?;
            Ok(None)
        } else {
            let mut file = Vec::<u8>::new();
            build_dot(py, self, &mut file, graph_attr, node_attr, edge_attr)?;
            Ok(Some(
                PyString::new(py, str::from_utf8(&file)?).to_object(py),
            ))
        }
    }

    /// Read an edge list file and create a new PyGraph object from the
    /// contents
    ///
    /// The expected format for the edge list file is a line seperated list
    /// of deliminated node ids. If there are more than 3 elements on
    /// a line the 3rd on will be treated as a string weight for the edge
    ///
    /// :param str path: The path of the file to open
    /// :param str comment: Optional character to use as a comment by default
    ///     there are no comment characters
    /// :param str deliminator: Optional character to use as a deliminator by
    ///     default any whitespace will be used
    ///
    /// For example:
    ///
    /// .. jupyter-execute::
    ///
    ///   import os
    ///   import tempfile
    ///
    ///   from PIL import Image
    ///   import pydot
    ///
    ///   import retworkx
    ///
    ///
    ///   with tempfile.NamedTemporaryFile('wt') as fd:
    ///       path = fd.name
    ///       fd.write('0 1\n')
    ///       fd.write('0 2\n')
    ///       fd.write('0 3\n')
    ///       fd.write('1 2\n')
    ///       fd.write('2 3\n')
    ///       fd.flush()
    ///       graph = retworkx.PyGraph.read_edge_list(path)
    ///
    ///   # Draw graph
    ///   dot = pydot.graph_from_dot_data(graph.to_dot())[0]
    ///
    ///   with tempfile.TemporaryDirectory() as tmpdirname:
    ///       tmp_path = os.path.join(tmpdirname, 'dag.png')
    ///       dot.write_png(tmp_path)
    ///       image = Image.open(tmp_path)
    ///       os.remove(tmp_path)
    ///   image
    ///
    #[staticmethod]
    #[text_signature = "(self, path, /, comment=None, deliminator=None)"]
    pub fn read_edge_list(
        py: Python,
        path: &str,
        comment: Option<String>,
        deliminator: Option<String>,
    ) -> PyResult<PyGraph> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut out_graph = StableUnGraph::<PyObject, PyObject>::default();
        for line_raw in buf_reader.lines() {
            let line = line_raw?;
            let skip = match &comment {
                Some(comm) => line.trim().starts_with(comm),
                None => line.trim().is_empty(),
            };
            if skip {
                continue;
            }
            let line_no_comments = match &comment {
                Some(comm) => line
                    .find(comm)
                    .map(|idx| &line[..idx])
                    .unwrap_or(&line)
                    .trim()
                    .to_string(),
                None => line,
            };
            let pieces: Vec<&str> = match &deliminator {
                Some(del) => line_no_comments.split(del).collect(),
                None => line_no_comments.split_whitespace().collect(),
            };
            let src = pieces[0].parse::<usize>()?;
            let target = pieces[1].parse::<usize>()?;
            let max_index = cmp::max(src, target);
            // Add nodes to graph
            while max_index >= out_graph.node_count() {
                out_graph.add_node(py.None());
            }
            // Add edges tp graph
            let weight = if pieces.len() > 2 {
                let weight_str = match &deliminator {
                    Some(del) => pieces[2..].join(del),
                    None => pieces[2..].join(&' '.to_string()),
                };
                PyString::new(py, &weight_str).into()
            } else {
                py.None()
            };
            out_graph.add_edge(
                NodeIndex::new(src),
                NodeIndex::new(target),
                weight,
            );
        }
        Ok(PyGraph {
            graph: out_graph,
            node_removed: false,
        })
    }

    /// Add another PyGraph object into this PyGraph
    ///
    /// :param PyGraph other: The other PyGraph object to add onto this
    ///     graph.
    /// :param dict node_map: A dictionary mapping node indexes from this
    ///     PyGraph object to node indexes in the other PyGraph object.
    ///     The keys are a node index in this graph and the value is a tuple
    ///     of the node index in the other graph to add an edge to and the
    ///     weight of that edge. For example::
    ///
    ///         {
    ///             1: (2, "weight"),
    ///             2: (4, "weight2")
    ///         }
    ///
    /// :param node_map_func: An optional python callable that will take in a
    ///     single node weight/data object and return a new node weight/data
    ///     object that will be used when adding an node from other onto this
    ///     graph.
    /// :param edge_map_func: An optional python callabble that will take in a
    ///     single edge weight/data object and return a new edge weight/data
    ///     object that will be used when adding an edge from other onto this
    ///     graph.
    ///
    /// :returns: new_node_ids: A dictionary mapping node index from the other
    ///     PyGraph to the equivalent node index in this PyDAG after they've
    ///     been combined
    /// :rtype: dict
    ///
    /// For example, start by building a graph:
    ///
    /// .. jupyter-execute::
    ///
    ///   import os
    ///   import tempfile
    ///
    ///   import pydot
    ///   from PIL import Image
    ///
    ///   import retworkx
    ///
    ///   # Build first graph and visualize:
    ///   graph = retworkx.PyGraph()
    ///   node_a, node_b, node_c = graph.add_nodes_from(['A', 'B', 'C'])
    ///   graph.add_edges_from_no_data([(node_a, node_b), (node_b, node_c)])
    ///   dot_str = graph.to_dot(
    ///       lambda node: dict(
    ///           color='black', fillcolor='lightblue', style='filled'))
    ///   dot = pydot.graph_from_dot_data(dot_str)[0]
    ///
    ///   with tempfile.TemporaryDirectory() as tmpdirname:
    ///       tmp_path = os.path.join(tmpdirname, 'graph.png')
    ///       dot.write_png(tmp_path)
    ///       image = Image.open(tmp_path)
    ///       os.remove(tmp_path)
    ///   image
    ///
    /// Then build a second one:
    ///
    /// .. jupyter-execute::
    ///
    ///   # Build second graph and visualize:
    ///   other_graph = retworkx.PyGraph()
    ///   node_d, node_e = other_graph.add_nodes_from(['D', 'E'])
    ///   other_graph.add_edge(node_d, node_e, None)
    ///   dot_str = other_graph.to_dot(
    ///       lambda node: dict(
    ///           color='black', fillcolor='lightblue', style='filled'))
    ///   dot = pydot.graph_from_dot_data(dot_str)[0]
    ///
    ///   with tempfile.TemporaryDirectory() as tmpdirname:
    ///       tmp_path = os.path.join(tmpdirname, 'other_graph.png')
    ///       dot.write_png(tmp_path)
    ///       image = Image.open(tmp_path)
    ///       os.remove(tmp_path)
    ///   image
    ///
    /// Finally compose the ``other_graph`` onto ``graph``
    ///
    /// .. jupyter-execute::
    ///
    ///   node_map = {node_b: (node_d, 'B to D')}
    ///   graph.compose(other_graph, node_map)
    ///   dot_str = graph.to_dot(
    ///       lambda node: dict(
    ///           color='black', fillcolor='lightblue', style='filled'))
    ///   dot = pydot.graph_from_dot_data(dot_str)[0]
    ///
    ///   with tempfile.TemporaryDirectory() as tmpdirname:
    ///       tmp_path = os.path.join(tmpdirname, 'combined_graph.png')
    ///       dot.write_png(tmp_path)
    ///       image = Image.open(tmp_path)
    ///       os.remove(tmp_path)
    ///   image
    ///
    #[text_signature = "(self, other, node_map, /, node_map_func=None, edge_map_func=None)"]
    pub fn compose(
        &mut self,
        py: Python,
        other: &PyGraph,
        node_map: HashMap<usize, (usize, PyObject)>,
        node_map_func: Option<PyObject>,
        edge_map_func: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let mut new_node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        // TODO: Reimplement this without looping over the graphs
        // Loop over other nodes add add to self graph
        for node in other.graph.node_indices() {
            let new_index = self.graph.add_node(weight_transform_callable(
                py,
                &node_map_func,
                &other.graph[node],
            )?);
            new_node_map.insert(node, new_index);
        }

        // loop over other edges and add to self graph
        for edge in other.graph.edge_references() {
            let new_p_index = new_node_map.get(&edge.source()).unwrap();
            let new_c_index = new_node_map.get(&edge.target()).unwrap();
            let weight =
                weight_transform_callable(py, &edge_map_func, edge.weight())?;
            self.graph.add_edge(*new_p_index, *new_c_index, weight);
        }
        // Add edges from map
        for (this_index, (index, weight)) in node_map.iter() {
            let new_index = new_node_map.get(&NodeIndex::new(*index)).unwrap();
            self.graph.add_edge(
                NodeIndex::new(*this_index),
                *new_index,
                weight.clone_ref(py),
            );
        }
        let out_dict = PyDict::new(py);
        for (orig_node, new_node) in new_node_map.iter() {
            out_dict.set_item(orig_node.index(), new_node.index())?;
        }
        Ok(out_dict.into())
    }

    /// Return a new PyGraph object for a subgraph of this graph
    ///
    /// :param list nodes: A list of node indices to generate the subgraph
    ///     from. If a node index is included that is not present in the graph
    ///     it will silently be ignored.
    ///
    /// :returns: A new PyGraph object representing a subgraph of this graph.
    ///     It is worth noting that node and edge weight/data payloads are
    ///     passed by reference so if you update (not replace) an object used
    ///     as the weight in graph or the subgraph it will also be updated in
    ///     the other.
    /// :rtype: PyGraph
    ///
    #[text_signature = "(self, nodes, /)"]
    pub fn subgraph(&self, py: Python, nodes: Vec<usize>) -> PyGraph {
        let node_set: HashSet<usize> = nodes.iter().cloned().collect();
        let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let node_filter =
            |node: NodeIndex| -> bool { node_set.contains(&node.index()) };
        let mut out_graph = StableUnGraph::<PyObject, PyObject>::default();
        let filtered = NodeFiltered(self, node_filter);
        for node in filtered.node_references() {
            let new_node = out_graph.add_node(node.1.clone_ref(py));
            node_map.insert(node.0, new_node);
        }
        for edge in filtered.edge_references() {
            let new_source = *node_map.get(&edge.source()).unwrap();
            let new_target = *node_map.get(&edge.target()).unwrap();
            out_graph.add_edge(
                new_source,
                new_target,
                edge.weight().clone_ref(py),
            );
        }
        PyGraph {
            graph: out_graph,
            node_removed: false,
        }
    }
}

#[pyproto]
impl PyMappingProtocol for PyGraph {
    /// Return the nmber of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<&'p PyObject> {
        match self.graph.node_weight(NodeIndex::new(idx)) {
            Some(data) => Ok(data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }

    fn __setitem__(&'p mut self, idx: usize, value: PyObject) -> PyResult<()> {
        let data = match self.graph.node_weight_mut(NodeIndex::new(idx)) {
            Some(node_data) => node_data,
            None => {
                return Err(PyIndexError::new_err("No node found for index"))
            }
        };
        *data = value;
        Ok(())
    }

    fn __delitem__(&'p mut self, idx: usize) -> PyResult<()> {
        match self.graph.remove_node(NodeIndex::new(idx as usize)) {
            Some(_) => Ok(()),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }
}

fn weight_transform_callable(
    py: Python,
    map_fn: &Option<PyObject>,
    value: &PyObject,
) -> PyResult<PyObject> {
    match map_fn {
        Some(map_fn) => {
            let res = map_fn.call1(py, (value,))?;
            Ok(res.to_object(py))
        }
        None => Ok(value.clone_ref(py)),
    }
}
