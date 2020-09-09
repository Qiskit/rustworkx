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

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::ops::{Index, IndexMut};
use std::str;

use hashbrown::HashMap;

use pyo3::class::PyMappingProtocol;
use pyo3::exceptions::IndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyLong, PyString, PyTuple};
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

use super::dot_utils::build_dot;
use super::{
    is_directed_acyclic_graph, DAGHasCycle, DAGWouldCycle, NoEdgeBetweenNodes,
    NoSuitableNeighbors,
};

/// A class for creating directed graphs
///
/// The PyDigraph class is constructed using the Rust library
/// `petgraph <https://github.com/petgraph/petgraph>`__ around the
/// ``StableGraph`` type. The limitations and quirks with this library and
/// type dictate how this operates. The biggest thing to be aware of when using
/// the PyDiGraph class is that an integer node and edge index is used for
/// accessing elements on the graph, not the data/weight of nodes and edges. By
/// default the PyDiGraph realtime cycle checking is disabled for performance,
/// however you can opt-in to having the PyDiGraph class ensure that no cycles
/// are added by setting the ``check_cycle`` attribute to True. For example::
///
///     import retworkx
///     dag = retworkx.PyDiGraph()
///     dag.check_cycle = True
///
/// or at object creation::
///
///     import retworkx
///     dag = retworkx.PyDiGraph(check_cycle=True)
///
/// With check_cycle set to true any calls to :meth:`PyDiGraph.add_edge` will
/// ensure that no cycles are added, ensuring that the PyDiGraph class truly
/// represents a directed acyclic graph.
///
///  .. note::
///        When using ``copy.deepcopy()`` or pickling node indexes are not
///        guaranteed to be preserved.
///
#[pyclass(module = "retworkx", subclass)]
#[text_signature = "(/, check_cycle=False)"]
pub struct PyDiGraph {
    pub graph: StableDiGraph<PyObject, PyObject>,
    pub cycle_state: algo::DfsSpace<
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

    /// Whether cycle checking is enabled for the DiGraph/DAG.
    ///
    /// If set to ``True`` adding new edges that would introduce a cycle
    /// will raise a :class:`DAGWouldCycle` exception.
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

    /// Return a list of all edge data.
    ///
    /// :returns: A list of all the edge data objects in the graph
    /// :rtype: list
    #[text_signature = "()"]
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
    #[text_signature = "()"]
    pub fn nodes(&self) -> Vec<&PyObject> {
        self.graph
            .node_indices()
            .map(|node| self.graph.node_weight(node).unwrap())
            .collect()
    }

    /// Return a list of all node indexes.
    ///
    /// :returns: A list of all the node indexes in the graph
    /// :rtype: list
    #[text_signature = "()"]
    pub fn node_indexes(&self) -> Vec<usize> {
        self.graph.node_indices().map(|node| node.index()).collect()
    }

    /// Return True if there is an edge from node_a to node_b.
    ///
    /// :param int node_a: The source node index to check for an edge
    /// :param int node_b: The destination node index to check for an edge
    ///
    /// :returns: True if there is an edge false if there is no edge
    /// :rtype: bool
    #[text_signature = "(node_a, node_b, /)"]
    pub fn has_edge(&self, node_a: usize, node_b: usize) -> bool {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        self.graph.find_edge(index_a, index_b).is_some()
    }

    /// Return a list of all the node successor data.
    ///
    /// :param int node: The index for the node to get the successors for
    ///
    /// :returns: A list of the node data for all the child neighbor nodes
    /// :rtype: list
    #[text_signature = "(node, /)"]
    pub fn successors(&self, node: usize) -> Vec<&PyObject> {
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
        succesors
    }

    /// Return a list of all the node predecessor data.
    ///
    /// :param int node: The index for the node to get the predecessors for
    ///
    /// :returns: A list of the node data for all the parent neighbor nodes
    /// :rtype: list
    #[text_signature = "(node, /)"]
    pub fn predecessors(&self, node: usize) -> Vec<&PyObject> {
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
        predec
    }

    /// Return the edge data for an edge between 2 nodes.
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node
    ///
    /// :returns: The data object set for the edge
    /// :raises NoEdgeBetweenNodes: When there is no edge between nodes
    #[text_signature = "(node_a, node_b, /)"]
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

    /// Return the node data for a given node index
    ///
    /// :param int node: The index for the node
    ///
    /// :returns: The data object set for that node
    /// :raises IndexError: when an invalid node index is provided
    #[text_signature = "(node, /)"]
    pub fn get_node_data(&self, node: usize) -> PyResult<&PyObject> {
        let index = NodeIndex::new(node);
        let node = match self.graph.node_weight(index) {
            Some(node) => node,
            None => return Err(IndexError::py_err("No node found for index")),
        };
        Ok(node)
    }

    /// Return the edge data for all the edges between 2 nodes.
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node

    /// :returns: A list with all the data objects for the edges between nodes
    /// :rtype: list
    /// :raises NoEdgeBetweenNodes: When there is no edge between nodes
    #[text_signature = "(node_a, node_b, /)"]
    pub fn get_all_edge_data(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> PyResult<Vec<&PyObject>> {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        let raw_edges = self
            .graph
            .edges_directed(index_a, petgraph::Direction::Outgoing);
        let out: Vec<&PyObject> = raw_edges
            .filter(|x| x.target() == index_b)
            .map(|edge| edge.weight())
            .collect();
        if out.is_empty() {
            Err(NoEdgeBetweenNodes::py_err("No edge found between nodes"))
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
    /// :rtype: list
    pub fn edge_list(&self) -> Vec<(usize, usize)> {
        self.edge_references()
            .map(|edge| (edge.source().index(), edge.target().index()))
            .collect()
    }

    /// Get edge list with weights
    ///
    /// Returns a list of tuples of the form ``(source, target, weight)`` where
    /// ``source`` and ``target`` are the node indices and ``weight`` is the
    /// payload of the edge.
    ///
    /// :returns: An edge list with weights
    /// :rtype: list
    pub fn weighted_edge_list(
        &self,
        py: Python,
    ) -> Vec<(usize, usize, PyObject)> {
        self.edge_references()
            .map(|edge| {
                (
                    edge.source().index(),
                    edge.target().index(),
                    edge.weight().clone_ref(py),
                )
            })
            .collect()
    }

    /// Remove a node from the graph.
    ///
    /// :param int node: The index of the node to remove. If the index is not
    ///     present in the graph it will be ignored and this function will have
    ///     no effect.
    #[text_signature = "(node, /)"]
    pub fn remove_node(&mut self, node: usize) -> PyResult<()> {
        let index = NodeIndex::new(node);
        self.graph.remove_node(index);
        self.node_removed = true;
        Ok(())
    }

    /// Remove a node from the graph and add edges from all predecessors to all
    /// successors
    ///
    /// By default the data/weight on edges into the removed node will be used
    /// for the retained edges.
    ///
    /// :param int node: The index of the node to remove. If the index is not
    ///     present in the graph it will be ingored and this function willl have
    ///     no effect.
    /// :param bool use_outgoing: If set to true the weight/data from the
    ///     edge outgoing from ``node`` will be used in the retained edge
    ///     instead of the default weight/data from the incoming edge.
    /// :param condition: A callable that will be passed 2 edge weight/data
    ///     objects, one from the incoming edge to ``node`` the other for the
    ///     outgoing edge, and will return a ``bool`` on whether an edge should
    ///     be retained. For example setting this kwarg to::
    ///
    ///         lambda in_edge, out_edge: in_edge == out_edge
    ///
    ///     would only retain edges if the input edge to ``node`` had the same
    ///     data payload as the outgoing edge.
    #[text_signature = "(node, /, use_outgoing=None, condition=None)"]
    #[args(use_outgoing = "false")]
    pub fn remove_node_retain_edges(
        &mut self,
        py: Python,
        node: usize,
        use_outgoing: bool,
        condition: Option<PyObject>,
    ) -> PyResult<()> {
        let index = NodeIndex::new(node);
        let mut edge_list: Vec<(NodeIndex, NodeIndex, PyObject)> = Vec::new();

        fn check_condition(
            py: Python,
            condition: &Option<PyObject>,
            in_weight: &PyObject,
            out_weight: &PyObject,
        ) -> PyResult<bool> {
            match condition {
                Some(condition) => {
                    let res = condition.call1(py, (in_weight, out_weight))?;
                    Ok(res.extract(py)?)
                }
                None => Ok(true),
            }
        }

        for (source, in_weight) in self
            .graph
            .edges_directed(index, petgraph::Direction::Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .graph
                .edges_directed(index, petgraph::Direction::Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                let weight = if use_outgoing { out_weight } else { in_weight };
                if check_condition(py, &condition, in_weight, out_weight)? {
                    edge_list.push((source, target, weight.clone_ref(py)));
                }
            }
        }
        for (source, target, weight) in edge_list {
            self._add_edge(source, target, weight)?;
        }
        self.graph.remove_node(index);
        self.node_removed = true;
        Ok(())
    }

    /// Add an edge between 2 nodes.
    ///
    /// Use add_child() or add_parent() to create a node with an edge at the
    /// same time as an edge for better performance. Using this method will
    /// enable adding duplicate edges between nodes if the ``check_cycle``
    /// attribute is set to ``True``.
    ///
    /// :param int parent: Index of the parent node
    /// :param int child: Index of the child node
    /// :param edge: The object to set as the data for the edge. It can be any
    ///     python object.
    ///
    /// :returns: The edge index of the created edge
    /// :rtype: int
    ///
    /// :raises: When the new edge will create a cycle
    #[text_signature = "(parent, child, edge, /)"]
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

    /// Add new edges to the dag.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(parent, child, obj)`` to attach to the graph. ``parent`` and
    ///     ``child`` are integer indexes describing where an edge should be
    ///     added, and obj is the python object for the edge data.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[text_signature = "(obj_list, /)"]
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

    /// Add new edges to the dag without python data.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(parent, child)`` to attach to the graph. ``parent`` and
    ///     ``child`` are integer indexes describing where an edge should be
    ///     added. Unlike :meth:`add_edges_from` there is no data payload and
    ///     when the edge is created None will be used.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[text_signature = "(obj_list, /)"]
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
    #[text_signature = "(parent, child, /)"]
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

    /// Remove an edge identified by the provided index
    ///
    /// :param int edge: The index of the edge to remove
    #[text_signature = "(edge, /)"]
    pub fn remove_edge_from_index(&mut self, edge: usize) -> PyResult<()> {
        let edge_index = EdgeIndex::new(edge);
        self.graph.remove_edge(edge_index);
        Ok(())
    }

    /// Add a new node to the graph.
    ///
    /// :param obj: The python object to attach to the node
    ///
    /// :returns: The index of the newly created node
    /// :rtype: int
    #[text_signature = "(obj, /)"]
    pub fn add_node(&mut self, obj: PyObject) -> PyResult<usize> {
        let index = self.graph.add_node(obj);
        Ok(index.index())
    }

    /// Add a new child node to the graph.
    ///
    /// This will create a new node on the graph and add an edge from the parent
    /// to that new node.
    ///
    /// :param int parent: The index for the parent node
    /// :param obj: The python object to attach to the node
    /// :param edge: The python object to attach to the edge
    ///
    /// :returns: The index of the newly created child node
    /// :rtype: int
    #[text_signature = "(parent, obj, edge, /)"]
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

    /// Add a new parent node to the dag.
    ///
    /// This create a new node on the dag and add an edge to the child from
    /// that new node
    ///
    /// :param int child: The index of the child node
    /// :param obj: The python object to attach to the node
    /// :param edge: The python object to attach to the edge
    ///
    /// :returns index: The index of the newly created parent node
    /// :rtype: int
    #[text_signature = "(child, obj, edge, /)"]
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

    /// Get the index and data for the neighbors of a node.
    ///
    /// This will return a dictionary where the keys are the node indexes of
    /// the adjacent nodes (inbound or outbound) and the value is the edge dat
    /// objects between that adjacent node and the provided node. Note in
    /// the case of a multigraph only one edge will be used, not all of the
    /// edges between two node.
    ///
    /// :param int node: The index of the node to get the neighbors
    ///
    /// :returns: A dictionary where the keys are node indexes and the value
    ///     is the edge data object for all nodes that share an edge with the
    ///     specified node.
    /// :rtype: dict
    #[text_signature = "(node, /)"]
    pub fn adj(&mut self, py: Python, node: usize) -> PyResult<PyObject> {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.neighbors(index);
        let mut out_map: HashMap<usize, &PyObject> = HashMap::new();
        for neighbor in neighbors {
            let mut edge = self.graph.find_edge(index, neighbor);
            // If there is no edge then it must be a parent neighbor
            if edge.is_none() {
                edge = self.graph.find_edge(neighbor, index);
            }
            let edge_w = self.graph.edge_weight(edge.unwrap());
            out_map.insert(neighbor.index(), edge_w.unwrap());
        }
        let out_dict = PyDict::new(py);
        for (index, value) in out_map {
            out_dict.set_item(index, value)?;
        }
        Ok(out_dict.into())
    }

    /// Get the index and data for either the parent or children of a node.
    ///
    /// This will return a dictionary where the keys are the node indexes of
    /// the adjacent nodes (inbound or outbound as specified) and the value
    /// is the edge data objects for the edges between that adjacent node
    /// and the provided node. Note in the case of a multigraph only one edge
    /// one edge will be used, not all of the edges between two node.
    ///
    /// :param int node: The index of the node to get the neighbors
    /// :param bool direction: The direction to use for finding nodes,
    ///     True means inbound edges and False means outbound edges.
    ///
    /// :returns: A dictionary where the keys are node indexes and
    ///     the value is the edge data object for all nodes that share an
    ///     edge with the specified node.
    /// :rtype: dict
    #[text_signature = "(node, direction, /)"]
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
        let mut out_map: HashMap<usize, &PyObject> = HashMap::new();
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
            out_map.insert(neighbor.index(), edge_w.unwrap());
        }
        let out_dict = PyDict::new(py);
        for (index, value) in out_map {
            out_dict.set_item(index, value)?;
        }
        Ok(out_dict.into())
    }

    /// Get the index and edge data for all parents of a node.
    ///
    /// This will return a list of tuples with the parent index the node index
    /// and the edge data. This can be used to recreate add_edge() calls.
    /// :param int node: The index of the node to get the edges for
    ///
    /// :param int node: The index of the node to get the edges for
    ///
    /// :returns: A list of tuples of the form:
    ///     ``(parent_index, node_index, edge_data)```
    /// :rtype: list
    #[text_signature = "(node, /)"]
    pub fn in_edges(&self, node: usize) -> Vec<(usize, usize, &PyObject)> {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Incoming;
        let raw_edges = self.graph.edges_directed(index, dir);
        let out_list: Vec<(usize, usize, &PyObject)> = raw_edges
            .map(|x| (x.source().index(), node, x.weight()))
            .collect();
        out_list
    }

    /// Get the index and edge data for all children of a node.
    ///
    /// This will return a list of tuples with the child index the node index
    /// and the edge data. This can be used to recreate add_edge() calls.
    ///
    /// :param int node: The index of the node to get the edges for
    ///
    /// :returns out_edges: A list of tuples of the form:
    ///     ```(node_index, child_index, edge_data)```
    /// :rtype: list
    #[text_signature = "(node, /)"]
    pub fn out_edges(&self, node: usize) -> Vec<(usize, usize, &PyObject)> {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let raw_edges = self.graph.edges_directed(index, dir);
        let out_list: Vec<(usize, usize, &PyObject)> = raw_edges
            .map(|x| (node, x.target().index(), x.weight()))
            .collect();
        out_list
    }

    /// Add new nodes to the graph.
    ///
    /// :param list obj_list: A list of python objects to attach to the graph
    ///     as new nodes
    ///
    /// :returns: A list of int indices of the newly created nodes
    /// :rtype: list
    #[text_signature = "(obj_list, /)"]
    pub fn add_nodes_from(&mut self, obj_list: Vec<PyObject>) -> Vec<usize> {
        let mut out_list: Vec<usize> = Vec::new();
        for obj in obj_list {
            let node_index = self.graph.add_node(obj);
            out_list.push(node_index.index());
        }
        out_list
    }

    /// Remove nodes from the graph.
    ///
    /// If a node index in the list is not present in the graph it will be
    /// ignored.
    ///
    /// :param list index_list: A list of node indicies to remove from the
    ///     the graph.
    #[text_signature = "(index_list, /)"]
    pub fn remove_nodes_from(
        &mut self,
        index_list: Vec<usize>,
    ) -> PyResult<()> {
        for node in index_list.iter().map(|x| NodeIndex::new(*x)) {
            self.graph.remove_node(node);
        }
        Ok(())
    }

    /// Get the degree of a node for inbound edges.
    ///
    /// :param int node: The index of the node to find the inbound degree of
    ///
    /// :returns: The inbound degree for the specified node
    /// :rtype: int
    #[text_signature = "(node, /)"]
    pub fn in_degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Incoming;
        let neighbors = self.graph.edges_directed(index, dir);
        neighbors.count()
    }

    /// Get the degree of a node for outbound edges.
    ///
    /// :param int node: The index of the node to find the outbound degree of
    /// :returns: The outbound degree for the specified node
    /// :rtype: int
    #[text_signature = "(node, /)"]
    pub fn out_degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let neighbors = self.graph.edges_directed(index, dir);
        neighbors.count()
    }

    /// Find a target node with a specific edge
    ///
    /// This method is used to find a target node that is a adjacent to a given
    /// node given an edge condition.
    ///
    /// :param int node: The node to use as the source of the search
    /// :param callable predicate: A python callable that will take a single
    ///     parameter, the edge object, and will return a boolean if the
    ///     edge matches or not
    ///
    /// :returns: The node object that has an edge to it from the provided
    ///     node index which matches the provided condition
    #[text_signature = "(node, predicate, /)"]
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

    /// Generate a dot file from the graph
    ///
    /// :param node_attr: A callable that will take in a node data object
    ///     and return a dictionary of attributes to be associated with the
    ///     node in the dot file. The key and value of this dictionary **must**
    ///     be strings. If they're not strings retworkx will raise TypeError
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
    /// :class:`retworkx.PyDiGraph` object. For example:
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
    ///   graph = retworkx.directed_gnp_random_graph(15, .25)
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
    #[text_signature = "(/, node_attr=None, edge_attr=None, graph_attr=None, filename=None)"]
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

    /// Read an edge list file and create a new PyDiGraph object from the
    /// contents
    ///
    /// The expected format for the edge list file is a line seperated list
    /// of deliminated node ids. If there are more than 3 elements on
    /// a line the 3rd on will be treated as a string weight for the edge
    ///
    /// :param str path: The path of the file to open
    /// :param str comment: Optional character to use as a comment by default
    ///     there are no comment character
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
    ///       graph = retworkx.PyDiGraph.read_edge_list(path)
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
    #[text_signature = "(path, /, comment=None, deliminator=None)"]
    pub fn read_edge_list(
        py: Python,
        path: &str,
        comment: Option<String>,
        deliminator: Option<String>,
    ) -> PyResult<PyDiGraph> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut out_graph = StableDiGraph::<PyObject, PyObject>::new();
        for line_raw in buf_reader.lines() {
            let line = line_raw?;
            let skip = match &comment {
                Some(comm) => line.starts_with(comm),
                None => false,
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
            if !out_graph.contains_node(NodeIndex::new(src)) {
                for _index in 0..((src + 1) - out_graph.node_count()) {
                    out_graph.add_node(py.None());
                }
            }
            if !out_graph.contains_node(NodeIndex::new(target)) {
                for _index in 0..((target + 1) - out_graph.node_count()) {
                    out_graph.add_node(py.None());
                }
            }
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
        Ok(PyDiGraph {
            graph: out_graph,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
        })
    }

    /// Add another PyDiGraph object into this PyDiGraph
    ///
    /// :param PyDiGraph other: The other PyDiGraph object to add onto this
    ///     graph.
    /// :param dict node_map: A dictionary mapping node indexes from this
    ///     PyDiGraph object to node indexes in the other PyDiGraph object.
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
    /// :param edge_map_func: An optional python callable that will take in a
    ///     single edge weight/data object and return a new edge weight/data
    ///     object that will be used when adding an edge from other onto this
    ///     graph.
    ///
    /// :returns: new_node_ids: A dictionary mapping node index from the other
    ///     PyDiGraph to the corresponding node index in this PyDAG after they've been
    ///     combined
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
    ///   graph = retworkx.PyDiGraph()
    ///   node_a = graph.add_node('A')
    ///   node_b = graph.add_child(node_a, 'B', 'A to B')
    ///   node_c = graph.add_child(node_b, 'C', 'B to C')
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
    ///   other_graph = retworkx.PyDiGraph()
    ///   node_d = other_graph.add_node('D')
    ///   other_graph.add_child(node_d, 'E', 'D to E')
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
    #[text_signature = "(other, node_map, /, node_map_func=None, edge_map_func=None)"]
    pub fn compose(
        &mut self,
        py: Python,
        other: &PyDiGraph,
        node_map: PyObject,
        node_map_func: Option<PyObject>,
        edge_map_func: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let mut new_node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let node_map_dict = node_map.cast_as::<PyDict>(py)?;
        let mut node_map_hashmap: HashMap<usize, (usize, PyObject)> =
            HashMap::default();
        for (k, v) in node_map_dict.iter() {
            node_map_hashmap.insert(k.extract()?, v.extract()?);
        }

        fn node_weight_callable(
            py: Python,
            node_map: &Option<PyObject>,
            node: &PyObject,
        ) -> PyResult<PyObject> {
            match node_map {
                Some(node_map) => {
                    let res = node_map.call1(py, (node,))?;
                    Ok(res.to_object(py))
                }
                None => Ok(node.clone_ref(py)),
            }
        }

        // TODO: Reimplement this without looping over the graphs
        // Loop over other nodes add add to self graph
        for node in other.graph.node_indices() {
            let new_index = self.graph.add_node(node_weight_callable(
                py,
                &node_map_func,
                &other.graph[node],
            )?);
            new_node_map.insert(node, new_index);
        }

        fn edge_weight_callable(
            py: Python,
            edge_map: &Option<PyObject>,
            edge: &PyObject,
        ) -> PyResult<PyObject> {
            match edge_map {
                Some(edge_map) => {
                    let res = edge_map.call1(py, (edge,))?;
                    Ok(res.to_object(py))
                }
                None => Ok(edge.clone_ref(py)),
            }
        }

        // loop over other edges and add to self graph
        for edge in other.graph.edge_references() {
            let new_p_index = new_node_map.get(&edge.source()).unwrap();
            let new_c_index = new_node_map.get(&edge.target()).unwrap();
            let weight =
                edge_weight_callable(py, &edge_map_func, edge.weight())?;
            self.graph.add_edge(*new_p_index, *new_c_index, weight);
        }
        // Add edges from map
        for (this_index, (index, weight)) in node_map_hashmap.iter() {
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
}

#[pyproto]
impl PyMappingProtocol for PyDiGraph {
    /// Return the number of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<&'p PyObject> {
        match self.graph.node_weight(NodeIndex::new(idx as usize)) {
            Some(data) => Ok(data),
            None => Err(IndexError::py_err("No node found for index")),
        }
    }

    fn __setitem__(&'p mut self, idx: usize, value: PyObject) -> PyResult<()> {
        let data = match self
            .graph
            .node_weight_mut(NodeIndex::new(idx as usize))
        {
            Some(node_data) => node_data,
            None => return Err(IndexError::py_err("No node found for index")),
        };
        *data = value;
        Ok(())
    }

    fn __delitem__(&'p mut self, idx: usize) -> PyResult<()> {
        match self.graph.remove_node(NodeIndex::new(idx as usize)) {
            Some(_) => Ok(()),
            None => Err(IndexError::py_err("No node found for index")),
        }
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
