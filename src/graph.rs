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

#![allow(clippy::borrow_as_ptr)]

use std::cmp;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::str;

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use rustworkx_core::dictmap::*;

use pyo3::exceptions::PyIndexError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyLong, PyString, PyTuple};
use pyo3::PyTraverseError;
use pyo3::Python;

use ndarray::prelude::*;
use num_traits::Zero;
use numpy::Complex64;
use numpy::PyReadonlyArray2;

use super::dot_utils::build_dot;
use super::iterators::{EdgeIndexMap, EdgeIndices, EdgeList, NodeIndices, WeightedEdgeList};
use super::{
    find_node_by_weight, merge_duplicates, weight_callable, IsNan, NoEdgeBetweenNodes,
    NodesRemoved, StablePyGraph,
};

use petgraph::algo;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::visit::{
    GraphBase, IntoEdgeReferences, IntoNodeReferences, NodeCount, NodeFiltered, NodeIndexable,
};

/// A class for creating undirected graphs
///
/// The PyGraph class is used to create an undirected graph. It can be a
/// multigraph (have multiple edges between nodes). Each node and edge
/// (although rarely used for edges) is indexed by an integer id. These ids
/// are stable for the lifetime of the graph object and on node or edge
/// deletions you can have holes in the list of indices for the graph.
/// Node indices will be reused on additions after removal. For example:
///
/// .. jupyter-execute::
///
///        import rustworkx as rx
///
///        graph = rx.PyGraph()
///        graph.add_nodes_from(list(range(5)))
///        graph.add_nodes_from(list(range(2)))
///        graph.remove_node(2)
///        print("After deletion:", graph.node_indices())
///        res_manual = graph.add_node(None)
///        print("After adding a new node:", graph.node_indices())
///
/// Additionally, each node and edge contains an arbitrary Python object as a
/// weight/data payload. You can use the index for access to the data payload
/// as in the following example:
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     graph = rx.PyGraph()
///     data_payload = "An arbitrary Python object"
///     node_index = graph.add_node(data_payload)
///     print("Node Index: %s" % node_index)
///     print(graph[node_index])
///
/// The PyGraph implements the Python mapping protocol for nodes so in
/// addition to access you can also update the data payload with:
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     graph = rx.PyGraph()
///     data_payload = "An arbitrary Python object"
///     node_index = graph.add_node(data_payload)
///     graph[node_index] = "New Payload"
///     print("Node Index: %s" % node_index)
///     print(graph[node_index])
///
/// By default a ``PyGraph`` is a multigraph (meaning there can be parallel
/// edges between nodes) however this can be disabled by setting the
/// ``multigraph`` kwarg to ``False`` when calling the ``PyGraph``
/// constructor. For example::
///
///     import rustworkx as rx
///     graph = rx.PyGraph(multigraph=False)
///
/// This can only be set at ``PyGraph`` initialization and not adjusted after
/// creation. When :attr:`~rustworkx.PyGraph.multigraph` is set to ``False``
/// if a method call is made that would add a parallel edge it will instead
/// update the existing edge's weight/data payload.
///
/// Each ``PyGraph`` object has an :attr:`~.PyGraph.attrs` attribute which is
/// used to contain additional attributes/metadata of the graph instance. By
/// default this is set to ``None`` but can optionally be specified by using the
/// ``attrs`` keyword argument when constructing a new graph::
///
///     graph = rustworkx.PyGraph(attrs=dict(source_path='/tmp/graph.csv'))
///
/// This attribute can be set to any Python object. Additionally, you can access
/// and modify this attribute after creating an object. For example::
///
///     source_path = graph.attrs
///     graph.attrs = {'new_path': '/tmp/new.csv', 'old_path': source_path}
///
/// The maximum number of nodes and edges allowed on a ``PyGraph`` object is
/// :math:`2^{32} - 1` (4,294,967,294) each. Attempting to add more nodes or
/// edges than this will result in an exception being raised.
///
/// :param bool multigraph: When this is set to ``False`` the created PyGraph
///     object will not be a multigraph. When ``False`` if a method call is
///     made that would add parallel edges the the weight/weight from that
///     method call will be used to update the existing edge in place.
/// :param attrs: An optional attributes payload to assign to the
///     :attr:`~.PyGraph.attrs` attribute. This can be any Python object. If
///     it is not specified :attr:`~.PyGraph.attrs` will be set to ``None``.
#[pyclass(mapping, module = "rustworkx", subclass)]
#[pyo3(text_signature = "(/, multigraph=True, attrs=None)")]
#[derive(Clone)]
pub struct PyGraph {
    pub graph: StablePyGraph<Undirected>,
    pub node_removed: bool,
    pub multigraph: bool,
    #[pyo3(get, set)]
    pub attrs: PyObject,
}

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

impl PyGraph {
    fn _add_edge(&mut self, u: NodeIndex, v: NodeIndex, edge: PyObject) -> usize {
        if !self.multigraph {
            let exists = self.graph.find_edge(u, v);
            if let Some(index) = exists {
                let edge_weight = self.graph.edge_weight_mut(index).unwrap();
                *edge_weight = edge;
                return index.index();
            }
        }
        let edge = self.graph.add_edge(u, v, edge);
        edge.index()
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    #[args(multigraph = "true")]
    fn new(py: Python, multigraph: bool, attrs: Option<PyObject>) -> Self {
        PyGraph {
            graph: StablePyGraph::<Undirected>::default(),
            node_removed: false,
            multigraph,
            attrs: attrs.unwrap_or_else(|| py.None()),
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let out_dict = PyDict::new(py);
        let node_dict = PyDict::new(py);
        let mut out_list: Vec<PyObject> = Vec::with_capacity(self.graph.edge_count());
        out_dict.set_item("nodes", node_dict)?;
        out_dict.set_item("nodes_removed", self.node_removed)?;
        out_dict.set_item("multigraph", self.multigraph)?;
        out_dict.set_item("attrs", self.attrs.clone_ref(py))?;
        for node_index in self.graph.node_indices() {
            let node_data = self.graph.node_weight(node_index).unwrap();
            node_dict.set_item(node_index.index(), node_data)?;
        }
        for edge in self.graph.edge_indices() {
            let edge_w = self.graph.edge_weight(edge);
            let endpoints = self.graph.edge_endpoints(edge).unwrap();

            let triplet = (endpoints.0.index(), endpoints.1.index(), edge_w).to_object(py);
            out_list.push(triplet);
        }
        let py_out_list: PyObject = PyList::new(py, out_list).into();
        out_dict.set_item("edges", py_out_list)?;
        Ok(out_dict.into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        self.graph = StablePyGraph::<Undirected>::default();
        let dict_state = state.cast_as::<PyDict>(py)?;
        let nodes_dict = dict_state.get_item("nodes").unwrap().downcast::<PyDict>()?;
        let edges_list = dict_state.get_item("edges").unwrap().downcast::<PyList>()?;
        let nodes_removed_raw = dict_state
            .get_item("nodes_removed")
            .unwrap()
            .downcast::<PyBool>()?;
        self.node_removed = nodes_removed_raw.extract()?;
        let multigraph_raw = dict_state
            .get_item("multigraph")
            .unwrap()
            .downcast::<PyBool>()?;
        self.multigraph = multigraph_raw.extract()?;
        let attrs = match dict_state.get_item("attrs") {
            Some(attr) => attr.into(),
            None => py.None(),
        };
        self.attrs = attrs;

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
            let raw_p_index = edge.get_item(0)?.downcast::<PyLong>()?;
            let parent: usize = raw_p_index.extract()?;
            let p_index = NodeIndex::new(parent);
            let raw_c_index = edge.get_item(1)?.downcast::<PyLong>()?;
            let child: usize = raw_c_index.extract()?;
            let c_index = NodeIndex::new(child);
            let edge_data = edge.get_item(2)?;

            self.graph.add_edge(p_index, c_index, edge_data.into());
        }
        Ok(())
    }

    /// Whether the graph is a multigraph (allows multiple edges between
    /// nodes) or not
    ///
    /// If set to ``False`` multiple edges between nodes are not allowed and
    /// calls that would add a parallel edge will instead update the existing
    /// edge
    #[getter]
    fn multigraph(&self) -> bool {
        self.multigraph
    }

    /// Detect if the graph has parallel edges or not
    ///
    /// :returns: ``True`` if the graph has parallel edges, otherwise ``False``
    /// :rtype: bool
    #[pyo3(text_signature = "(self)")]
    fn has_parallel_edges(&self) -> bool {
        if !self.multigraph {
            return false;
        }
        let mut edges: HashSet<[NodeIndex; 2]> =
            HashSet::with_capacity(2 * self.graph.edge_count());
        for edge in self.graph.edge_references() {
            let endpoints = [edge.source(), edge.target()];
            let endpoints_rev = [edge.target(), edge.source()];
            if edges.contains(&endpoints) || edges.contains(&endpoints_rev) {
                return true;
            }
            edges.insert(endpoints);
            edges.insert(endpoints_rev);
        }
        false
    }

    /// Return the number of nodes in the graph
    #[pyo3(text_signature = "(self)")]
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// Return the number of edges in the graph
    #[pyo3(text_signature = "(self)")]
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Return a list of all edge data.
    ///
    /// :returns: A list of all the edge data objects in the graph
    /// :rtype: list
    #[pyo3(text_signature = "(self)")]
    pub fn edges(&self) -> Vec<&PyObject> {
        self.graph
            .edge_indices()
            .map(|edge| self.graph.edge_weight(edge).unwrap())
            .collect()
    }

    /// Return a list of all edge indices.
    ///
    /// :returns: A list of all the edge indices in the graph
    /// :rtype: EdgeIndices
    #[pyo3(text_signature = "(self)")]
    pub fn edge_indices(&self) -> EdgeIndices {
        EdgeIndices {
            edges: self.graph.edge_indices().map(|edge| edge.index()).collect(),
        }
    }

    /// Return a list of all node data.
    ///
    /// :returns: A list of all the node data objects in the graph
    /// :rtype: list
    #[pyo3(text_signature = "(self)")]
    pub fn nodes(&self) -> Vec<&PyObject> {
        self.graph
            .node_indices()
            .map(|node| self.graph.node_weight(node).unwrap())
            .collect()
    }

    /// Return a list of all node indices.
    ///
    /// :returns: A list of all the node indices in the graph
    /// :rtype: NodeIndices
    #[pyo3(text_signature = "(self)")]
    pub fn node_indices(&self) -> NodeIndices {
        NodeIndices {
            nodes: self.graph.node_indices().map(|node| node.index()).collect(),
        }
    }

    /// Return a list of all node indices.
    ///
    /// .. note::
    ///
    ///     This is identical to :meth:`.node_indices()`, which is the
    ///     preferred method to get the node indices in the graph. This
    ///     exists for backwards compatibility with earlier releases.
    ///
    /// :returns: A list of all the node indices in the graph
    /// :rtype: NodeIndices
    #[pyo3(text_signature = "(self)")]
    pub fn node_indexes(&self) -> NodeIndices {
        self.node_indices()
    }

    /// Return True if there is an edge between ``node_a`` and ``node_b``.
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node
    ///
    /// :returns: True if there is an edge false if there is no edge
    /// :rtype: bool
    #[pyo3(text_signature = "(self, node_a, node_b, /)")]
    pub fn has_edge(&self, node_a: usize, node_b: usize) -> bool {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        self.graph.find_edge(index_a, index_b).is_some()
    }

    ///  Return the edge data for the edge between 2 nodes.
    ///
    ///  Note if there are multiple edges between the nodes only one will be
    ///  returned. To get all edge data objects use
    ///  :meth:`~rustworkx.PyGraph.get_all_edge_data`
    ///
    /// :param int node_a: The index for the first node
    /// :param int node_b: The index for the second node
    ///
    /// :returns: The data object set for the edge
    /// :raises NoEdgeBetweenNodes: when there is no edge between the provided
    ///     nodes
    #[pyo3(text_signature = "(self, node_a, node_b, /)")]
    pub fn get_edge_data(&self, node_a: usize, node_b: usize) -> PyResult<&PyObject> {
        let index_a = NodeIndex::new(node_a);
        let index_b = NodeIndex::new(node_b);
        let edge_index = match self.graph.find_edge(index_a, index_b) {
            Some(edge_index) => edge_index,
            None => return Err(NoEdgeBetweenNodes::new_err("No edge found between nodes")),
        };

        let data = self.graph.edge_weight(edge_index).unwrap();
        Ok(data)
    }

    /// Return the list of edge indices incident to a provided node
    ///
    /// You can later retrieve the data payload of this edge with
    /// :meth:`~rustworkx.PyGraph.get_edge_data_by_index` or its
    /// endpoints with :meth:`~rustworkx.PyGraph.get_edge_endpoints_by_index`.
    ///
    /// :param int node: The node index to get incident edges from. If
    ///     this node index is not present in the graph this method will
    ///     return an empty list and not error.
    ///
    /// :returns: A list of the edge indices incident to a node in the graph
    /// :rtype: EdgeIndices
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn incident_edges(&self, node: usize) -> EdgeIndices {
        EdgeIndices {
            edges: self
                .graph
                .edges(NodeIndex::new(node))
                .map(|e| e.id().index())
                .collect(),
        }
    }

    /// Return the index map of edges incident to a provided node
    ///
    /// :param int node: The node index to get incident edges from. If
    ///     this node index is not present in the graph this method will
    ///     return an empty mapping and not error.
    ///
    /// :returns: A mapping of incident edge indices to the tuple
    ///     ``(source, target, data)``. The source endpoint node index in
    ///     this tuple will always be ``node`` to ensure you can easily
    ///     identify that the neighbor node is the second element in the
    ///     tuple for a given edge index.
    /// :rtype: EdgeIndexMap
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn incident_edge_index_map(&self, py: Python, node: usize) -> EdgeIndexMap {
        let node_index = NodeIndex::new(node);
        EdgeIndexMap {
            edge_map: self
                .graph
                .edges_directed(node_index, petgraph::Direction::Outgoing)
                .map(|edge| {
                    (
                        edge.id().index(),
                        (
                            edge.source().index(),
                            edge.target().index(),
                            edge.weight().clone_ref(py),
                        ),
                    )
                })
                .collect(),
        }
    }

    /// Get the endpoint indices and edge data for all edges of a node.
    ///
    /// This will return a list of tuples with the parent index, the node index
    /// and the edge data. This can be used to recreate add_edge() calls. As
    /// :class:`~rustworkx.PyGraph` is undirected this will return all edges
    /// with the second endpoint node index always being ``node``.
    ///
    /// :param int node: The index of the node to get the edges for
    ///
    /// :returns: A list of tuples of the form:
    ///     ``(parent_index, node_index, edge_data)```
    /// :rtype: WeightedEdgeList
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn in_edges(&self, py: Python, node: usize) -> WeightedEdgeList {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Incoming;
        let raw_edges = self.graph.edges_directed(index, dir);
        let out_list: Vec<(usize, usize, PyObject)> = raw_edges
            .map(|x| (x.source().index(), node, x.weight().clone_ref(py)))
            .collect();
        WeightedEdgeList { edges: out_list }
    }

    /// Get the endpoint indices and edge data for all edges of a node.
    ///
    /// This will return a list of tuples with the child index, the node index
    /// and the edge data. This can be used to recreate add_edge() calls. As
    /// :class:`~rustworkx.PyGraph` is undirected this will return all edges
    /// with the first endpoint node index always being ``node``.
    ///
    /// :param int node: The index of the node to get the edges for
    ///
    /// :returns out_edges: A list of tuples of the form:
    ///     ```(node_index, child_index, edge_data)```
    /// :rtype: WeightedEdgeList
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn out_edges(&self, py: Python, node: usize) -> WeightedEdgeList {
        let index = NodeIndex::new(node);
        let dir = petgraph::Direction::Outgoing;
        let raw_edges = self.graph.edges_directed(index, dir);
        let out_list: Vec<(usize, usize, PyObject)> = raw_edges
            .map(|x| (node, x.target().index(), x.weight().clone_ref(py)))
            .collect();
        WeightedEdgeList { edges: out_list }
    }

    /// Return the edge data for the edge by its given index
    ///
    /// :param int edge_index: The edge index to get the data for
    ///
    /// :returns: The data object for the edge
    /// :raises IndexError: when there is no edge present with the provided
    ///     index
    #[pyo3(text_signature = "(self, edge_index, /)")]
    pub fn get_edge_data_by_index(&self, edge_index: usize) -> PyResult<&PyObject> {
        let data = match self.graph.edge_weight(EdgeIndex::new(edge_index)) {
            Some(data) => data,
            None => {
                return Err(PyIndexError::new_err(format!(
                    "Provided edge index {} is not present in the graph",
                    edge_index
                )));
            }
        };
        Ok(data)
    }

    /// Return the edge endpoints for the edge by its given index
    ///
    /// :param int edge_index: The edge index to get the endpoints for
    ///
    /// :returns: The endpoint tuple for the edge
    /// :rtype: tuple
    /// :raises IndexError: when there is no edge present with the provided
    ///     index
    #[pyo3(text_signature = "(self, edge_index, /)")]
    pub fn get_edge_endpoints_by_index(&self, edge_index: usize) -> PyResult<(usize, usize)> {
        let endpoints = match self.graph.edge_endpoints(EdgeIndex::new(edge_index)) {
            Some(endpoints) => (endpoints.0.index(), endpoints.1.index()),
            None => {
                return Err(PyIndexError::new_err(format!(
                    "Provided edge index {} is not present in the graph",
                    edge_index
                )));
            }
        };
        Ok(endpoints)
    }

    /// Update an edge's weight/payload in place
    ///
    /// If there are parallel edges in the graph only one edge will be updated.
    /// if you need to update a specific edge or need to ensure all parallel
    /// edges get updated you should use
    /// :meth:`~rustworkx.PyGraph.update_edge_by_index` instead.
    ///
    /// :param int source: The index for the first node
    /// :param int target: The index for the second node
    ///
    /// :raises NoEdgeBetweenNodes: When there is no edge between nodes
    #[pyo3(text_signature = "(self, source, target, edge /)")]
    pub fn update_edge(&mut self, source: usize, target: usize, edge: PyObject) -> PyResult<()> {
        let index_a = NodeIndex::new(source);
        let index_b = NodeIndex::new(target);
        let edge_index = match self.graph.find_edge(index_a, index_b) {
            Some(edge_index) => edge_index,
            None => return Err(NoEdgeBetweenNodes::new_err("No edge found between nodes")),
        };
        let data = self.graph.edge_weight_mut(edge_index).unwrap();
        *data = edge;
        Ok(())
    }

    /// Update an edge's weight/data payload in place by the edge index
    ///
    /// :param int edge_index: The index for the edge
    /// :param object edge: The data payload/weight to update the edge with
    ///
    /// :raises IndexError: when there is no edge present with the provided
    ///     index
    #[pyo3(text_signature = "(self, edge_index, edge, /)")]
    pub fn update_edge_by_index(&mut self, edge_index: usize, edge: PyObject) -> PyResult<()> {
        match self.graph.edge_weight_mut(EdgeIndex::new(edge_index)) {
            Some(data) => *data = edge,
            None => return Err(PyIndexError::new_err("No edge found for index")),
        };
        Ok(())
    }

    /// Return the node data for a given node index
    ///
    /// :param int node: The index for the node
    ///
    /// :returns: The data object set for that node
    /// :raises IndexError: when an invalid node index is provided
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn get_node_data(&self, node: usize) -> PyResult<&PyObject> {
        let index = NodeIndex::new(node);
        let node = match self.graph.node_weight(index) {
            Some(node) => node,
            None => return Err(PyIndexError::new_err("No node found for index")),
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
    #[pyo3(text_signature = "(self, node_a, node_b, /)")]
    pub fn get_all_edge_data(&self, node_a: usize, node_b: usize) -> PyResult<Vec<&PyObject>> {
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
    /// :returns: An edge list without weights
    /// :rtype: EdgeList
    #[pyo3(text_signature = "(self)")]
    pub fn edge_list(&self) -> EdgeList {
        EdgeList {
            edges: self
                .graph
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
    #[pyo3(text_signature = "(self)")]
    pub fn weighted_edge_list(&self, py: Python) -> WeightedEdgeList {
        WeightedEdgeList {
            edges: self
                .graph
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

    /// Get an edge index map
    ///
    /// Returns a read only mapping from edge indices to the weighted edge
    /// tuple. The return is a mapping of the form:
    /// ``{0: (0, 1, "weight"), 1: (2, 3, 2.3)}``
    ///
    /// :returns: An edge index map
    /// :rtype: EdgeIndexMap
    #[pyo3(text_signature = "(self)")]
    pub fn edge_index_map(&self, py: Python) -> EdgeIndexMap {
        EdgeIndexMap {
            edge_map: self
                .graph
                .edge_references()
                .map(|edge| {
                    (
                        edge.id().index(),
                        (
                            edge.source().index(),
                            edge.target().index(),
                            edge.weight().clone_ref(py),
                        ),
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
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn remove_node(&mut self, node: usize) -> PyResult<()> {
        let index = NodeIndex::new(node);
        self.graph.remove_node(index);
        self.node_removed = true;
        Ok(())
    }

    /// Add an edge between 2 nodes.
    ///
    /// If :attr:`~rustworkx.PyGraph.multigraph` is ``False`` and an edge already
    /// exists between ``node_a`` and ``node_b`` the weight/payload of that
    /// existing edge will be updated to be ``edge``.
    ///
    /// :param int node_a: Index of the parent node
    /// :param int node_b: Index of the child node
    /// :param edge: The object to set as the data for the edge. It can be any
    ///     python object.
    ///
    /// :returns: The edge index for the newly created (or updated in the case
    ///     of an existing edge with ``multigraph=False``) edge.
    /// :rtype: int
    #[pyo3(text_signature = "(self, node_a, node_b, edge, /)")]
    pub fn add_edge(&mut self, node_a: usize, node_b: usize, edge: PyObject) -> usize {
        let p_index = NodeIndex::new(node_a);
        let c_index = NodeIndex::new(node_b);
        self._add_edge(p_index, c_index, edge)
    }

    /// Add new edges to the graph.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(node_a, node_b, obj)`` to attach to the graph. ``node_a`` and
    ///     ``node_b`` are integer indices describing where an edge should be
    ///     added, and ``obj`` is the python object for the edge data.
    ///
    /// If :attr:`~rustworkx.PyGraph.multigraph` is ``False`` and an edge already
    /// exists between ``node_a`` and ``node_b`` the weight/payload of that
    /// existing edge will be updated to be ``edge``. This will occur in order
    /// from ``obj_list`` so if there are multiple parallel edges in ``obj_list``
    /// the last entry will be used.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[pyo3(text_signature = "(self, obj_list, /)")]
    pub fn add_edges_from(&mut self, obj_list: Vec<(usize, usize, PyObject)>) -> EdgeIndices {
        let mut out_list: Vec<usize> = Vec::with_capacity(obj_list.len());
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            out_list.push(self._add_edge(p_index, c_index, obj.2));
        }
        EdgeIndices { edges: out_list }
    }

    /// Add new edges to the graph without python data.
    ///
    /// :param list obj_list: A list of tuples of the form
    ///     ``(parent, child)`` to attach to the graph. ``parent`` and
    ///     ``child`` are integer indices describing where an edge should be
    ///     added. Unlike :meth:`add_edges_from` there is no data payload and
    ///     when the edge is created None will be used.
    ///
    /// If :attr:`~rustworkx.PyGraph.multigraph` is ``False`` and an edge already
    /// exists between ``node_a`` and ``node_b`` the weight/payload of that
    /// existing edge will be updated to be ``None``.
    ///
    /// :returns: A list of int indices of the newly created edges
    /// :rtype: list
    #[pyo3(text_signature = "(self, obj_list, /)")]
    pub fn add_edges_from_no_data(
        &mut self,
        py: Python,
        obj_list: Vec<(usize, usize)>,
    ) -> EdgeIndices {
        let mut out_list: Vec<usize> = Vec::with_capacity(obj_list.len());
        for obj in obj_list {
            let p_index = NodeIndex::new(obj.0);
            let c_index = NodeIndex::new(obj.1);
            out_list.push(self._add_edge(p_index, c_index, py.None()));
        }
        EdgeIndices { edges: out_list }
    }

    /// Extend graph from an edge list
    ///
    /// This method differs from :meth:`add_edges_from_no_data` in that it will
    /// add nodes if a node index is not present in the edge list.
    ///
    /// If :attr:`~rustworkx.PyGraph.multigraph` is ``False`` and an edge already
    /// exists between ``node_a`` and ``node_b`` the weight/payload of that
    /// existing edge will be updated to be ``None``.
    ///
    /// :param list edge_list: A list of tuples of the form ``(source, target)``
    ///     where source and target are integer node indices. If the node index
    ///     is not present in the graph, nodes will be added (with a node
    ///     weight of ``None``) to that index.
    #[pyo3(text_signature = "(self, edge_list, /)")]
    pub fn extend_from_edge_list(&mut self, py: Python, edge_list: Vec<(usize, usize)>) {
        for (source, target) in edge_list {
            let max_index = cmp::max(source, target);
            while max_index >= self.node_count() {
                self.graph.add_node(py.None());
            }
            let source_index = NodeIndex::new(source);
            let target_index = NodeIndex::new(target);
            self._add_edge(source_index, target_index, py.None());
        }
    }

    /// Extend graph from a weighted edge list
    ///
    /// This method differs from :meth:`add_edges_from` in that it will
    /// add nodes if a node index is not present in the edge list.
    ///
    /// If :attr:`~rustworkx.PyGraph.multigraph` is ``False`` and an edge already
    /// exists between ``node_a`` and ``node_b`` the weight/payload of that
    /// existing edge will be updated to be ``edge``. This will occur in order
    /// from ``obj_list`` so if there are multiple parallel edges in ``obj_list``
    /// the last entry will be used.
    ///
    /// :param list edge_list: A list of tuples of the form
    ///     ``(source, target, weight)`` where source and target are integer
    ///     node indices. If the node index is not present in the graph,
    ///     nodes will be added (with a node weight of ``None``) to that index.
    #[pyo3(text_signature = "(self, edge_lsit, /)")]
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
            let source_index = NodeIndex::new(source);
            let target_index = NodeIndex::new(target);
            self._add_edge(source_index, target_index, weight);
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
    #[pyo3(text_signature = "(self, node_a, node_b, /)")]
    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) -> PyResult<()> {
        let p_index = NodeIndex::new(node_a);
        let c_index = NodeIndex::new(node_b);
        let edge_index = match self.graph.find_edge(p_index, c_index) {
            Some(edge_index) => edge_index,
            None => return Err(NoEdgeBetweenNodes::new_err("No edge found between nodes")),
        };
        self.graph.remove_edge(edge_index);
        Ok(())
    }

    /// Remove an edge identified by the provided index
    ///
    /// :param int edge: The index of the edge to remove
    #[pyo3(text_signature = "(self, edge, /)")]
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
    ///
    /// :raises NoEdgeBetweenNodes: If there are no edges between a specified
    ///     pair of nodes.
    #[pyo3(text_signature = "(self, index_list, /)")]
    pub fn remove_edges_from(&mut self, index_list: Vec<(usize, usize)>) -> PyResult<()> {
        for (p_index, c_index) in index_list
            .iter()
            .map(|(x, y)| (NodeIndex::new(*x), NodeIndex::new(*y)))
        {
            let edge_index = match self.graph.find_edge(p_index, c_index) {
                Some(edge_index) => edge_index,
                None => return Err(NoEdgeBetweenNodes::new_err("No edge found between nodes")),
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
    #[pyo3(text_signature = "(self, obj, /)")]
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
    #[pyo3(text_signature = "(self, obj_list, /)")]
    pub fn add_nodes_from(&mut self, obj_list: Vec<PyObject>) -> NodeIndices {
        let out_list: Vec<usize> = obj_list
            .into_iter()
            .map(|obj| self.graph.add_node(obj).index())
            .collect();
        NodeIndices { nodes: out_list }
    }

    /// Remove nodes from the graph.
    ///
    /// If a node index in the list is not present in the graph it will be
    /// ignored.
    ///
    /// :param list index_list: A list of node indicies to remove from the
    ///     the graph
    #[pyo3(text_signature = "(self, index_list, /)")]
    pub fn remove_nodes_from(&mut self, index_list: Vec<usize>) -> PyResult<()> {
        for node in index_list.iter().map(|x| NodeIndex::new(*x)) {
            self.graph.remove_node(node);
        }
        Ok(())
    }

    /// Find node within this graph given a specific weight
    ///
    /// This algorithm has a worst case of O(n) since it searches the node
    /// indices in order. If there is more than one node in the graph with the
    /// same weight only the first match (by node index) will be returned.
    ///
    /// :param obj: The weight to look for in the graph.
    ///
    /// :returns: the index of the first node in the graph that is equal to the
    ///     weight. If no match is found ``None`` will be returned.
    /// :rtype: int
    #[pyo3(text_signature = "(self, obj, /)")]
    pub fn find_node_by_weight(&self, py: Python, obj: PyObject) -> PyResult<Option<usize>> {
        find_node_by_weight(py, &self.graph, &obj).map(|node| node.map(|x| x.index()))
    }

    /// Get the index and data for the neighbors of a node.
    ///
    /// This will return a dictionary where the keys are the node indices of
    /// the adjacent nodes (inbound or outbound) and the value is the edge data
    /// objects between that adjacent node and the provided node. Note, that
    /// in the case of multigraphs only a single edge data object will be
    /// returned
    ///
    /// :param int node: The index of the node to get the neighbors
    ///
    /// :returns neighbors: A dictionary where the keys are node indices and
    ///     the value is the edge data object for all nodes that share an
    ///     edge with the specified node.
    /// :rtype: dict
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn adj(&mut self, node: usize) -> DictMap<usize, &PyObject> {
        let index = NodeIndex::new(node);
        self.graph
            .edges_directed(index, petgraph::Direction::Outgoing)
            .map(|edge| (edge.target().index(), edge.weight()))
            .collect()
    }

    /// Get the neighbors of a node.
    ///
    /// This with return a list of neighbor node indices
    ///
    /// :param int node: The index of the node to get the neighbors of
    ///
    /// :returns: A list of the neighbor node indicies
    /// :rtype: NodeIndices
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn neighbors(&self, node: usize) -> NodeIndices {
        NodeIndices {
            nodes: self
                .graph
                .neighbors(NodeIndex::new(node))
                .map(|node| node.index())
                .collect::<HashSet<usize>>()
                .drain()
                .collect(),
        }
    }

    /// Get the degree for a node
    ///
    /// :param int node: The index of the  node to find the inbound degree of
    ///
    /// :returns degree: The inbound degree for the specified node
    /// :rtype: int
    #[pyo3(text_signature = "(self, node, /)")]
    pub fn degree(&self, node: usize) -> usize {
        let index = NodeIndex::new(node);
        let neighbors = self.graph.edges(index);
        neighbors.fold(0, |count, edge| {
            if edge.source() == edge.target() {
                return count + 2;
            }
            count + 1
        })
    }

    /// Generate a new :class:`~rustworkx.PyDiGraph` object from this graph
    ///
    /// This will create a new :class:`~rustworkx.PyDiGraph` object from this
    /// graph. All edges in this graph will result in a bidirectional edge
    /// pair in the output graph.
    ///
    /// .. note::
    ///
    ///     The node indices in the output :class:`~rustworkx.PyDiGraph` may
    ///     differ if nodes have been removed.
    ///
    /// :returns: A new :class:`~rustworkx.PyDiGraph` object with a
    ///     bidirectional edge pair for each edge in this graph. Also all
    ///     node and edge weights/data payloads are copied by reference to
    ///     the output graph
    /// :rtype: PyDiGraph
    #[pyo3(text_signature = "(self)")]
    pub fn to_directed(&self, py: Python) -> crate::digraph::PyDiGraph {
        let node_count = self.node_count();
        let mut new_graph =
            StablePyGraph::<Directed>::with_capacity(node_count, 2 * self.graph.edge_count());
        let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(node_count);
        for node_index in self.graph.node_indices() {
            let node = self.graph[node_index].clone_ref(py);
            let new_index = new_graph.add_node(node);
            node_map.insert(node_index, new_index);
        }
        for edge in self.graph.edge_references() {
            let &source = node_map.get(&edge.source()).unwrap();
            let &target = node_map.get(&edge.target()).unwrap();
            let weight = edge.weight();
            new_graph.add_edge(source, target, weight.clone_ref(py));
            new_graph.add_edge(target, source, weight.clone_ref(py));
        }
        crate::digraph::PyDiGraph {
            graph: new_graph,
            node_removed: false,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            multigraph: self.multigraph,
            attrs: py.None(),
        }
    }

    /// Generate a dot file from the graph
    ///
    /// :param node_attr: A callable that will take in a node data object
    ///     and return a dictionary of attributes to be associated with the
    ///     node in the dot file. The key and value of this dictionary **must**
    ///     be a string. If they're not strings rustworkx will raise TypeError
    ///     (unfortunately without an error message because of current
    ///     limitations in the PyO3 type checking)
    /// :param edge_attr: A callable that will take in an edge data object
    ///     and return a dictionary of attributes to be associated with the
    ///     node in the dot file. The key and value of this dictionary **must**
    ///     be a string. If they're not strings rustworkx will raise TypeError
    ///     (unfortunately without an error message because of current
    ///     limitations in the PyO3 type checking)
    /// :param dict graph_attr: An optional dictionary that specifies any graph
    ///     attributes for the output dot file. The key and value of this
    ///     dictionary **must** be a string. If they're not strings rustworkx
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
    /// :class:`rustworkx.PyGraph` object. For example:
    ///
    /// .. jupyter-execute::
    ///
    ///   import os
    ///   import tempfile
    ///
    ///   import pydot
    ///   from PIL import Image
    ///
    ///   import rustworkx as rx
    ///
    ///   graph = rx.undirected_gnp_random_graph(15, .25)
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
    #[pyo3(
        text_signature = "(self, /, node_attr=None, edge_attr=None, graph_attr=None, filename=None)"
    )]
    pub fn to_dot(
        &self,
        py: Python,
        node_attr: Option<PyObject>,
        edge_attr: Option<PyObject>,
        graph_attr: Option<BTreeMap<String, String>>,
        filename: Option<String>,
    ) -> PyResult<Option<PyObject>> {
        match filename {
            Some(filename) => {
                let mut file = File::create(filename)?;
                build_dot(py, &self.graph, &mut file, graph_attr, node_attr, edge_attr)?;
                Ok(None)
            }
            None => {
                let mut file = Vec::<u8>::new();
                build_dot(py, &self.graph, &mut file, graph_attr, node_attr, edge_attr)?;
                Ok(Some(
                    PyString::new(py, str::from_utf8(&file)?).to_object(py),
                ))
            }
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
    /// :param bool labels: If set to ``True`` the first two separated fields
    ///     will be treated as string labels uniquely identifying a node
    ///     instead of node indices.
    ///
    /// For example:
    ///
    /// .. jupyter-execute::
    ///
    ///   import tempfile
    ///
    ///   import rustworkx as rx
    ///   from rustworkx.visualization import mpl_draw
    ///
    ///   with tempfile.NamedTemporaryFile('wt') as fd:
    ///       path = fd.name
    ///       fd.write('0 1\n')
    ///       fd.write('0 2\n')
    ///       fd.write('0 3\n')
    ///       fd.write('1 2\n')
    ///       fd.write('2 3\n')
    ///       fd.flush()
    ///       graph = rx.PyGraph.read_edge_list(path)
    ///   mpl_draw(graph)
    ///
    #[staticmethod]
    #[args(labels = "false")]
    #[pyo3(text_signature = "(path, /, comment=None, deliminator=None, labels=False)")]
    pub fn read_edge_list(
        py: Python,
        path: &str,
        comment: Option<String>,
        deliminator: Option<String>,
        labels: bool,
    ) -> PyResult<PyGraph> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut out_graph = StablePyGraph::<Undirected>::default();
        let mut label_map: HashMap<String, usize> = HashMap::new();
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
            let src: usize;
            let target: usize;
            if labels {
                let src_str = pieces[0];
                let target_str = pieces[1];
                src = match label_map.get(src_str) {
                    Some(index) => *index,
                    None => {
                        let index = out_graph.add_node(src_str.to_object(py)).index();
                        label_map.insert(src_str.to_string(), index);
                        index
                    }
                };
                target = match label_map.get(target_str) {
                    Some(index) => *index,
                    None => {
                        let index = out_graph.add_node(target_str.to_object(py)).index();
                        label_map.insert(target_str.to_string(), index);
                        index
                    }
                };
            } else {
                src = pieces[0].parse::<usize>()?;
                target = pieces[1].parse::<usize>()?;
                let max_index = cmp::max(src, target);
                // Add nodes to graph
                while max_index >= out_graph.node_count() {
                    out_graph.add_node(py.None());
                }
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
            out_graph.add_edge(NodeIndex::new(src), NodeIndex::new(target), weight);
        }
        Ok(PyGraph {
            graph: out_graph,
            node_removed: false,
            multigraph: true,
            attrs: py.None(),
        })
    }

    /// Write an edge list file from the PyGraph object
    ///
    /// :param str path: The path to write the output file to
    /// :param str deliminator: The optional character to use as a deliminator
    ///     if not specified ``" "`` is used.
    /// :param callable weight_fn: An optional callback function that will be
    ///     passed an edge's data payload/weight object and is expected to
    ///     return a string (a ``TypeError`` will be raised if it doesn't
    ///     return a string). If specified the weight in the output file
    ///     for each edge will be set to the returned string.
    ///
    ///  For example:
    ///
    ///  .. jupyter-execute::
    ///
    ///     import os
    ///     import tempfile
    ///
    ///     import rustworkx as rx
    ///
    ///     graph = rx.generators.path_graph(5)
    ///     path = os.path.join(tempfile.gettempdir(), "edge_list")
    ///     graph.write_edge_list(path, deliminator=',')
    ///     # Print file contents
    ///     with open(path, 'rt') as edge_file:
    ///         print(edge_file.read())
    ///
    #[pyo3(text_signature = "(self, path, /, deliminator=None, weight_fn=None)")]
    pub fn write_edge_list(
        &self,
        py: Python,
        path: &str,
        deliminator: Option<char>,
        weight_fn: Option<PyObject>,
    ) -> PyResult<()> {
        let file = File::create(path)?;
        let mut buf_writer = BufWriter::new(file);
        let delim = match deliminator {
            Some(delim) => delim.to_string(),
            None => " ".to_string(),
        };

        for edge in self.graph.edge_references() {
            buf_writer.write_all(
                format!(
                    "{}{}{}",
                    edge.source().index(),
                    delim,
                    edge.target().index()
                )
                .as_bytes(),
            )?;
            match weight_callable(py, &weight_fn, edge.weight(), None as Option<String>)? {
                Some(weight) => buf_writer.write_all(format!("{}{}\n", delim, weight).as_bytes()),
                None => buf_writer.write_all(b"\n"),
            }?;
        }
        buf_writer.flush()?;
        Ok(())
    }

    /// Create a new :class:`~rustworkx.PyGraph` object from an adjacency matrix
    /// with matrix elements of type ``float``
    ///
    /// This method can be used to construct a new :class:`~rustworkx.PyGraph`
    /// object from an input adjacency matrix. The node weights will be the
    /// index from the matrix. The edge weights will be a float value of the
    /// value from the matrix.
    ///
    /// This differs from the
    /// :meth:`~rustworkx.PyGraph.from_complex_adjacency_matrix` in that the
    /// type of the elements of input matrix must be a ``float`` (specifically
    /// a ``numpy.float64``) and the output graph edge weights will be ``float``
    /// too. While in :meth:`~rustworkx.PyGraph.from_complex_adjacency_matrix`
    /// the matrix elements are of type ``complex`` (specifically
    /// ``numpy.complex128``) and the edge weights in the output graph will be
    /// ``complex`` too.
    ///
    /// :param ndarray matrix: The input numpy array adjacency matrix to create
    ///     a new :class:`~rustworkx.PyGraph` object from. It must be a 2
    ///     dimensional array and be a ``float``/``np.float64`` data type.
    /// :param float null_value: An optional float that will treated as a null
    ///     value. If any element in the input matrix is this value it will be
    ///     treated as not an edge. By default this is ``0.0``.
    ///
    /// :returns: A new graph object generated from the adjacency matrix
    /// :rtype: PyGraph
    #[staticmethod]
    #[args(null_value = "0.0")]
    #[pyo3(text_signature = "(matrix, /, null_value=0.0)")]
    pub fn from_adjacency_matrix<'p>(
        py: Python<'p>,
        matrix: PyReadonlyArray2<'p, f64>,
        null_value: f64,
    ) -> PyGraph {
        _from_adjacency_matrix(py, matrix, null_value)
    }

    /// Create a new :class:`~rustworkx.PyGraph` object from an adjacency matrix
    /// with matrix elements of type ``complex``
    ///
    /// This method can be used to construct a new :class:`~rustworkx.PyGraph`
    /// object from an input adjacency matrix. The node weights will be the
    /// index from the matrix. The edge weights will be a complex value of the
    /// value from the matrix.
    ///
    /// This differs from the
    /// :meth:`~rustworkx.PyGraph.from_adjacency_matrix` in that the type of
    /// the elements of the input matrix in this method must be a ``complex``
    /// (specifically a ``numpy.complex128``) and the output graph edge weights
    /// will be ``complex`` too. While in
    /// :meth:`~rustworkx.PyGraph.from_adjacency_matrix` the matrix elements
    /// are of type ``float`` (specifically ``numpy.float64``) and the edge
    /// weights in the output graph will be ``float`` too.
    ///
    /// :param ndarray matrix: The input numpy array adjacency matrix to create
    ///     a new :class:`~rustworkx.PyGraph` object from. It must be a 2
    ///     dimensional array and be a ``complex``/``np.complex128`` data type.
    /// :param float null_value: An optional complex that will treated as a null
    ///     value. If any element in the input matrix is this value it will be
    ///     treated as not an edge. By default this is ``0.0+0.0j``
    ///
    /// :returns: A new graph object generated from the adjacency matrix
    /// :rtype: PyGraph
    #[staticmethod]
    #[args(null_value = "Complex64::zero()")]
    #[pyo3(text_signature = "(matrix, /, null_value=0.0+0.0j)")]
    pub fn from_complex_adjacency_matrix<'p>(
        py: Python<'p>,
        matrix: PyReadonlyArray2<'p, Complex64>,
        null_value: Complex64,
    ) -> PyGraph {
        _from_adjacency_matrix(py, matrix, null_value)
    }

    /// Add another PyGraph object into this PyGraph
    ///
    /// :param PyGraph other: The other PyGraph object to add onto this
    ///     graph.
    /// :param dict node_map: A dictionary mapping node indices from this
    ///     PyGraph object to node indices in the other PyGraph object.
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
    ///   import rustworkx as rx
    ///   from rustworkx.visualization import mpl_draw
    ///
    ///   # Build first graph and visualize:
    ///   graph = rx.PyGraph()
    ///   node_a, node_b, node_c = graph.add_nodes_from(['A', 'B', 'C'])
    ///   graph.add_edges_from([(node_a, node_b, 'A to B'),
    ///                         (node_b, node_c, 'B to C')])
    ///   mpl_draw(graph, with_labels=True, labels=str, edge_labels=str)
    ///
    /// Then build a second one:
    ///
    /// .. jupyter-execute::
    ///
    ///   # Build second graph and visualize:
    ///   other_graph = rx.PyGraph()
    ///   node_d, node_e = other_graph.add_nodes_from(['D', 'E'])
    ///   other_graph.add_edge(node_d, node_e, 'D to E')
    ///   mpl_draw(other_graph, with_labels=True, labels=str, edge_labels=str)
    ///
    /// Finally compose the ``other_graph`` onto ``graph``
    ///
    /// .. jupyter-execute::
    ///
    ///   node_map = {node_b: (node_d, 'B to D')}
    ///   graph.compose(other_graph, node_map)
    ///   mpl_draw(graph, with_labels=True, labels=str, edge_labels=str)
    ///
    #[pyo3(text_signature = "(self, other, node_map, /, node_map_func=None, edge_map_func=None)")]
    pub fn compose(
        &mut self,
        py: Python,
        other: &PyGraph,
        node_map: HashMap<usize, (usize, PyObject)>,
        node_map_func: Option<PyObject>,
        edge_map_func: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let mut new_node_map: DictMap<NodeIndex, NodeIndex> =
            DictMap::with_capacity(other.node_count());

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
            let weight = weight_transform_callable(py, &edge_map_func, edge.weight())?;
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

    /// Substitute a set of nodes with a single new node.
    ///
    /// .. note::
    ///     This method does not preserve the ordering of endpoints in
    ///     edge tuple representations (e.g. the tuples returned from
    ///     :meth:`~rustworkx.PyGraph.edge_list`).
    ///
    /// :param list nodes: A set of nodes to be removed and replaced
    ///     by the new node. Any nodes not in the graph are ignored.
    ///     If empty, this method behaves like :meth:`~PyGraph.add_node`
    ///     (but slower).
    /// :param object obj: The data/weight to associate with the new node.
    /// :param weight_combo_fn: An optional python callable that, when
    ///     specified, is used to merge parallel edges introduced by the
    ///     contraction, which will occur if any two edges between ``nodes``
    ///     and the rest of the graph share an endpoint.
    ///     If this instance of :class:`~rustworkx.PyGraph` is a multigraph,
    ///     leave this unspecified to preserve parallel edges. If unspecified
    ///     when not a multigraph, parallel edges and their weights will be
    ///     combined by choosing one of the edge's weights arbitrarily based
    ///     on an internal iteration order, subject to change.
    /// :returns: The index of the newly created node.
    #[pyo3(text_signature = "(self, nodes, obj, /, weight_combo_fn=None)")]
    pub fn contract_nodes(
        &mut self,
        py: Python,
        nodes: Vec<usize>,
        obj: PyObject,
        weight_combo_fn: Option<PyObject>,
    ) -> PyResult<usize> {
        let mut indices_to_remove: IndexSet<NodeIndex, ahash::RandomState> =
            nodes.into_iter().map(NodeIndex::new).collect();

        // Create new node.
        let node_index = self.graph.add_node(obj);

        // Sanitize new node index from user input.
        indices_to_remove.remove(&node_index);

        // Determine edges for new node.
        // note: `edges_directed` returns all edges with `i` as
        // an endpoint. `Direction::Incoming` configures `edge.target()`
        // to return `i` and `edge.source()` to return the other node.
        let mut edges: Vec<_> = indices_to_remove
            .iter()
            .flat_map(|&i| self.graph.edges_directed(i, Direction::Incoming))
            .filter_map(|edge| {
                let pred = edge.source();
                if !indices_to_remove.contains(&pred) {
                    Some((pred, edge.weight().clone_ref(py)))
                } else {
                    None
                }
            })
            .collect();

        // Remove nodes that will be replaced.
        for index in indices_to_remove {
            self.graph.remove_node(index);
        }

        // If `weight_combo_fn` was specified, merge edges according
        // to that function, even if this is a multigraph. If unspecified,
        // defer parallel edge handling to `add_edge`.
        if let Some(merge_fn) = weight_combo_fn {
            edges = merge_duplicates(edges, |w1, w2| merge_fn.call1(py, (w1, w2)))?;
        }

        for (source, weight) in edges {
            self.add_edge(source.index(), node_index.index(), weight);
        }

        Ok(node_index.index())
    }

    /// Return a new PyGraph object for a subgraph of this graph
    ///
    /// :param list nodes: A list of node indices to generate the subgraph
    ///     from. If a node index is included that is not present in the graph
    ///     it will silently be ignored.
    /// :param preserve_attrs: If set to the True the attributes of the PyGraph
    ///     will be copied by reference to be the attributes of the output
    ///     subgraph. By default this is set to False and the :attr:`~.PyGraph.attrs`
    ///     attribute will be ``None`` in the subgraph.
    ///
    /// :returns: A new PyGraph object representing a subgraph of this graph.
    ///     It is worth noting that node and edge weight/data payloads are
    ///     passed by reference so if you update (not replace) an object used
    ///     as the weight in graph or the subgraph it will also be updated in
    ///     the other.
    /// :rtype: PyGraph
    ///
    #[args(preserve_attrs = "false")]
    #[pyo3(text_signature = "(self, nodes, /, preserve_attrs=False)")]
    pub fn subgraph(&self, py: Python, nodes: Vec<usize>, preserve_attrs: bool) -> PyGraph {
        let node_set: HashSet<usize> = nodes.iter().cloned().collect();
        let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(nodes.len());
        let node_filter = |node: NodeIndex| -> bool { node_set.contains(&node.index()) };
        let mut out_graph = StablePyGraph::<Undirected>::default();
        let filtered = NodeFiltered(&self.graph, node_filter);
        for node in filtered.node_references() {
            let new_node = out_graph.add_node(node.1.clone_ref(py));
            node_map.insert(node.0, new_node);
        }
        for edge in filtered.edge_references() {
            let new_source = *node_map.get(&edge.source()).unwrap();
            let new_target = *node_map.get(&edge.target()).unwrap();
            out_graph.add_edge(new_source, new_target, edge.weight().clone_ref(py));
        }
        let attrs = if preserve_attrs {
            self.attrs.clone_ref(py)
        } else {
            py.None()
        };
        PyGraph {
            graph: out_graph,
            node_removed: false,
            multigraph: self.multigraph,
            attrs,
        }
    }

    /// Return a new PyGraph object for an edge induced subgraph of this graph
    ///
    /// The induced subgraph contains each edge in `edge_list` and each node
    /// incident to any of those edges.
    ///
    /// :param list edge_list: A list of edge tuples (2-tuples with the source
    ///     and target node) to generate the subgraph from. In cases of parallel
    ///     edges for a multigraph all edges between the specified node. In case
    ///     of an edge specified that doesn't exist in the graph it will be
    ///     silently ignored.
    ///
    /// :returns: The edge subgraph
    /// :rtype: PyGraph
    ///
    #[pyo3(text_signature = "(self, edge_list, /)")]
    pub fn edge_subgraph(&self, edge_list: Vec<[usize; 2]>) -> PyGraph {
        // Filter non-existent edges
        let edges: Vec<[usize; 2]> = edge_list
            .into_iter()
            .filter(|x| {
                let source = NodeIndex::new(x[0]);
                let target = NodeIndex::new(x[1]);
                self.graph.find_edge(source, target).is_some()
            })
            .collect();

        let nodes: HashSet<NodeIndex> = edges
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .map(NodeIndex::new)
            .collect();
        let mut edge_set: HashSet<[NodeIndex; 2]> = HashSet::with_capacity(edges.len());
        for edge in edges {
            let source_index = NodeIndex::new(edge[0]);
            let target_index = NodeIndex::new(edge[1]);
            edge_set.insert([source_index, target_index]);
        }
        let mut out_graph = self.clone();
        for node in self
            .graph
            .node_indices()
            .filter(|node| !nodes.contains(node))
        {
            out_graph.graph.remove_node(node);
            out_graph.node_removed = true;
        }
        for edge in self.graph.edge_references().filter(|edge| {
            !edge_set.contains(&[edge.source(), edge.target()])
                && !edge_set.contains(&[edge.target(), edge.source()])
        }) {
            out_graph.graph.remove_edge(edge.id());
        }
        out_graph
    }

    /// Return a shallow copy of the graph
    ///
    /// All node and edge weight/data payloads in the copy will have a
    /// shared reference to the original graph.
    #[pyo3(text_signature = "(self)")]
    pub fn copy(&self) -> PyGraph {
        self.clone()
    }

    /// Return the nmber of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }

    fn __getitem__(&self, idx: usize) -> PyResult<&PyObject> {
        match self.graph.node_weight(NodeIndex::new(idx)) {
            Some(data) => Ok(data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }

    fn __setitem__(&mut self, idx: usize, value: PyObject) -> PyResult<()> {
        let data = match self.graph.node_weight_mut(NodeIndex::new(idx)) {
            Some(node_data) => node_data,
            None => return Err(PyIndexError::new_err("No node found for index")),
        };
        *data = value;
        Ok(())
    }

    fn __delitem__(&mut self, idx: usize) -> PyResult<()> {
        match self.graph.remove_node(NodeIndex::new(idx as usize)) {
            Some(_) => Ok(()),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }

    // Functions to enable Python Garbage Collection

    // Function for PyTypeObject.tp_traverse [1][2] used to tell Python what
    // objects the PyGraph has strong references to.
    //
    // [1] https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
    // [2] https://pyo3.rs/v0.12.4/class/protocols.html#garbage-collector-integration
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for node in self
            .graph
            .node_indices()
            .map(|node| self.graph.node_weight(node).unwrap())
        {
            visit.call(node)?;
        }
        for edge in self
            .graph
            .edge_indices()
            .map(|edge| self.graph.edge_weight(edge).unwrap())
        {
            visit.call(edge)?;
        }
        visit.call(&self.attrs)?;
        Ok(())
    }

    // Function for PyTypeObject.tp_clear [1][2] used to tell Python's GC how
    // to drop all references held by a PyGraph object when the GC needs to
    // break reference cycles.
    //
    // ]1] https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_clear
    // [2] https://pyo3.rs/v0.12.4/class/protocols.html#garbage-collector-integration
    fn __clear__(&mut self, py: Python) {
        self.graph = StablePyGraph::<Undirected>::default();
        self.node_removed = false;
        self.attrs = py.None();
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

fn _from_adjacency_matrix<'p, T>(
    py: Python<'p>,
    matrix: PyReadonlyArray2<'p, T>,
    null_value: T,
) -> PyGraph
where
    T: Copy + std::cmp::PartialEq + numpy::Element + pyo3::ToPyObject + IsNan,
{
    let array = matrix.as_array();
    let shape = array.shape();
    let mut out_graph = StablePyGraph::<Undirected>::default();
    let _node_indices: Vec<NodeIndex> = (0..shape[0])
        .map(|node| out_graph.add_node(node.to_object(py)))
        .collect();
    array
        .axis_iter(Axis(0))
        .enumerate()
        .for_each(|(index, row)| {
            let source_index = NodeIndex::new(index);
            for (target_index, elem) in row.iter().enumerate() {
                if target_index < index {
                    continue;
                }
                if null_value.is_nan() {
                    if !elem.is_nan() {
                        out_graph.add_edge(
                            source_index,
                            NodeIndex::new(target_index),
                            elem.to_object(py),
                        );
                    }
                } else if *elem != null_value {
                    out_graph.add_edge(
                        source_index,
                        NodeIndex::new(target_index),
                        elem.to_object(py),
                    );
                }
            }
        });
    PyGraph {
        graph: out_graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    }
}
