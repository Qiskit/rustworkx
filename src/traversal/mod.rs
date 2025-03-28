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

mod bfs_visit;
pub mod dfs_visit;
mod dijkstra_visit;

use bfs_visit::{bfs_handler, PyBfsVisitor};
use dfs_visit::{dfs_handler, PyDfsVisitor};
use dijkstra_visit::{dijkstra_handler, PyDijkstraVisitor};

use rustworkx_core::traversal::{
    ancestors as core_ancestors, bfs_predecessors as core_bfs_predecessors,
    bfs_successors as core_bfs_successors, breadth_first_search, depth_first_search,
    descendants as core_descendants, dfs_edges, dijkstra_search,
};

use super::{digraph, graph, iterators, CostFn};

use std::convert::TryFrom;

use hashbrown::HashSet;

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::EdgeType;

use crate::iterators::EdgeList;
use crate::StablePyGraph;

fn validate_source_nodes<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    starts: &[NodeIndex],
) -> PyResult<()> {
    for index in starts.iter() {
        if !graph.contains_node(*index) {
            return Err(PyIndexError::new_err(format!(
                "Node source index \"{}\" out of graph bound",
                index.index()
            )));
        }
    }
    Ok(())
}

/// Get an edge list of the tree edges from a depth-first traversal
///
/// The pseudo-code for the DFS algorithm is listed below. The output
/// contains the tree edges found by the procedure.
///
/// ::
///
///     DFS(G, v)
///       let S be a stack
///       label v as discovered
///       PUSH(S, (v, iterator of G.neighbors(v)))
///       while (S != Ø)
///           let (v, iterator) := LAST(S)
///           if hasNext(iterator) then
///               w := next(iterator)
///               if w is not labeled as discovered then
///                   label w as discovered                   # (v, w) is a tree edge
///                   PUSH(S, (w, iterator of G.neighbors(w)))
///           else
///               POP(S)
///       end while
///
/// :param PyDiGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarily and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)", signature = (graph, source=None))]
pub fn digraph_dfs_edges(graph: &digraph::PyDiGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: dfs_edges(&graph.graph, source.map(NodeIndex::new)),
    }
}

/// Get an edge list of the tree edges from a depth-first traversal
///
/// The pseudo-code for the DFS algorithm is listed below. The output
/// contains the tree edges found by the procedure.
///
/// ::
///
///     DFS(G, v)
///       let S be a stack
///       label v as discovered
///       PUSH(S, (v, iterator of G.neighbors(v)))
///       while (S != Ø)
///           let (v, iterator) := LAST(S)
///           if hasNext(iterator) then
///               w := next(iterator)
///               if w is not labeled as discovered then
///                   label w as discovered                   # (v, w) is a tree edge
///                   PUSH(S, (w, iterator of G.neighbors(w)))
///           else
///               POP(S)
///       end while
///
/// .. note::
///
///    If the input is an undirected graph with a single connected component,
///    the output of this function is a spanning tree.
///
/// :param PyGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarily and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)", signature = (graph, source=None))]
pub fn graph_dfs_edges(graph: &graph::PyGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: dfs_edges(&graph.graph, source.map(NodeIndex::new)),
    }
}

/// Return successors in a breadth-first-search from a source node.
///
/// The return format is ``[(Parent Node, [Children Nodes])]`` in a bfs order
/// from the source node provided.
///
/// :param PyDiGraph graph: The DAG to get the bfs_successors from
/// :param int node: The index of the dag node to get the bfs successors for
///
/// :returns: A list of nodes's data and their children in bfs order. The
///     BFSSuccessors class that is returned is a custom container class that
///     implements the sequence protocol. This can be used as a python list
///     with index based access.
/// :rtype: BFSSuccessors
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn bfs_successors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> iterators::BFSSuccessors {
    let index = NodeIndex::new(node);
    let out_list = core_bfs_successors(&graph.graph, index)
        .filter_map(|(nx, succ_list)| {
            if succ_list.is_empty() {
                None
            } else {
                Some((
                    graph.graph.node_weight(nx).unwrap().clone_ref(py),
                    succ_list
                        .into_iter()
                        .map(|pred| graph.graph.node_weight(pred).unwrap().clone_ref(py))
                        .collect(),
                ))
            }
        })
        .collect();
    iterators::BFSSuccessors {
        bfs_successors: out_list,
    }
}

/// Return predecessors in a breadth-first-search from a source node.
///
/// The return format is ``[(Parent Node, [Children Nodes])]`` in a bfs order
/// from the source node provided.
///
/// :param PyDiGraph graph: The DAG to get the bfs_predecessors from
/// :param int node: The index of the dag node to get the bfs predecessors for
///
/// :returns: A list of nodes's data and their children in bfs order. The
///     BFSPredecessors class that is returned is a custom container class that
///     implements the sequence protocol. This can be used as a python list
///     with index based access.
/// :rtype: BFSPredecessors
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn bfs_predecessors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> iterators::BFSPredecessors {
    let index = NodeIndex::new(node);
    let out_list = core_bfs_predecessors(&graph.graph, index)
        .filter_map(|(nx, succ_list)| {
            if succ_list.is_empty() {
                None
            } else {
                Some((
                    graph.graph.node_weight(nx).unwrap().clone_ref(py),
                    succ_list
                        .into_iter()
                        .map(|pred| graph.graph.node_weight(pred).unwrap().clone_ref(py))
                        .collect(),
                ))
            }
        })
        .collect();
    iterators::BFSPredecessors {
        bfs_predecessors: out_list,
    }
}

/// Retrieve all ancestors of a specified node in a directed graph.
///
/// This function differs from the :meth:`PyDiGraph.predecessors` method,
/// which only returns nodes that have a direct edge leading to the specified
/// node. In contrast, this function returns all nodes that have a path
/// leading to the specified node, regardless of the number of edges in
/// between.
///
///     >>> G = rx.PyDiGraph()
///     >>> G.add_nodes_from(range(5))
///     NodeIndices[0, 1, 2, 3, 4]
///     >>> G.add_edges_from_no_data([(0, 2), (1, 2), (2, 3), (3, 4)])
///     [0, 1, 2, 3]
///     >>> rx.ancestors(G, 3)
///     {0, 1, 2}
///
/// .. seealso ::
///   See also :func:`~predecessors`.
///
/// :param PyDiGraph graph: The directed graph from which to retrieve ancestors.
/// :param int node: The index of the node for which to find ancestors.
///
/// :returns: A set containing the indices of all ancestor nodes of the
///          specified node.
/// :rtype: set[int]
///
/// :raises IndexError: If the specified node is not present in the directed graph.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn ancestors(graph: &digraph::PyDiGraph, node: usize) -> PyResult<HashSet<usize>> {
    let index = NodeIndex::new(node);
    if !graph.graph.contains_node(index) {
        return Err(PyIndexError::new_err(format!(
            "Node source index \"{}\" out of graph bound",
            node
        )));
    }
    Ok(core_ancestors(&graph.graph, index)
        .map(|x| x.index())
        .filter(|x| *x != node)
        .collect())
}

/// Retrieve all descendants of a specified node in a directed graph.
///
/// This function differs from the :meth:`PyDiGraph.successors` method,
/// which only returns nodes that have a direct edge leading from the specified
/// node. In contrast, this function returns all nodes that have a path
/// leading from the specified node, regardless of the number of edges in
/// between.
///
///     >>> G = rx.PyDiGraph()
///     >>> G.add_nodes_from(range(5))
///     NodeIndices[0, 1, 2, 3, 4]
///     >>> G.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (2, 4)])
///     [0, 1, 2, 3]
///     >>> rx.descendants(G, 1)
///     {2, 3, 4}
///
/// .. seealso ::
///   See also :func:`~ancestors`.
///
/// :param PyDiGraph graph: The directed graph from which to retrieve descendants.
/// :param int node: The index of the node for which to find descendants.
///
/// :returns: A set containing the indices of all descendant nodes of the
///          specified node.
/// :rtype: set[int]
///
/// :raises IndexError: If the specified node is not present in the directed graph.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn descendants(graph: &digraph::PyDiGraph, node: usize) -> PyResult<HashSet<usize>> {
    let index = NodeIndex::new(node);
    if !graph.graph.contains_node(index) {
        return Err(PyIndexError::new_err(format!(
            "Node source index \"{}\" out of graph bound",
            node
        )));
    }
    Ok(core_descendants(&graph.graph, index)
        .map(|x| x.index())
        .filter(|x| *x != node)
        .collect())
}

/// Breadth-first traversal of a directed graph with several source vertices.
///
/// Pseudo-code for the breadth-first search algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.BFSVisitor` is called.
///
/// ::
///
///   # G - graph, s - single source node
///   BFS(G, s)
///     let color be a mapping             # color[u] - vertex u color WHITE/GRAY/BLACK
///     for each u in G                    # u - vertex in G
///       color[u] := WHITE                # color all vertices as undiscovered
///     end for
///     let Q be a queue
///     ENQUEUE(Q, s)
///     color[s] := GRAY                   # event: discover_vertex(s)
///     while (Q is not empty)
///       u := DEQUEUE(Q)
///       for each v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
///         if (WHITE = color[v])          # event: tree_edge((u, v, w))
///           color[v] := GRAY             # event: discover_vertex(v)
///           ENQUEUE(Q, v)
///         else                           # event: non_tree_edge((u, v, w))
///           if (GRAY = color[v])         # event: gray_target_edge((u, v, w))
///             ...                       
///           elif (BLACK = color[v])      # event: black_target_edge((u, v, w))
///             ...                       
///       end for
///       color[u] := BLACK                # event: finish_vertex(u)
///     end while
///
/// For several source nodes, the BFS algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.BFSVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we keep track of the tree edges:
///
/// .. jupyter-execute::
///
///        import rustworkx as rx
///        from rustworkx.visit import BFSVisitor
///   
///        class TreeEdgesRecorder(BFSVisitor):
///
///            def __init__(self):
///                self.edges = []
///
///            def tree_edge(self, edge):
///                self.edges.append(edge)
///
///        graph = rx.PyDiGraph()
///        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
///        vis = TreeEdgesRecorder()
///        rx.digraph_bfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// Here is another example, using the :class:`~rustworkx.visit.PruneSearch`
/// exception, to find the shortest path between two vertices with some
/// restrictions on the edges
/// (for a more efficient ways to find the shortest path, see :ref:`shortest-paths`):
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visit import BFSVisitor, PruneSearch
///
///
///     graph = rx.PyDiGraph()
///     home, market, school = graph.add_nodes_from(['home', 'market', 'school'])
///     graph.add_edges_from_no_data(
///         [(school, home), (school, market), (market, home)]
///     )
///    
///     class DistanceHomeFinder(BFSVisitor):
///
///         def __init__(self):
///             self.distance = {}
///
///         def discover_vertex(self, vertex):
///             self.distance.setdefault(vertex, 0)
///
///         def tree_edge(self, edge):
///             source, target, _ = edge
///             # the road directly from home to school is closed
///             if {source, target} == {home, school}:
///                 raise PruneSearch
///             self.distance[target] = self.distance[source] + 1
///
///     vis = DistanceHomeFinder()
///     rx.digraph_bfs_search(graph, [school], vis)
///     print('Distance from school to home:', vis.distance[home])
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///
///     An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
///     raised in the :class:`~rustworkx.visit.BFSVisitor.finish_vertex` event.
///
///
/// :param PyDiGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the breadth-first search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, visitor=None))]
pub fn digraph_bfs_search(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: Option<Vec<usize>>,
    visitor: Option<PyBfsVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    breadth_first_search(&graph.graph, starts, |event| {
        bfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Breadth-first traversal of a undirected graph with several source vertices.
///
/// Pseudo-code for the breadth-first search algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.BFSVisitor` is called.
///
/// ::
///
///   # G - graph, s - single source node
///   BFS(G, s)
///     let color be a mapping             # color[u] - vertex u color WHITE/GRAY/BLACK
///     for each u in G                    # u - vertex in G
///       color[u] := WHITE                # color all vertices as undiscovered
///     end for
///     let Q be a queue
///     ENQUEUE(Q, s)
///     color[s] := GRAY                   # event: discover_vertex(s)
///     while (Q is not empty)
///       u := DEQUEUE(Q)
///       for each v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
///         if (WHITE = color[v])          # event: tree_edge((u, v, w))
///           color[v] := GRAY             # event: discover_vertex(v)
///           ENQUEUE(Q, v)
///         else                           # event: non_tree_edge((u, v, w))
///           if (GRAY = color[v])         # event: gray_target_edge((u, v, w))
///             ...                       
///           elif (BLACK = color[v])      # event: black_target_edge((u, v, w))
///             ...                       
///       end for
///       color[u] := BLACK                # event: finish_vertex(u)
///     end while
///
/// For several source nodes, the BFS algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.BFSVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we keep track of the tree edges:
///
/// .. jupyter-execute::
///
///        import rustworkx as rx
///        from rustworkx.visit import BFSVisitor
///   
///        class TreeEdgesRecorder(BFSVisitor):
///
///            def __init__(self):
///                self.edges = []
///
///            def tree_edge(self, edge):
///                self.edges.append(edge)
///
///        graph = rx.PyGraph()
///        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
///        vis = TreeEdgesRecorder()
///        rx.graph_bfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// Here is another example, using the :class:`~rustworkx.visit.PruneSearch`
/// exception, to find the shortest path between two vertices with some
/// restrictions on the edges
/// (for a more efficient ways to find the shortest path, see :ref:`shortest-paths`):
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visit import BFSVisitor, PruneSearch
///
///
///     graph = rx.PyGraph()
///     home, market, school = graph.add_nodes_from(['home', 'market', 'school'])
///     graph.add_edges_from_no_data(
///         [(school, home), (school, market), (market, home)]
///     )
///
///     class DistanceHomeFinder(BFSVisitor):
///
///         def __init__(self):
///             self.distance = {}
///
///         def discover_vertex(self, vertex):
///             self.distance.setdefault(vertex, 0)
///
///         def tree_edge(self, edge):
///             source, target, _ = edge
///             # the road directly from home to school is closed
///             if {source, target} == {home, school}:
///                 raise PruneSearch
///             self.distance[target] = self.distance[source] + 1
///
///     vis = DistanceHomeFinder()
///     rx.graph_bfs_search(graph, [school], vis)
///     print('Distance from school to home:', vis.distance[home])
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///     An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is raised in the
///     :class:`~rustworkx.visit.BFSVisitor.finish_vertex` event.
///
///
/// :param PyGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the breadth-first search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, visitor=None))]
pub fn graph_bfs_search(
    py: Python,
    graph: &graph::PyGraph,
    source: Option<Vec<usize>>,
    visitor: Option<PyBfsVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    breadth_first_search(&graph.graph, starts, |event| {
        bfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Depth-first traversal of a directed graph with several source vertices.
///
/// Pseudo-code for the depth-first search algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.DFSVisitor` is called.
///
/// ::
///
///   # G - graph, s - single source node
///   DFS(G, s)
///     let color be a mapping                        # color[u] - vertex u color WHITE/GRAY/BLACK
///     for each u in G                               # u - vertex in G
///       color[u] := WHITE                           # color all as undiscovered
///     end for
///     time := 0
///     let S be a stack
///     PUSH(S, (s, iterator of OutEdges(G, s)))      # S - stack of vertices and edge iterators
///     color[s] := GRAY                              # event: discover_vertex(s, time)
///     while (S is not empty)
///       let (u, iterator) := LAST(S)
///       flag := False                               # whether edge to undiscovered vertex found
///       for each v, w in iterator                   # v - target vertex, w - edge weight
///         if (WHITE = color[v])                     # event: tree_edge((u, v, w))
///           time := time + 1
///           color[v] := GRAY                        # event: discover_vertex(v, time)
///           flag := True
///           break
///         elif (GRAY = color[v])                    # event: back_edge((u, v, w))
///           ...
///         elif (BLACK = color[v])                   # event: forward_or_cross_edge((u, v, w))
///           ...
///       end for
///       if (flag is True)
///         PUSH(S, (v, iterator of OutEdges(G, v)))
///       elif (flag is False)
///         time := time + 1
///         color[u] := BLACK                         # event: finish_vertex(u, time)
///         POP(S)
///     end while
///
/// For several source nodes, the DFS algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.DFSVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we keep track of the tree edges:
///
/// .. jupyter-execute::
///
///        import rustworkx as rx
///        from rustworkx.visit import DFSVisitor
///   
///        class TreeEdgesRecorder(DFSVisitor):
///
///            def __init__(self):
///                self.edges = []
///
///            def tree_edge(self, edge):
///                self.edges.append(edge)
///
///        graph = rx.PyDiGraph()
///        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
///        vis = TreeEdgesRecorder()
///        rx.digraph_dfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can *not* be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///     An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
///     raised in the :class:`~rustworkx.visit.DFSVisitor.finish_vertex` event.
///
/// :param PyDiGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the depth-first search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, visitor=None))]
pub fn digraph_dfs_search(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: Option<Vec<usize>>,
    visitor: Option<PyDfsVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    depth_first_search(&graph.graph, starts, |event| {
        dfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Depth-first traversal of an undirected graph with several source vertices.
///
/// Pseudo-code for the depth-first search algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.DFSVisitor` is called.
///
/// ::
///
///   # G - graph, s - single source node
///   DFS(G, s)
///     let color be a mapping                        # color[u] - vertex u color WHITE/GRAY/BLACK
///     for each u in G                               # u - vertex in G
///       color[u] := WHITE                           # color all as undiscovered
///     end for
///     time := 0
///     let S be a stack
///     PUSH(S, (s, iterator of OutEdges(G, s)))      # S - stack of vertices and edge iterators
///     color[s] := GRAY                              # event: discover_vertex(s, time)
///     while (S is not empty)
///       let (u, iterator) := LAST(S)
///       flag := False                               # whether edge to undiscovered vertex found
///       for each v, w in iterator                   # v - target vertex, w - edge weight
///         if (WHITE = color[v])                     # event: tree_edge((u, v, w))
///           time := time + 1
///           color[v] := GRAY                        # event: discover_vertex(v, time)
///           flag := True
///           break
///         elif (GRAY = color[v])                    # event: back_edge((u, v, w))
///           ...
///         elif (BLACK = color[v])                   # event: forward_or_cross_edge((u, v, w))
///           ...
///       end for
///       if (flag is True)
///         PUSH(S, (v, iterator of OutEdges(G, v)))
///       elif (flag is False)
///         time := time + 1
///         color[u] := BLACK                         # event: finish_vertex(u, time)
///         POP(S)
///     end while
///
/// For several source nodes, the DFS algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.DFSVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we keep track of the tree edges:
///
/// .. jupyter-execute::
///
///        import rustworkx as rx
///        from rustworkx.visit import DFSVisitor
///   
///        class TreeEdgesRecorder(DFSVisitor):
///
///            def __init__(self):
///                self.edges = []
///
///            def tree_edge(self, edge):
///                self.edges.append(edge)
///
///        graph = rx.PyGraph()
///        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
///        vis = TreeEdgesRecorder()
///        rx.graph_dfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can *not* be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///     An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
///     raised in the :class:`~rustworkx.visit.DFSVisitor.finish_vertex` event.
///
/// :param PyGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the depth-first search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, visitor=None))]
pub fn graph_dfs_search(
    py: Python,
    graph: &graph::PyGraph,
    source: Option<Vec<usize>>,
    visitor: Option<PyDfsVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    depth_first_search(&graph.graph, starts, |event| {
        dfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Dijkstra traversal of a directed graph with several source vertices.
///
/// Pseudo-code for the Dijkstra algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.DijkstraVisitor` is called.
///
/// ::
///
///     # G - graph, s - single source node, weight - edge cost function
///     DIJKSTRA(G, s, weight)
///       let score be empty mapping
///       let visited be empty set
///       let Q be a priority queue
///       score[s] := 0.0
///       PUSH(Q, (score[s], s))                # only score determines the priority
///       while Q is not empty
///         cost, u := POP-MIN(Q)
///         if u in visited
///           continue
///         PUT(visited, u)                     # event: discover_vertex(u, cost)
///         for each _, v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
///           ...                               # event: examine_edge((u, v, w))
///           if v in visited
///             continue
///           next_cost = cost + weight(w)
///           if {(v is key in score)
///               and (score[v] <= next_cost)}  # event: edge_not_relaxed((u, v, w))
///             ...
///           else:                             # v not scored or scored higher
///             score[v] = next_cost            # event: edge_relaxed((u, v, w))
///             PUSH(Q, (next_cost, v))
///         end for                             # event: finish_vertex(u)
///       end while
///
/// For several source nodes, the Dijkstra algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.DijkstraVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we find the shortest path from vertex 0 to 5, and exit the visit as
/// soon as we reach the goal vertex:
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visit import DijkstraVisitor, StopSearch
///
///     graph = rx.PyDiGraph()
///     graph.extend_from_edge_list([
///         (0, 1), (0, 2), (0, 3), (0, 4),
///         (1, 3),
///         (2, 3), (2, 4),
///         (4, 5),
///     ])
///
///     class PathFinder(DijkstraVisitor):
///
///         def __init__(self, start, goal):
///             self.start = start
///             self.goal = goal
///             self.predecessors = {}
///
///         def get_path(self):
///             n = self.goal
///             rev_path = [n]
///             while n != self.start:
///                 n = self.predecessors[n]
///                 rev_path.append(n)
///             return list(reversed(rev_path))
///
///         def discover_vertex(self, vertex, cost):
///             if vertex == self.goal:
///                 raise StopSearch
///
///         def edge_relaxed(self, edge):
///             self.predecessors[edge[1]] = edge[0]
///
///     start = 0
///     vis = PathFinder(start=start, goal=5)
///     rx.digraph_dijkstra_search(graph, [start], weight_fn=None, visitor=vis)
///     print('Path:', vis.get_path())
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///
///    An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
///    raised in the :class:`~rustworkx.visit.DijkstraVisitor.finish_vertex` event.
///
/// :param PyDiGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the dijkstra search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge. If not specified,
///     a default value of cost ``1.0`` will be used for each edge.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DijkstraVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, weight_fn=None, visitor=None))]
pub fn digraph_dijkstra_search(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: Option<Vec<usize>>,
    weight_fn: Option<PyObject>,
    visitor: Option<PyDijkstraVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    let edge_cost_fn = CostFn::try_from((weight_fn, 1.0))?;
    dijkstra_search(
        &graph.graph,
        starts,
        |e| edge_cost_fn.call(py, e.weight()),
        |event| dijkstra_handler(py, &visitor, event),
    )??;

    Ok(())
}

/// Dijkstra traversal of an undirected graph with several source vertices.
///
/// Pseudo-code for the Dijkstra algorithm with a single source vertex is
/// listed below with annotated event points at which a method of the given
/// :class:`~rustworkx.visit.DijkstraVisitor` is called.
///
/// ::
///
///     # G - graph, s - single source node, weight - edge cost function
///     DIJKSTRA(G, s, weight)
///       let score be empty mapping
///       let visited be empty set
///       let Q be a priority queue
///       score[s] := 0.0
///       PUSH(Q, (score[s], s))                # only score determines the priority
///       while Q is not empty
///         cost, u := POP-MIN(Q)
///         if u in visited
///           continue
///         PUT(visited, u)                     # event: discover_vertex(u, cost)
///         for each _, v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
///           ...                               # event: examine_edge((u, v, w))
///           if v in visited
///             continue
///           next_cost = cost + weight(w)
///           if {(v is key in score)
///               and (score[v] <= next_cost)}  # event: edge_not_relaxed((u, v, w))
///             ...
///           else:                             # v not scored or scored higher
///             score[v] = next_cost            # event: edge_relaxed((u, v, w))
///             PUSH(Q, (next_cost, v))
///         end for                             # event: finish_vertex(u)
///       end while
///
/// For several source nodes, the Dijkstra algorithm is applied on source nodes by the given order.
///
/// If an exception is raised inside the callback method of the
/// :class:`~rustworkx.visit.DijkstraVisitor` instance, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// In the following example we find the shortest path from vertex 0 to 5, and exit the visit as
/// soon as we reach the goal vertex:
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visit import DijkstraVisitor, StopSearch
///
///     graph = rx.PyGraph()
///     graph.extend_from_edge_list([
///         (0, 1), (0, 2), (0, 3), (0, 4),
///         (1, 3),
///         (2, 3), (2, 4),
///         (4, 5),
///     ])
///
///     class PathFinder(DijkstraVisitor):
///
///         def __init__(self, start, goal):
///             self.start = start
///             self.goal = goal
///             self.predecessors = {}
///
///         def get_path(self):
///             n = self.goal
///             rev_path = [n]
///             while n != self.start:
///                 n = self.predecessors[n]
///                 rev_path.append(n)
///             return list(reversed(rev_path))
///
///         def discover_vertex(self, vertex, cost):
///             if vertex == self.goal:
///                 raise StopSearch
///
///         def edge_relaxed(self, edge):
///             self.predecessors[edge[1]] = edge[0]
///
///     start = 0
///     vis = PathFinder(start=start, goal=5)
///     rx.graph_dijkstra_search(graph, [start], weight_fn=None, visitor=vis)
///     print('Path:', vis.get_path())
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///     Trying to do so raises an exception.
///
///
/// .. note::
///
///    An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
///    raised in the :class:`~rustworkx.visit.DijkstraVisitor.finish_vertex` event.
///
/// :param PyGraph graph: The graph to be used.
/// :param source: An optional list of node indices to use as the starting nodes
///     for the dijkstra search. If ``None`` or not specified then a source
///     will be chosen arbitrarily and repeated until all components of the
///     graph are searched.
///     This can be a ``Sequence[int]`` or ``None``.
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge. If not specified,
///     a default value of cost ``1.0`` will be used for each edge.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DijkstraVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
#[pyo3(signature = (graph, source=None, weight_fn=None, visitor=None))]
pub fn graph_dijkstra_search(
    py: Python,
    graph: &graph::PyGraph,
    source: Option<Vec<usize>>,
    weight_fn: Option<PyObject>,
    visitor: Option<PyDijkstraVisitor>,
) -> PyResult<()> {
    if visitor.is_none() {
        return Err(PyTypeError::new_err("Missing required argument visitor"));
    }
    let visitor = visitor.unwrap();
    let starts: Vec<_> = match source {
        Some(nx) => nx.into_iter().map(NodeIndex::new).collect(),
        None => graph.graph.node_indices().collect(),
    };

    validate_source_nodes(&graph.graph, &starts)?;

    let edge_cost_fn = CostFn::try_from((weight_fn, 1.0))?;
    dijkstra_search(
        &graph.graph,
        starts,
        |e| edge_cost_fn.call(py, e.weight()),
        |event| dijkstra_handler(py, &visitor, event),
    )??;

    Ok(())
}
