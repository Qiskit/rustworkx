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

use super::{digraph, export_rustworkx_functions, graph, iterators, CostFn};

use std::convert::TryFrom;

use hashbrown::HashSet;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;

use crate::iterators::EdgeList;

export_rustworkx_functions!(
    digraph_dfs_edges,
    graph_dfs_edges,
    digraph_bfs_search,
    graph_bfs_search,
    digraph_dfs_search,
    graph_dfs_search,
    digraph_dijkstra_search,
    graph_dijkstra_search,
    bfs_successors,
    bfs_predecessors,
    descendants,
    ancestors
);

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
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
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
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
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

/// Return the ancestors of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.predecessors` method  in that
/// ``predecessors`` returns only nodes with a direct edge into the provided
/// node. While this function returns all nodes that have a path into the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the ancestors from.
/// :param int node: The index of the graph node to get the ancestors for
///
/// :returns: A set of node indices of ancestors of provided node.
/// :rtype: set
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn ancestors(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    core_ancestors(&graph.graph, NodeIndex::new(node))
        .map(|x| x.index())
        .filter(|x| *x != node)
        .collect()
}

/// Return the descendants of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.successors` method in that
/// ``successors``` returns only nodes with a direct edge out of the provided
/// node. While this function returns all nodes that have a path from the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the descendants from
/// :param int node: The index of the graph node to get the descendants for
///
/// :returns: A set of node indices of descendants of provided node.
/// :rtype: set
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn descendants(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    let index = NodeIndex::new(node);
    core_descendants(&graph.graph, index)
        .map(|x| x.index())
        .filter(|x| *x != node)
        .collect()
}

/// Breadth-first traversal of a directed graph.
///
/// The pseudo-code for the BFS algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     BFS(G, s)
///       for each vertex u in V
///           color[u] := WHITE
///       end for
///       color[s] := GRAY
///       EQUEUE(Q, s)                             discover vertex s
///       while (Q != Ø)
///           u := DEQUEUE(Q)
///           for each vertex v in Adj[u]          (u,v) is a tree edge
///               if (color[v] = WHITE)
///                   color[v] = GRAY
///               else                             (u,v) is a non - tree edge
///                   if (color[v] = GRAY)         (u,v) has a gray target
///                       ...
///                   else if (color[v] = BLACK)   (u,v) has a black target
///                       ...
///           end for
///           color[u] := BLACK                    finish vertex u
///       end while
///
/// If an exception is raised inside the callback function, the graph traversal
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
///        rx.bfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///
/// :param PyDiGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the breadth-first search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
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

    breadth_first_search(&graph.graph, starts, |event| {
        bfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Breadth-first traversal of an undirected graph.
///
/// The pseudo-code for the BFS algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     BFS(G, s)
///       for each vertex u in V
///           color[u] := WHITE
///       end for
///       color[s] := GRAY
///       EQUEUE(Q, s)                             discover vertex s
///       while (Q != Ø)
///           u := DEQUEUE(Q)
///           for each vertex v in Adj[u]          (u,v) is a tree edge
///               if (color[v] = WHITE)
///                   color[v] = GRAY
///               else                             (u,v) is a non - tree edge
///                   if (color[v] = GRAY)         (u,v) has a gray target
///                       ...
///                   else if (color[v] = BLACK)   (u,v) has a black target
///                       ...
///           end for
///           color[u] := BLACK                    finish vertex u
///       end while
///
/// If an exception is raised inside the callback function, the graph traversal
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
///        rx.bfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///
/// :param PyGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the breadth-first search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
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

    breadth_first_search(&graph.graph, starts, |event| {
        bfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Depth-first traversal of a directed graph.
///
/// The pseudo-code for the DFS algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     DFS(G)
///       for each vertex u in V
///           color[u] := WHITE                 initialize vertex u
///       end for
///       time := 0
///       call DFS-VISIT(G, source)             start vertex s
///
///     DFS-VISIT(G, u)
///       color[u] := GRAY                      discover vertex u
///       for each v in Adj[u]                  examine edge (u,v)
///           if (color[v] = WHITE)             (u,v) is a tree edge
///               all DFS-VISIT(G, v)
///           else if (color[v] = GRAY)         (u,v) is a back edge
///           ...
///           else if (color[v] = BLACK)        (u,v) is a cross or forward edge
///           ...
///       end for
///       color[u] := BLACK                     finish vertex u
///
/// If an exception is raised inside the callback function, the graph traversal
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
///        rx.dfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can *not* be mutated while traversing.
///
/// :param PyDiGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the depth-first search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
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

    depth_first_search(&graph.graph, starts, |event| {
        dfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Depth-first traversal of an undirected graph.
///
/// The pseudo-code for the DFS algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     DFS(G)
///       for each vertex u in V
///           color[u] := WHITE                 initialize vertex u
///       end for
///       time := 0
///       call DFS-VISIT(G, source)             start vertex s
///
///     DFS-VISIT(G, u)
///       color[u] := GRAY                      discover vertex u
///       for each v in Adj[u]                  examine edge (u,v)
///           if (color[v] = WHITE)             (u,v) is a tree edge
///               all DFS-VISIT(G, v)
///           else if (color[v] = GRAY)         (u,v) is a back edge
///           ...
///           else if (color[v] = BLACK)        (u,v) is a cross or forward edge
///           ...
///       end for
///       color[u] := BLACK                     finish vertex u
///
/// If an exception is raised inside the callback function, the graph traversal
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
///        rx.dfs_search(graph, [0], vis)
///        print('Tree edges:', vis.edges)
///
/// .. note::
///
///     Graph can *not* be mutated while traversing.
///
/// :param PyGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the depth-first search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
/// :param visitor: A visitor object that is invoked at the event points inside the
///     algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
///     This has a default value of ``None`` as a backwards compatibility artifact (to
///     preserve argument ordering from an earlier version) but it is a required argument
///     and will raise a ``TypeError`` if not specified.
#[pyfunction]
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

    depth_first_search(&graph.graph, starts, |event| {
        dfs_handler(py, &visitor, event)
    })?;

    Ok(())
}

/// Dijkstra traversal of a directed graph.
///
/// The pseudo-code for the Dijkstra algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     DIJKSTRA(G, source, weight)
///       for each vertex u in V
///           d[u] := infinity
///           p[u] := u
///       end for
///       d[source] := 0
///       INSERT(Q, source)
///       while (Q != Ø)
///           u := EXTRACT-MIN(Q)                         discover vertex u
///           for each vertex v in Adj[u]                 examine edge (u,v)
///               if (weight[(u,v)] + d[u] < d[v])        edge (u,v) relaxed
///                   d[v] := weight[(u,v)] + d[u]
///                   p[v] := u
///                   DECREASE-KEY(Q, v)
///               else                                    edge (u,v) not relaxed
///                   ...
///               if (d[v] was originally infinity)
///                   INSERT(Q, v)
///           end for                                     finish vertex u
///       end while
///
/// If an exception is raised inside the callback function, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///
/// :param PyDiGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the dijkstra search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
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

    let edge_cost_fn = CostFn::try_from((weight_fn, 1.0))?;
    dijkstra_search(
        &graph.graph,
        starts,
        |e| edge_cost_fn.call(py, e.weight()),
        |event| dijkstra_handler(py, &visitor, event),
    )??;

    Ok(())
}

/// Dijkstra traversal of an undirected graph.
///
/// The pseudo-code for the Dijkstra algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ::
///
///     DIJKSTRA(G, source, weight)
///       for each vertex u in V
///           d[u] := infinity
///           p[u] := u
///       end for
///       d[source] := 0
///       INSERT(Q, source)
///       while (Q != Ø)
///           u := EXTRACT-MIN(Q)                         discover vertex u
///           for each vertex v in Adj[u]                 examine edge (u,v)
///               if (weight[(u,v)] + d[u] < d[v])        edge (u,v) relaxed
///                   d[v] := weight[(u,v)] + d[u]
///                   p[v] := u
///                   DECREASE-KEY(Q, v)
///               else                                    edge (u,v) not relaxed
///                   ...
///               if (d[v] was originally infinity)
///                   INSERT(Q, v)
///           end for                                     finish vertex u
///       end while
///
/// If an exception is raised inside the callback function, the graph traversal
/// will be stopped immediately. You can exploit this to exit early by raising a
/// :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
/// will return but without raising back the exception. You can also prune part of the
/// search tree by raising :class:`~rustworkx.visit.PruneSearch`.
///
/// .. note::
///
///     Graph can **not** be mutated while traversing.
///
/// :param PyGraph graph: The graph to be used.
/// :param List[int] source: An optional list of node indices to use as the starting nodes
///     for the dijkstra search. If this is not specified then a source
///     will be chosen arbitrarly and repeated until all components of the
///     graph are searched.
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

    let edge_cost_fn = CostFn::try_from((weight_fn, 1.0))?;
    dijkstra_search(
        &graph.graph,
        starts,
        |e| edge_cost_fn.call(py, e.weight()),
        |event| dijkstra_handler(py, &visitor, event),
    )??;

    Ok(())
}
