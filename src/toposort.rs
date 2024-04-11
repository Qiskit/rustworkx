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

use crate::digraph::PyDiGraph;

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::IntoNodeIdentifiers;

use crate::dag_algo::{is_directed_acyclic_graph, traversal_directions};
use crate::DAGHasCycle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    Ready,
    Done,
}

/// Provides functionality to topologically sort a directed graph.
///
/// The steps required to perform the sorting of a given graph are as follows:
///
/// 1. Create an instance of the TopologicalSorter with an initial graph.
/// 2. While `is_active()` is True, iterate over the nodes returned by `get_ready()` and process them.
/// 3. Call `done()` on each node as it finishes processing.
///
/// For example:
///
/// .. jupyter-execute::
///
///   import rustworkx as rx
///
///   graph = rx.generators.directed_path_graph(5)
///   sorter = rx.TopologicalSorter(graph)
///   while sorter.is_active():
///       nodes = sorter.get_ready()
///       print(nodes)
///       sorter.done(nodes)
///
/// The underlying graph can be mutated and `TopologicalSorter` will pick-up the modifications
/// but it's not recommended doing it as it may result in a logical-error.
///
/// :param PyDiGraph graph: The directed graph to be used.
/// :param bool check_cycle: When this is set to ``True``, we search
///     for cycles in the graph during initialization of topological sorter
///     and raise :class:`~rustworkx.DAGHasCycle` if any cycle is detected. If
///     it's set to ``False``, topological sorter will output as many nodes
///     as possible until cycles block more progress. By default is ``True``.
/// :param bool reverse: If ``False`` (the default), perform a regular topological ordering.  If
///     ``True``, the ordering will be a reversed topological ordering; that is, a topological
///     order if all the edges had their directions flipped, such that the first nodes returned are
///     the ones that have only incoming edges in the DAG.
/// :param Iterable[int] initial: If given, the initial node indices to start the topological
///     ordering from.  If not given, the topological ordering will certainly contain every node in
///     the graph.  If given, only the ``initial`` nodes and nodes that are dominated by the
///     ``initial`` set will be in the ordering.  Notably, the first return from :meth:`get_ready`
///     will be the same set of values as ``initial``, and any node that has a natural in
///     degree of zero will not be in the output ordering if ``initial`` is given and the
///     zero-in-degree node is not in it.
///
///     It is a :exc:`ValueError` to give an `initial` set where the nodes have even a partial
///     topological order between themselves, though this might not appear until some call
///     to :meth:`done`.
/// :param bool check_args: If ``True`` (the default), then all arguments to :meth:`done` are
///     checked for validity, and a :exc:`ValueError` is raised if any were not ready, already
///     done, or not indices of the circuit.  If ``False``, the tracking for this is disabled,
///     which can provide a meaningful performance and memory improvement, but the results will
///     be undefined if invalid values are given.
#[pyclass(module = "rustworkx")]
pub struct TopologicalSorter {
    dag: Py<PyDiGraph>,
    ready_nodes: Vec<NodeIndex>,
    predecessor_count: HashMap<NodeIndex, usize>,
    node2state: Option<HashMap<NodeIndex, NodeState>>,
    num_passed_out: usize,
    num_finished: usize,
    in_dir: petgraph::Direction,
    out_dir: petgraph::Direction,
}

#[pymethods]
impl TopologicalSorter {
    #[new]
    #[pyo3(signature=(dag, /, check_cycle=true, *, reverse=false, initial=None, check_args=true))]
    fn new(
        py: Python,
        dag: Py<PyDiGraph>,
        check_cycle: bool,
        reverse: bool,
        initial: Option<&Bound<PyAny>>,
        check_args: bool,
    ) -> PyResult<Self> {
        {
            let dag = &dag.borrow(py);
            if !dag.check_cycle && check_cycle && !is_directed_acyclic_graph(dag) {
                return Err(DAGHasCycle::new_err("PyDiGraph object has a cycle"));
            }
        }

        let (in_dir, out_dir) = traversal_directions(reverse);
        let mut predecessor_count = HashMap::new();
        let ready_nodes = if let Some(initial) = initial {
            let dag = &dag.borrow(py);
            initial
                .iter()?
                .map(|maybe_index| {
                    let node = NodeIndex::new(maybe_index?.extract::<usize>()?);
                    // If we're using an initial set, it's possible that the user gave us an
                    // initial list with topological ordering defined between the nodes.  With this
                    // online sorter we detect that with a lag (it'll happen in a later call to
                    // `done`), but we'll see it as an attempt to reduce a predecessor count below
                    // this initial zero.
                    predecessor_count.insert(node, 0);
                    dag.graph
                        .contains_node(node)
                        .then_some(node)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "node index {} is not in this graph",
                                node.index()
                            ))
                        })
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            let dag = &dag.borrow(py);
            dag.graph
                .node_identifiers()
                .filter(|node| dag.graph.neighbors_directed(*node, in_dir).next().is_none())
                .collect()
        };

        Ok(TopologicalSorter {
            dag,
            ready_nodes,
            predecessor_count,
            node2state: check_args.then(HashMap::new),
            num_passed_out: 0,
            num_finished: 0,
            in_dir,
            out_dir,
        })
    }

    /// Return ``True`` if more progress can be made and ``False`` otherwise.
    ///
    /// Progress can be made if either there are still nodes ready that haven't yet
    /// been returned by "get_ready" or the number of nodes marked "done" is less than the
    /// number that have been returned by "get_ready".
    fn is_active(&self) -> bool {
        self.num_finished < self.num_passed_out || !self.ready_nodes.is_empty()
    }

    /// Return a list of all the nodes that are ready.
    ///
    /// Initially it returns all nodes with no predecessors; once those are marked
    /// as processed by calling "done", further calls will return all new nodes that
    /// have all their predecessors already processed. Once no more progress can be made,
    /// empty lists are returned.
    ///
    /// :returns: A list of node indices of all the ready nodes.
    /// :rtype: List
    fn get_ready<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyList> {
        self.num_passed_out += self.ready_nodes.len();
        if let Some(node2state) = self.node2state.as_mut() {
            PyList::new_bound(
                py,
                self.ready_nodes.drain(..).map(|nx| {
                    node2state.insert(nx, NodeState::Ready);
                    nx.index()
                }),
            )
        } else {
            PyList::new_bound(py, self.ready_nodes.drain(..).map(|nx| nx.index()))
        }
    }

    /// Marks a set of nodes returned by "get_ready" as processed.
    ///
    /// This method unblocks any successor of each node in *nodes* for being returned
    /// in the future by a call to "get_ready".
    ///
    /// :param nodes: A node index or list of node indices to mark as done.
    /// :type nodes: int | list[int]
    ///
    /// :raises `ValueError`: If any node in *nodes* has already been marked as
    ///     processed by a previous call to this method or node has not yet been returned
    ///     by "get_ready".
    /// :raises `ValueError`: If one of the given ``initial`` nodes is a direct successor of one
    ///     of the nodes given to :meth:`done`.  This can only happen if the ``initial`` nodes had
    ///     even a partial topological ordering amongst themselves, which is not a valid
    ///     starting input.
    fn done(&mut self, nodes: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(node) = nodes.extract::<usize>() {
            self.done_single(nodes.py(), NodeIndex::new(node))
        } else if let Ok(nodes) = nodes.downcast::<PyList>() {
            for node in nodes {
                self.done_single(nodes.py(), NodeIndex::new(node.extract()?))?
            }
            Ok(())
        } else {
            for node in nodes.iter()? {
                self.done_single(nodes.py(), NodeIndex::new(node?.extract()?))?
            }
            Ok(())
        }
    }
}

impl TopologicalSorter {
    #[inline(always)]
    fn done_single(&mut self, py: Python, node: NodeIndex) -> PyResult<()> {
        let dag = self.dag.borrow(py);
        if let Some(node2state) = self.node2state.as_mut() {
            match node2state.get_mut(&node) {
                None => {
                    return Err(PyValueError::new_err(format!(
                        "node {} was not passed out (still not ready).",
                        node.index()
                    )));
                }
                Some(NodeState::Done) => {
                    return Err(PyValueError::new_err(format!(
                        "node {} was already marked done.",
                        node.index()
                    )));
                }
                Some(state) => {
                    debug_assert_eq!(*state, NodeState::Ready);
                    *state = NodeState::Done;
                }
            }
        }

        for succ in dag.graph.neighbors_directed(node, self.out_dir) {
            match self.predecessor_count.entry(succ) {
                Entry::Occupied(mut entry) => {
                    let in_degree = entry.get_mut();
                    if *in_degree == 0 {
                        return Err(PyValueError::new_err(
                            "at least one initial node is reachable from another",
                        ));
                    } else if *in_degree == 1 {
                        self.ready_nodes.push(succ);
                        entry.remove_entry();
                    } else {
                        *in_degree -= 1;
                    }
                }
                Entry::Vacant(entry) => {
                    let in_degree = dag.graph.neighbors_directed(succ, self.in_dir).count() - 1;

                    if in_degree == 0 {
                        self.ready_nodes.push(succ);
                    } else {
                        entry.insert(in_degree);
                    }
                }
            }
        }
        self.num_finished += 1;
        Ok(())
    }
}
