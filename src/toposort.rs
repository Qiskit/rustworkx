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
use pyo3::Python;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::IntoNodeIdentifiers;

use crate::dag_algo::is_directed_acyclic_graph;
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
#[pyclass(module = "rustworkx")]
#[pyo3(text_signature = "(graph, /, check_cycle=True)")]
pub struct TopologicalSorter {
    dag: Py<PyDiGraph>,
    ready_nodes: Vec<NodeIndex>,
    predecessor_count: HashMap<NodeIndex, usize>,
    node2state: HashMap<NodeIndex, NodeState>,
    num_passed_out: usize,
    num_finished: usize,
}

#[pymethods]
impl TopologicalSorter {
    #[new]
    #[args(check_cycle = "true")]
    fn new(py: Python, dag: Py<PyDiGraph>, check_cycle: bool) -> PyResult<Self> {
        {
            let dag = &dag.borrow(py);
            if !dag.check_cycle && check_cycle && !is_directed_acyclic_graph(dag) {
                return Err(DAGHasCycle::new_err("PyDiGraph object has a cycle"));
            }
        }

        let ready_nodes = {
            let dag = &dag.borrow(py);

            dag.graph
                .node_identifiers()
                .filter(|node| {
                    dag.graph
                        .neighbors_directed(*node, petgraph::Direction::Incoming)
                        .next()
                        .is_none()
                })
                .collect()
        };

        Ok(TopologicalSorter {
            dag,
            ready_nodes,
            predecessor_count: HashMap::new(),
            node2state: HashMap::new(),
            num_passed_out: 0,
            num_finished: 0,
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
    fn get_ready(&mut self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.ready_nodes.len());
        for nx in &self.ready_nodes {
            out.push(nx.index());
            self.node2state.insert(*nx, NodeState::Ready);
        }

        self.ready_nodes.clear();
        self.num_passed_out += out.len();
        out
    }

    /// Marks a set of nodes returned by "get_ready" as processed.
    ///
    /// This method unblocks any successor of each node in *nodes* for being returned
    /// in the future by a call to "get_ready".
    ///
    /// :param list nodes: A list of node indices to marks as done.
    ///
    /// :raises `ValueError`: If any node in *nodes* has already been marked as
    ///     processed by a previous call to this method or node has not yet been returned
    ///     by "get_ready".
    fn done(&mut self, py: Python, nodes: Vec<usize>) -> PyResult<()> {
        let dag = &self.dag.borrow(py);
        for node in nodes {
            let node = NodeIndex::new(node);
            match self.node2state.get_mut(&node) {
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

            for succ in dag
                .graph
                .neighbors_directed(node, petgraph::Direction::Outgoing)
            {
                match self.predecessor_count.entry(succ) {
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() -= 1;
                        if *entry.get() == 0 {
                            self.ready_nodes.push(succ);
                            entry.remove_entry();
                        }
                    }
                    Entry::Vacant(entry) => {
                        let in_degree = dag
                            .graph
                            .neighbors_directed(succ, petgraph::Direction::Incoming)
                            .count()
                            - 1;

                        if in_degree == 0 {
                            self.ready_nodes.push(succ);
                        } else {
                            entry.insert(in_degree);
                        }
                    }
                }
            }

            self.num_finished += 1;
        }

        Ok(())
    }
}
