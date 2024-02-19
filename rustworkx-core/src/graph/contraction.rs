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

//! This module defines graph traits for node contraction.

use crate::dictmap::{DictMap, InitWithHasher};
use crate::err::{ContractError, ContractSimpleError};
use crate::graph::NodeRemovable;
use indexmap::map::Entry::{Occupied, Vacant};
use indexmap::IndexSet;
use petgraph::data::Build;
use petgraph::visit::{Data, Dfs, EdgeRef, GraphBase, GraphProp, IntoEdgesDirected, Visitable};
use petgraph::{Directed, Direction, Undirected};
use std::convert::Infallible;
use std::error::Error;
use std::hash::Hash;
use std::ops::Deref;

/// A trait to provide merge functions with an associated
/// [Error] type, automatically.
///
/// This is mostly just a convenience to clean up the contract
/// methods (saves a generic parameter on each).
pub trait MergeFn<W>: FnMut(&W, &W) -> Result<W, Self::Error> {
    type Error: Error;
}

impl<F, W, E: Error> MergeFn<W> for F
where
    F: FnMut(&W, &W) -> Result<W, E>,
{
    type Error = E;
}

pub trait ContractNodesDirected: Data {
    /// The error type returned by contraction.
    type Error;

    /// Substitute a set of nodes with a single new node.
    ///
    /// The specified `nodes` are removed and replaced with a new node
    /// with the given `weight`. Any nodes not in the graph are ignored.
    /// It is valid for `nodes` to be empty, in which case the new node
    /// is added to the graph without edges.
    ///
    /// The contraction may result in multiple edges between nodes if
    /// the underlying graph is a multi-graph. If this is not desired,
    /// use [ContractNodesSimpleDirected::contract_nodes_simple].
    ///
    /// If `check_cycle` is enabled and the contraction would introduce
    /// a cycle, an error is returned and the graph is not modified.
    ///
    /// The `NodeId` of the newly created node is returned.
    ///
    /// # Example
    /// ```
    /// use std::convert::Infallible;
    /// use petgraph::prelude::*;
    /// use rustworkx_core::graph::directed::*;
    ///
    /// // Performs the following transformation:
    /// //      ┌─┐
    /// //      │a│
    /// //      └┬┘              ┌─┐
    /// //       0               │a│
    /// //      ┌▼┐              └┬┘
    /// //      │b│               0
    /// //      └┬┘              ┌▼┐
    /// //       1      ───►     │m│
    /// //      ┌▼┐              └┬┘
    /// //      │c│               2
    /// //      └┬┘              ┌▼┐
    /// //       2               │d│
    /// //      ┌▼┐              └─┘
    /// //      │d│
    /// //      └─┘
    /// let mut dag: StableDiGraph<char, usize> = StableDiGraph::default();
    /// let a = dag.add_node('a');
    /// let b = dag.add_node('b');
    /// let c = dag.add_node('c');
    /// let d = dag.add_node('d');
    /// dag.add_edge(a.clone(), b.clone(), 0);
    /// dag.add_edge(b.clone(), c.clone(), 1);
    /// dag.add_edge(c.clone(), d.clone(), 2);
    ///
    /// let m = dag.contract_nodes([b, c], 'm', true).unwrap();
    /// assert_eq!(dag.edge_weight(dag.find_edge(a.clone(), m.clone()).unwrap()).unwrap(), &0);
    /// assert_eq!(dag.edge_weight(dag.find_edge(m.clone(), d.clone()).unwrap()).unwrap(), &2);
    /// ```
    fn contract_nodes<I>(
        &mut self,
        nodes: I,
        weight: Self::NodeWeight,
        check_cycle: bool,
    ) -> Result<Self::NodeId, Self::Error>
    where
        I: IntoIterator<Item = Self::NodeId>;
}

pub trait ContractNodesSimpleDirected: Data {
    /// The error type returned by contraction.
    type Error<E: Error>;

    /// Substitute a set of nodes with a single new node.
    ///
    /// The specified `nodes` are removed and replaced with a new node
    /// with the given `weight`. Any nodes not in the graph are ignored.
    /// It is valid for `nodes` to be empty, in which case the new node
    /// is added to the graph without edges.
    ///
    /// The specified function `weight_combo_fn` is used to merge
    /// would-be parallel edges during contraction; this function
    /// preserves simple graphs.
    ///
    /// If `check_cycle` is enabled and the contraction would introduce
    /// a cycle, an error is returned and the graph is not modified.
    ///
    /// The `NodeId` of the newly created node is returned.
    ///
    /// # Example
    /// ```
    /// use std::convert::Infallible;
    /// use petgraph::prelude::*;
    /// use rustworkx_core::graph::directed::*;
    ///
    /// // Performs the following transformation:
    /// //                          ┌─┐
    /// //     ┌─┐                  │a│
    /// //  ┌0─┤a├─1┐               └┬┘
    /// //  │  └─┘  │                1
    /// // ┌▼┐     ┌▼┐              ┌▼┐
    /// // │b│     │c│     ───►     │m│
    /// // └┬┘     └┬┘              └┬┘
    /// //  │  ┌─┐  │                3
    /// //  └2►│d│◄3┘               ┌▼┐
    /// //     └─┘                  │d│
    /// //                          └─┘
    /// let mut dag: StableDiGraph<char, usize> = StableDiGraph::default();
    /// let a = dag.add_node('a');
    /// let b = dag.add_node('b');
    /// let c = dag.add_node('c');
    /// let d = dag.add_node('d');
    /// dag.add_edge(a.clone(), b.clone(), 0);
    /// dag.add_edge(a.clone(), c.clone(), 1);
    /// dag.add_edge(b.clone(), d.clone(), 2);
    /// dag.add_edge(c.clone(), d.clone(), 3);
    ///
    /// let m = dag.contract_nodes_simple([b, c], 'm', true, |&e1, &e2| Ok::<_, Infallible>(if e1 > e2 { e1 } else { e2 } )).unwrap();
    /// assert_eq!(dag.edge_weight(dag.find_edge(a.clone(), m.clone()).unwrap()).unwrap(), &1);
    /// assert_eq!(dag.edge_weight(dag.find_edge(m.clone(), d.clone()).unwrap()).unwrap(), &3);
    /// ```
    fn contract_nodes_simple<I, F>(
        &mut self,
        nodes: I,
        weight: Self::NodeWeight,
        check_cycle: bool,
        weight_combo_fn: F,
    ) -> Result<Self::NodeId, Self::Error<F::Error>>
    where
        I: IntoIterator<Item = Self::NodeId>,
        F: MergeFn<Self::EdgeWeight>;
}

impl<G> ContractNodesDirected for G
where
    G: GraphProp<EdgeType = Directed> + NodeRemovable + Build + Visitable,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Data<EdgeWeight = G::EdgeWeight> + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: Clone,
{
    type Error = ContractError;

    fn contract_nodes<I>(
        &mut self,
        nodes: I,
        obj: Self::NodeWeight,
        check_cycle: bool,
    ) -> Result<Self::NodeId, Self::Error>
    where
        I: IntoIterator<Item = Self::NodeId>,
    {
        let indices_to_remove: IndexSet<G::NodeId, ahash::RandomState> = IndexSet::from_iter(nodes);

        if check_cycle && !can_contract(self.deref(), &indices_to_remove) {
            return Err(ContractError::DAGWouldCycle);
        }

        Ok(contract(self, indices_to_remove, obj, NoCallback::None).unwrap())
    }
}

impl<G> ContractNodesSimpleDirected for G
where
    G: GraphProp<EdgeType = Directed> + NodeRemovable + Build + Visitable,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Data<EdgeWeight = G::EdgeWeight> + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: Clone,
{
    type Error<E: Error> = ContractSimpleError<E>;

    fn contract_nodes_simple<I, F>(
        &mut self,
        nodes: I,
        obj: Self::NodeWeight,
        check_cycle: bool,
        weight_combo_fn: F,
    ) -> Result<Self::NodeId, Self::Error<F::Error>>
    where
        I: IntoIterator<Item = Self::NodeId>,
        F: MergeFn<Self::EdgeWeight>,
    {
        let indices_to_remove: IndexSet<G::NodeId, ahash::RandomState> = IndexSet::from_iter(nodes);

        if check_cycle && !can_contract(self.deref(), &indices_to_remove) {
            return Err(ContractSimpleError::DAGWouldCycle);
        }

        contract(self, indices_to_remove, obj, Some(weight_combo_fn))
            .map_err(ContractSimpleError::MergeError)
    }
}

pub trait ContractNodesUndirected: Data {
    type Error;

    /// Substitute a set of nodes with a single new node.
    ///
    /// The specified `nodes` are removed and replaced with a new node
    /// with the given `weight`. Any nodes not in the graph are ignored.
    /// It is valid for `nodes` to be empty, in which case the new node
    /// is added to the graph without edges.
    ///
    /// The contraction may result in multiple edges between nodes if
    /// the underlying graph is a multi-graph. If this is not desired,
    /// use [ContractNodesSimpleUndirected::contract_nodes_simple].
    ///
    /// The `NodeId` of the newly created node is returned.
    ///
    /// # Example
    /// ```
    /// use std::convert::Infallible;
    /// use petgraph::prelude::*;
    /// use rustworkx_core::graph::undirected::*;
    ///
    /// // Performs the following transformation:
    /// //      ┌─┐
    /// //      │a│
    /// //      └┬┘              ┌─┐
    /// //       0               │a│
    /// //      ┌┴┐              └┬┘
    /// //      │b│               0
    /// //      └┬┘              ┌┴┐
    /// //       1      ───►     │m│
    /// //      ┌┴┐              └┬┘
    /// //      │c│               2
    /// //      └┬┘              ┌┴┐
    /// //       2               │d│
    /// //      ┌┴┐              └─┘
    /// //      │d│
    /// //      └─┘
    /// let mut dag: StableUnGraph<char, usize> = StableUnGraph::default();
    /// let a = dag.add_node('a');
    /// let b = dag.add_node('b');
    /// let c = dag.add_node('c');
    /// let d = dag.add_node('d');
    /// dag.add_edge(a.clone(), b.clone(), 0);
    /// dag.add_edge(b.clone(), c.clone(), 1);
    /// dag.add_edge(c.clone(), d.clone(), 2);
    ///
    /// let m = dag.contract_nodes([b, c], 'm').unwrap();
    /// assert_eq!(dag.edge_weight(dag.find_edge(a.clone(), m.clone()).unwrap()).unwrap(), &0);
    /// assert_eq!(dag.edge_weight(dag.find_edge(m.clone(), d.clone()).unwrap()).unwrap(), &2);
    /// ```
    fn contract_nodes<I>(
        &mut self,
        nodes: I,
        weight: Self::NodeWeight,
    ) -> Result<Self::NodeId, Self::Error>
    where
        I: IntoIterator<Item = Self::NodeId>;
}

pub trait ContractNodesSimpleUndirected: Data {
    type Error<E: Error>;

    /// Substitute a set of nodes with a single new node.
    ///
    /// The specified `nodes` are removed and replaced with a new node
    /// with the given `weight`. Any nodes not in the graph are ignored.
    /// It is valid for `nodes` to be empty, in which case the new node
    /// is added to the graph without edges.
    ///
    /// The specified function `weight_combo_fn` is used to merge
    /// would-be parallel edges during contraction; this function
    /// preserves simple graphs.
    ///
    /// The `NodeId` of the newly created node is returned.
    ///
    /// # Example
    /// ```
    /// use std::convert::Infallible;
    /// use petgraph::prelude::*;
    /// use rustworkx_core::graph::undirected::*;
    ///
    /// // Performs the following transformation:
    /// //                          ┌─┐
    /// //     ┌─┐                  │a│
    /// //  ┌0─┤a├─1┐               └┬┘
    /// //  │  └─┘  │                1
    /// // ┌┴┐     ┌┴┐              ┌┴┐
    /// // │b│     │c│     ───►     │m│
    /// // └┬┘     └┬┘              └┬┘
    /// //  │  ┌─┐  │                3
    /// //  └2─│d├─3┘               ┌┴┐
    /// //     └─┘                  │d│
    /// //                          └─┘
    /// let mut dag: StableUnGraph<char, usize> = StableUnGraph::default();
    /// let a = dag.add_node('a');
    /// let b = dag.add_node('b');
    /// let c = dag.add_node('c');
    /// let d = dag.add_node('d');
    /// dag.add_edge(a.clone(), b.clone(), 0);
    /// dag.add_edge(a.clone(), c.clone(), 1);
    /// dag.add_edge(b.clone(), d.clone(), 2);
    /// dag.add_edge(c.clone(), d.clone(), 3);
    ///
    /// let m = dag.contract_nodes_simple([b, c], 'm', |&e1, &e2| Ok::<_, Infallible>(if e1 > e2 { e1 } else { e2 } )).unwrap();
    /// assert_eq!(dag.edge_weight(dag.find_edge(a.clone(), m.clone()).unwrap()).unwrap(), &1);
    /// assert_eq!(dag.edge_weight(dag.find_edge(m.clone(), d.clone()).unwrap()).unwrap(), &3);
    /// ```
    fn contract_nodes_simple<I, F>(
        &mut self,
        nodes: I,
        weight: Self::NodeWeight,
        weight_combo_fn: F,
    ) -> Result<Self::NodeId, Self::Error<F::Error>>
    where
        I: IntoIterator<Item = Self::NodeId>,
        F: MergeFn<Self::EdgeWeight>;
}

impl<G> ContractNodesUndirected for G
where
    G: GraphProp<EdgeType = Undirected> + NodeRemovable + Build + Visitable,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Data<EdgeWeight = G::EdgeWeight> + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: Clone,
{
    type Error = ContractError;

    fn contract_nodes<I>(
        &mut self,
        nodes: I,
        obj: Self::NodeWeight,
    ) -> Result<Self::NodeId, Self::Error>
    where
        I: IntoIterator<Item = Self::NodeId>,
    {
        Ok(contract(self, IndexSet::from_iter(nodes), obj, NoCallback::None).unwrap())
    }
}

impl<G> ContractNodesSimpleUndirected for G
where
    G: GraphProp<EdgeType = Undirected> + NodeRemovable + Build + Visitable,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Data<EdgeWeight = G::EdgeWeight> + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: Clone,
{
    type Error<E: Error> = ContractSimpleError<E>;

    fn contract_nodes_simple<I, F>(
        &mut self,
        nodes: I,
        obj: Self::NodeWeight,
        weight_combo_fn: F,
    ) -> Result<Self::NodeId, Self::Error<F::Error>>
    where
        I: IntoIterator<Item = Self::NodeId>,
        F: MergeFn<Self::EdgeWeight>,
    {
        contract(self, IndexSet::from_iter(nodes), obj, Some(weight_combo_fn))
            .map_err(ContractSimpleError::MergeError)
    }
}

fn merge_duplicates<K, V, F, E>(xs: Vec<(K, V)>, mut merge_fn: F) -> Result<Vec<(K, V)>, E>
where
    K: Hash + Eq,
    F: FnMut(&V, &V) -> Result<V, E>,
{
    let mut kvs = DictMap::with_capacity(xs.len());
    for (k, v) in xs {
        match kvs.entry(k) {
            Occupied(entry) => {
                *entry.into_mut() = merge_fn(&v, entry.get())?;
            }
            Vacant(entry) => {
                entry.insert(v);
            }
        }
    }
    Ok(kvs.into_iter().collect::<Vec<_>>())
}

fn can_contract<G>(graph: G, nodes: &IndexSet<G::NodeId, ahash::RandomState>) -> bool
where
    G: Data + Visitable + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
{
    // Start with successors of `nodes` that aren't in `nodes` itself.
    let visit_next: Vec<G::NodeId> = nodes
        .iter()
        .flat_map(|n| graph.edges(*n))
        .filter_map(|edge| {
            let target_node = edge.target();
            if !nodes.contains(&target_node) {
                Some(target_node)
            } else {
                None
            }
        })
        .collect();

    // Now, if we can reach any of `nodes`, there exists a path from `nodes`
    // back to `nodes` of length > 1, meaning contraction is disallowed.
    let mut dfs = Dfs::from_parts(visit_next, graph.visit_map());
    while let Some(node) = dfs.next(graph) {
        if nodes.contains(&node) {
            // we found a path back to `nodes`
            return false;
        }
    }
    true
}

// Helper type for specifying `NoCallback::None` at callsites of `contract`.
type NoCallback<E> = Option<fn(&E, &E) -> Result<E, Infallible>>;

fn contract<G, F, E>(
    graph: &mut G,
    mut nodes: IndexSet<G::NodeId, ahash::RandomState>,
    obj: G::NodeWeight,
    mut weight_combo_fn: Option<F>,
) -> Result<G::NodeId, E>
where
    G: GraphProp + NodeRemovable + Build + Visitable,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + Data<EdgeWeight = G::EdgeWeight> + IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeWeight: Clone,
    F: FnMut(&G::EdgeWeight, &G::EdgeWeight) -> Result<G::EdgeWeight, E>,
{
    let node_index = graph.add_node(obj);

    // Sanitize new node index from user input.
    nodes.swap_remove(&node_index);

    // Determine and add edges for new node.
    {
        // Note: even when the graph is undirected, we used edges_directed because
        // it gives us a consistent endpoint order.
        let mut incoming_edges: Vec<(G::NodeId, G::EdgeWeight)> = nodes
            .iter()
            .flat_map(|i| graph.edges_directed(*i, Direction::Incoming))
            .filter_map(|edge| {
                let pred = edge.source();
                if !nodes.contains(&pred) {
                    Some((pred, edge.weight().clone()))
                } else {
                    None
                }
            })
            .collect();

        if let Some(merge_fn) = &mut weight_combo_fn {
            incoming_edges = merge_duplicates(incoming_edges, merge_fn)?;
        }

        for (source, weight) in incoming_edges.into_iter() {
            graph.add_edge(source, node_index, weight);
        }
    }

    if graph.is_directed() {
        let mut outgoing_edges: Vec<(G::NodeId, G::EdgeWeight)> = nodes
            .iter()
            .flat_map(|&i| graph.edges_directed(i, Direction::Outgoing))
            .filter_map(|edge| {
                let succ = edge.target();
                if !nodes.contains(&succ) {
                    Some((succ, edge.weight().clone()))
                } else {
                    None
                }
            })
            .collect();

        if let Some(merge_fn) = &mut weight_combo_fn {
            outgoing_edges = merge_duplicates(outgoing_edges, merge_fn)?;
        }

        for (target, weight) in outgoing_edges.into_iter() {
            graph.add_edge(node_index, target, weight);
        }
    }

    // Remove nodes that have been replaced.
    for index in nodes {
        graph.remove_node(index);
    }

    Ok(node_index)
}
