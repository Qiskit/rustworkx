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

//! This module defines graph traits for node substitution.

use crate::dictmap::{DictMap, InitWithHasher};
use petgraph::data::DataMap;
use petgraph::stable_graph;
use petgraph::visit::{
    Data, EdgeRef, GraphBase, IntoEdgeReferences, IntoNodeReferences, NodeCount, NodeRef,
};
use petgraph::{Directed, Direction};
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

#[derive(Debug)]
pub enum SubstituteNodeWithGraphError<N, E> {
    ReplacementGraphIndexError(N),
    CallbackError(E),
}

impl<N: Debug, E: Error> Display for SubstituteNodeWithGraphError<N, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstituteNodeWithGraphError::ReplacementGraphIndexError(n) => {
                write!(f, "Node {:?} was not found in the replacement graph.", n)
            }
            SubstituteNodeWithGraphError::CallbackError(e) => {
                write!(f, "Callback failed with: {}", e)
            }
        }
    }
}

impl<N: Debug, E: Error> Error for SubstituteNodeWithGraphError<N, E> {}

pub type SubstitutionResult<T, N, E> = Result<T, SubstituteNodeWithGraphError<N, E>>;

pub struct NoCallback;

pub trait NodeFilter<G: GraphBase> {
    type CallbackError;
    fn filter(
        &mut self,
        graph: &G,
        node: G::NodeId,
    ) -> SubstitutionResult<bool, G::NodeId, Self::CallbackError>;
}

impl<G: GraphBase> NodeFilter<G> for NoCallback {
    type CallbackError = Infallible;
    #[inline]
    fn filter(
        &mut self,
        _graph: &G,
        _node: G::NodeId,
    ) -> SubstitutionResult<bool, G::NodeId, Self::CallbackError> {
        Ok(true)
    }
}

impl<G, F, E> NodeFilter<G> for F
where
    G: GraphBase + DataMap,
    F: FnMut(&G::NodeWeight) -> Result<bool, E>,
{
    type CallbackError = E;
    #[inline]
    fn filter(
        &mut self,
        graph: &G,
        node: G::NodeId,
    ) -> SubstitutionResult<bool, G::NodeId, Self::CallbackError> {
        if let Some(x) = graph.node_weight(node) {
            self(x).map_err(|e| SubstituteNodeWithGraphError::CallbackError(e))
        } else {
            Ok(false)
        }
    }
}

pub trait EdgeWeightMapper<G: Data> {
    type CallbackError;
    type MappedWeight;
    fn map(
        &mut self,
        graph: &G,
        edge: G::EdgeId,
    ) -> SubstitutionResult<Self::MappedWeight, G::NodeId, Self::CallbackError>;
}

impl<G: DataMap> EdgeWeightMapper<G> for NoCallback
where
    G::EdgeWeight: Clone,
{
    type CallbackError = Infallible;
    type MappedWeight = G::EdgeWeight;
    #[inline]
    fn map(
        &mut self,
        graph: &G,
        edge: G::EdgeId,
    ) -> SubstitutionResult<Self::MappedWeight, G::NodeId, Self::CallbackError> {
        Ok(graph.edge_weight(edge).unwrap().clone())
    }
}

impl<G, EW, F, E> EdgeWeightMapper<G> for F
where
    G: GraphBase + DataMap,
    F: FnMut(&G::EdgeWeight) -> Result<EW, E>,
{
    type CallbackError = E;
    type MappedWeight = EW;

    #[inline]
    fn map(
        &mut self,
        graph: &G,
        edge: G::EdgeId,
    ) -> SubstitutionResult<Self::MappedWeight, G::NodeId, Self::CallbackError> {
        if let Some(x) = graph.edge_weight(edge) {
            self(x).map_err(|e| SubstituteNodeWithGraphError::CallbackError(e))
        } else {
            panic!("Edge MUST exist in graph.")
        }
    }
}

pub trait SubstituteNodeWithGraph: DataMap {
    /// Substitute a node with a Graph.
    ///
    /// The nodes and edges of Graph `other` are cloned into this
    /// graph and connected to its preexisting nodes using an edge mapping
    /// function, `edge_map_fn`.
    ///
    /// The specified `edge_map_fn` is called for each of the edges between
    /// the `node` being replaced and the rest of the graph and is expected
    /// to return an index in `other` that the edge should be connected
    /// to after the replacement, i.e. the node in `graph` that the edge
    /// should be connected to once `node` is gone. It is also acceptable
    /// for `edge_map_fn` to return `None`, in which case the edge is
    /// ignored and will be dropped.
    ///
    /// It accepts the following three arguments:
    ///   - The [Direction], which designates whether the original edge was
    ///     incoming or outgoing to `node`.
    ///   - The [Self::NodeId] of the _other_ node of the original edge (i.e. the
    ///     one that isn't `node`).
    ///   - A reference to the edge weight of the original edge.
    ///
    /// An optional `node_filter` can be provided to ignore nodes in `other` that
    /// should not be copied into this graph. This parameter accepts implementations
    /// of the trait [NodeFilter], which has a blanket implementation for callables
    /// which are `FnMut(&G1::NodeWeight) -> Result<bool, E>`, i.e. functions which
    /// take a reference to a node weight in `other` and return a boolean to indicate
    /// if the node corresponding to this weight should be included or not. To disable
    /// filtering, simply provide [NoCallback].
    ///
    /// A _sometimes_ optional `edge_weight_map` can be provided to transform edge weights from
    /// the source graph `other` into weights of this graph. This parameter accepts
    /// implementations of the trait [EdgeWeightMapper], which has a blanket
    /// implementation for callables which are
    /// `F: FnMut(&G1::EdgeWeight) -> Result<Self::EdgeWeight, E>`,
    /// i.e. functions which take a reference to an edge weight in `graph` and return
    /// an owned weight typed for this graph. An `edge_weight_map` must be provided
    /// when `other` uses a different type for its edge weights, but can otherwise
    /// be specified as [NoCallback] to disable mapping.
    ///
    /// This method returns a mapping of nodes in `other` to the copied node in
    /// this graph.
    #[allow(clippy::type_complexity)]
    fn substitute_node_with_graph<G, EM, NF, ET, E>(
        &mut self,
        node: Self::NodeId,
        other: &G,
        edge_map_fn: EM,
        node_filter: NF,
        edge_weight_map: ET,
    ) -> SubstitutionResult<DictMap<G::NodeId, Self::NodeId>, G::NodeId, E>
    where
        G: Data<NodeWeight = Self::NodeWeight> + DataMap + NodeCount,
        G::NodeId: Debug + Hash + Eq,
        G::NodeWeight: Clone,
        for<'a> &'a G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
            + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
            + IntoNodeReferences
            + IntoEdgeReferences,
        EM: FnMut(Direction, Self::NodeId, &Self::EdgeWeight) -> Result<Option<G::NodeId>, E>,
        NF: NodeFilter<G, CallbackError = E>,
        ET: EdgeWeightMapper<G, MappedWeight = Self::EdgeWeight, CallbackError = E>;
}

impl<N, E, Ix> SubstituteNodeWithGraph for stable_graph::StableGraph<N, E, Directed, Ix>
where
    Ix: stable_graph::IndexType,
    E: Clone,
{
    fn substitute_node_with_graph<G, EM, NF, ET, ER>(
        &mut self,
        node: Self::NodeId,
        other: &G,
        mut edge_map_fn: EM,
        mut node_filter: NF,
        mut edge_weight_map: ET,
    ) -> SubstitutionResult<DictMap<G::NodeId, Self::NodeId>, G::NodeId, ER>
    where
        G: Data<NodeWeight = Self::NodeWeight> + DataMap + NodeCount,
        G::NodeId: Debug + Hash + Eq,
        G::NodeWeight: Clone,
        for<'a> &'a G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
            + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
            + IntoNodeReferences
            + IntoEdgeReferences,
        EM: FnMut(Direction, Self::NodeId, &Self::EdgeWeight) -> Result<Option<G::NodeId>, ER>,
        NF: NodeFilter<G, CallbackError = ER>,
        ET: EdgeWeightMapper<G, MappedWeight = Self::EdgeWeight, CallbackError = ER>,
    {
        let node_index = node;
        if self.node_weight(node_index).is_none() {
            panic!("Node `node` MUST be present in graph.");
        }
        // Copy nodes from other to self
        let mut out_map: DictMap<G::NodeId, Self::NodeId> =
            DictMap::with_capacity(other.node_count());
        for node in other.node_references() {
            if !node_filter.filter(other, node.id())? {
                continue;
            }
            let new_index = self.add_node(node.weight().clone());
            out_map.insert(node.id(), new_index);
        }
        // If no nodes are copied bail here since there is nothing left
        // to do.
        if out_map.is_empty() {
            self.remove_node(node_index);
            // Return a new empty map to clear allocation from out_map
            return Ok(DictMap::new());
        }
        // Copy edges from other to self
        for edge in other.edge_references().filter(|edge| {
            out_map.contains_key(&edge.target()) && out_map.contains_key(&edge.source())
        }) {
            self.add_edge(
                out_map[&edge.source()],
                out_map[&edge.target()],
                edge_weight_map.map(other, edge.id())?,
            );
        }
        // Add edges to/from node to nodes in other
        let in_edges: Vec<Option<_>> = self
            .edges_directed(node_index, petgraph::Direction::Incoming)
            .map(|edge| {
                let Some(target_in_other) =
                    edge_map_fn(Direction::Incoming, edge.source(), edge.weight())
                        .map_err(|e| SubstituteNodeWithGraphError::CallbackError(e))?
                else {
                    return Ok(None);
                };
                let Some(target_in_self) = out_map.get(&target_in_other) else {
                    return Err(SubstituteNodeWithGraphError::ReplacementGraphIndexError(
                        target_in_other,
                    ));
                };
                Ok(Some((
                    edge.source(),
                    *target_in_self,
                    edge.weight().clone(),
                )))
            })
            .collect::<Result<_, _>>()?;
        let out_edges: Vec<Option<_>> = self
            .edges_directed(node_index, petgraph::Direction::Outgoing)
            .map(|edge| {
                let Some(source_in_other) =
                    edge_map_fn(Direction::Outgoing, edge.target(), edge.weight())
                        .map_err(|e| SubstituteNodeWithGraphError::CallbackError(e))?
                else {
                    return Ok(None);
                };
                let Some(source_in_self) = out_map.get(&source_in_other) else {
                    return Err(SubstituteNodeWithGraphError::ReplacementGraphIndexError(
                        source_in_other,
                    ));
                };
                Ok(Some((
                    *source_in_self,
                    edge.target(),
                    edge.weight().clone(),
                )))
            })
            .collect::<Result<_, _>>()?;
        for (source, target, weight) in in_edges
            .into_iter()
            .flatten()
            .chain(out_edges.into_iter().flatten())
        {
            self.add_edge(source, target, weight);
        }
        // Remove node
        self.remove_node(node_index);
        Ok(out_map)
    }
}
