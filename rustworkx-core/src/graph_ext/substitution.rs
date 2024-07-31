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
use petgraph::data::DataMap;
use petgraph::stable_graph;
use petgraph::visit::{
    Data, EdgeRef, GraphBase, IntoEdgeReferences, IntoNodeReferences, NodeCount, NodeRef,
};
use petgraph::{Directed, Direction};
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::hash::Hash;

#[derive(Debug)]
pub enum SubstituteNodeWithGraphError<EME, NFE, ETE> {
    EdgeMapErr(EME),
    NodeFilterErr(NFE),
    EdgeWeightTransformErr(ETE),
}

impl<EME: Error, NFE: Error, ETE: Error> Display for SubstituteNodeWithGraphError<EME, NFE, ETE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstituteNodeWithGraphError::EdgeMapErr(e) => {
                write!(f, "Edge map callback failed with: {}", e)
            }
            SubstituteNodeWithGraphError::NodeFilterErr(e) => {
                write!(f, "Node filter callback failed with: {}", e)
            }
            SubstituteNodeWithGraphError::EdgeWeightTransformErr(e) => {
                write!(f, "Edge weight transform callback failed with: {}", e)
            }
        }
    }
}

impl<EME: Error, NFE: Error, ETE: Error> Error for SubstituteNodeWithGraphError<EME, NFE, ETE> {}

pub struct NoCallback;

pub trait NodeFilter<G0: GraphBase> {
    type Error;
    fn enabled(&self) -> bool;
    fn filter(&mut self, _g0: &G0, _n0: G0::NodeId) -> Result<bool, Self::Error>;
}

impl<G0: GraphBase> NodeFilter<G0> for NoCallback {
    type Error = Infallible;
    #[inline]
    fn enabled(&self) -> bool {
        false
    }
    #[inline]
    fn filter(&mut self, _g0: &G0, _n0: G0::NodeId) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl<G0, F, E> NodeFilter<G0> for F
where
    G0: GraphBase + DataMap,
    F: FnMut(&G0::NodeWeight) -> Result<bool, E>,
{
    type Error = E;
    #[inline]
    fn enabled(&self) -> bool {
        true
    }
    #[inline]
    fn filter(&mut self, g0: &G0, n0: G0::NodeId) -> Result<bool, Self::Error> {
        if let Some(x) = g0.node_weight(n0) {
            self(x)
        } else {
            Ok(false)
        }
    }
}

pub trait EdgeWeightMapper<G: Data> {
    type Error;
    type MappedWeight;

    fn map(&mut self, g: &G, e: G::EdgeId) -> Result<Self::MappedWeight, Self::Error>;
}

impl<G: DataMap> EdgeWeightMapper<G> for NoCallback
where
    G::EdgeWeight: Clone,
{
    type Error = Infallible;
    type MappedWeight = G::EdgeWeight;
    #[inline]
    fn map(&mut self, g: &G, e: G::EdgeId) -> Result<Self::MappedWeight, Self::Error> {
        Ok(g.edge_weight(e).unwrap().clone())
    }
}

impl<G0, EW, F, E> EdgeWeightMapper<G0> for F
where
    G0: GraphBase + DataMap,
    F: FnMut(&G0::EdgeWeight) -> Result<EW, E>,
{
    type Error = E;
    type MappedWeight = EW;

    #[inline]
    fn map(&mut self, g0: &G0, e0: G0::EdgeId) -> Result<Self::MappedWeight, Self::Error> {
        if let Some(x) = g0.edge_weight(e0) {
            self(x)
        } else {
            panic!("Edge MUST exist in graph.")
        }
    }
}
pub trait SubstituteNodeWithGraph: DataMap {
    /// The error type returned by the substitution.
    type Error<EME: Error, NME: Error, ETE: Error>: Error;

    /// Substitute a node with a Graph.
    ///
    /// The specified `node` is replaced with the Graph `other`.
    ///
    /// To control the
    ///
    /// :param int node: The node to replace with the PyDiGraph object
    /// :param PyDiGraph other: The other graph to replace ``node`` with
    /// :param callable edge_map_fn: A callable object that will take 3 position
    ///     parameters, ``(source, target, weight)`` to represent an edge either to
    ///     or from ``node`` in this graph. The expected return value from this
    ///     callable is the node index of the node in ``other`` that an edge should
    ///     be to/from. If None is returned, that edge will be skipped and not
    ///     be copied.
    /// :param callable node_filter: An optional callable object that when used
    ///     will receive a node's payload object from ``other`` and return
    ///     ``True`` if that node is to be included in the graph or not.
    /// :param callable edge_weight_map: An optional callable object that when
    ///     used will receive an edge's weight/data payload from ``other`` and
    ///     will return an object to use as the weight for a newly created edge
    ///     after the edge is mapped from ``other``. If not specified the weight
    ///     from the edge in ``other`` will be copied by reference and used.
    ///
    /// :returns: A mapping of node indices in ``other`` to the equivalent node
    ///     in this graph.
    /// :rtype: NodeMap
    ///
    /// .. note::
    ///
    ///    The return type is a :class:`rustworkx.NodeMap` which is an unordered
    ///    type. So it does not provide a deterministic ordering between objects
    ///    when iterated over (although the same object will have a consistent
    ///    order when iterated over multiple times).
    fn substitute_node_with_graph<G1, EM, NF, ET, EME: Error>(
        &mut self,
        node: Self::NodeId,
        other: &G1,
        edge_map_fn: EM,
        node_filter: NF,
        edge_weight_map: ET,
    ) -> Result<DictMap<G1::NodeId, Self::NodeId>, Self::Error<EME, NF::Error, ET::Error>>
    where
        G1: Data<NodeWeight = Self::NodeWeight> + DataMap + NodeCount,
        G1::NodeId: Hash + Eq,
        G1::NodeWeight: Clone,
        for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
            + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
            + IntoNodeReferences
            + IntoEdgeReferences,
        EM: FnMut(Direction, Self::NodeId, &Self::EdgeWeight) -> Result<Option<G1::NodeId>, EME>,
        NF: NodeFilter<G1>,
        ET: EdgeWeightMapper<G1, MappedWeight = Self::EdgeWeight>,
        NF::Error: Error,
        ET::Error: Error;
}

impl<N, E, Ix> SubstituteNodeWithGraph for stable_graph::StableGraph<N, E, Directed, Ix>
where
    Ix: stable_graph::IndexType,
    E: Clone,
{
    type Error<EME: Error, NFE: Error, ETE: Error> = SubstituteNodeWithGraphError<EME, NFE, ETE>;

    fn substitute_node_with_graph<G1, EM, NF, ET, EME: Error>(
        &mut self,
        node: Self::NodeId,
        other: &G1,
        mut edge_map_fn: EM,
        mut node_filter: NF,
        mut edge_weight_map: ET,
    ) -> Result<DictMap<G1::NodeId, Self::NodeId>, Self::Error<EME, NF::Error, ET::Error>>
    where
        G1: Data<NodeWeight = Self::NodeWeight> + DataMap + NodeCount,
        G1::NodeId: Hash + Eq,
        G1::NodeWeight: Clone,
        for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
            + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
            + IntoNodeReferences
            + IntoEdgeReferences,
        EM: FnMut(Direction, Self::NodeId, &Self::EdgeWeight) -> Result<Option<G1::NodeId>, EME>,
        NF: NodeFilter<G1>,
        ET: EdgeWeightMapper<G1, MappedWeight = Self::EdgeWeight>,
        NF::Error: Error,
        ET::Error: Error,
    {
        let node_index = node;
        if self.node_weight(node_index).is_none() {
            panic!("Node `node` MUST be present in graph.");
        }
        // Copy nodes from other to self
        let mut out_map: DictMap<G1::NodeId, Self::NodeId> =
            DictMap::with_capacity(other.node_count());
        for node in other.node_references() {
            if node_filter.enabled()
                && !node_filter
                    .filter(other, node.id())
                    .map_err(|e| SubstituteNodeWithGraphError::NodeFilterErr(e))?
            {
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
                edge_weight_map
                    .map(other, edge.id())
                    .map_err(|e| SubstituteNodeWithGraphError::EdgeWeightTransformErr(e))?,
            );
        }
        // Add edges to/from node to nodes in other
        let in_edges: Vec<Option<_>> = self
            .edges_directed(node_index, petgraph::Direction::Incoming)
            .map(|edge| {
                let Some(target_in_other) =
                    edge_map_fn(Direction::Incoming, edge.source(), edge.weight())
                        .map_err(|e| SubstituteNodeWithGraphError::EdgeMapErr(e))?
                else {
                    return Ok(None);
                };
                let target_in_self = out_map.get(&target_in_other).unwrap();
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
                        .map_err(|e| SubstituteNodeWithGraphError::EdgeMapErr(e))?
                else {
                    return Ok(None);
                };
                let source_in_self = out_map.get(&source_in_other).unwrap();
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
