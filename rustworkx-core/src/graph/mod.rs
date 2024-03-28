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

//! This module defines traits that extend PetGraph's graph
//! data structures.

use petgraph::graph::IndexType;
use petgraph::graphmap::{GraphMap, NodeTrait};
use petgraph::matrix_graph::{MatrixGraph, Nullable};
use petgraph::stable_graph::StableGraph;
use petgraph::visit::Data;
use petgraph::{EdgeType, Graph};

pub mod contraction;
pub mod multigraph;

pub mod directed {
    //! Extension methods for directed graphs.
    pub use super::contraction::{ContractNodesDirected, ContractNodesSimpleDirected};
    pub use super::multigraph::HasParallelEdgesDirected;
}

pub mod undirected {
    //! Extension methods for undirected graphs.
    pub use super::contraction::{ContractNodesSimpleUndirected, ContractNodesUndirected};
    pub use super::multigraph::HasParallelEdgesUndirected;
}

/// A graph whose nodes may be removed.
pub trait NodeRemovable: Data {
    type RemoveResult;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::RemoveResult;
}

impl<N, E, Ty, Ix> NodeRemovable for StableGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type RemoveResult = Option<Self::NodeWeight>;
    fn remove_node(&mut self, node: Self::NodeId) -> Option<Self::NodeWeight> {
        self.remove_node(node)
    }
}

impl<N, E, Ty, Ix> NodeRemovable for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type RemoveResult = Option<Self::NodeWeight>;
    fn remove_node(&mut self, node: Self::NodeId) -> Option<Self::NodeWeight> {
        self.remove_node(node)
    }
}

impl<N, E, Ty> NodeRemovable for GraphMap<N, E, Ty>
where
    N: NodeTrait,
    Ty: EdgeType,
{
    type RemoveResult = bool;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::RemoveResult {
        self.remove_node(node)
    }
}

impl<N, E, Ty: EdgeType, Null: Nullable<Wrapped = E>, Ix: IndexType> NodeRemovable
    for MatrixGraph<N, E, Ty, Null, Ix>
{
    type RemoveResult = Self::NodeWeight;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::RemoveResult {
        self.remove_node(node)
    }
}
