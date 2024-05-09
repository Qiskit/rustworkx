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
//!
//! The `-Directed` and `-Undirected` trait variants are implemented as
//! applicable for directed and undirected graph types. For example, only
//! directed graph types are concerned with cycle checking and corresponding
//! error handling, so these traits provide applicable parameters and return
//! types to account for this.
//!
//! ### Node Contraction
//!
//! There are four traits related to node contraction available for different
//! graphs / configurations, including:
//!
//! - [`ContractNodesDirected`][ContractNodesDirected]
//! - [`ContractNodesSimpleDirected`][ContractNodesSimpleDirected]
//! - [`ContractNodesUndirected`][ContractNodesUndirected]
//! - [`ContractNodesSimpleUndirected`][ContractNodesSimpleUndirected]
//!
//! Of these, the `ContractNodesSimple-` traits provide a
//! `contract_nodes_simple` method for applicable graph types, which performs
//! node contraction without introducing multiple edge between nodes (edges
//! between any two given nodes are merged via the method's merge function).
//! These traits can be used for node contraction within simple graphs to
//! preserve this property, or on multi-graphs to ensure that the contraction
//! does not introduce additional parallel edges.
//!
//! The other `ContractNodes-` traits provide a `contract_nodes` method, which
//! happily introduces parallel edges when multiple nodes in the contraction
//! have an incoming edge from the same source node or when multiple nodes in
//! the contraction have an outgoing edge to the same target node.
//!
//! ### Multi-graph Extensions
//!
//! These traits provide additional helper methods for use with multi-graphs,
//! e.g. [`HasParallelEdgesDirected`][HasParallelEdgesDirected].
//!
//! ### Graph Extension Trait Implementations
//!
//! The following table lists the traits that are implemented for each graph type:
//!
//! |                               | Graph | StableGraph | GraphMap | MatrixGraph | Csr   | List  |
//! | ----------------------------- | :---: | :---------: | :------: | :---------: | :---: | :---: |
//! | ContractNodesDirected         |       |  x          |    x     |             |       |       |
//! | ContractNodesSimpleDirected   |       |  x          |    x     |             |       |       |
//! | ContractNodesUndirected       |       |  x          |    x     |             |       |       |
//! | ContractNodesSimpleUndirected |       |  x          |    x     |             |       |       |
//! | HasParallelEdgesDirected      | x     |  x          |    x     | x           | x     | x     |
//! | HasParallelEdgesUndirected    | x     |  x          |    x     | x           | x     | x     |
//! | NodeRemovable                 | x     |  x          |    x     | x           |       |       |

use petgraph::graph::IndexType;
use petgraph::graphmap::{GraphMap, NodeTrait};
use petgraph::matrix_graph::{MatrixGraph, Nullable};
use petgraph::stable_graph::StableGraph;
use petgraph::visit::{Data, IntoNodeIdentifiers};
use petgraph::{EdgeType, Graph};

pub mod contraction;
pub mod multigraph;

pub use contraction::{
    ContractNodesDirected, ContractNodesSimpleDirected, ContractNodesSimpleUndirected,
    ContractNodesUndirected,
};
pub use multigraph::{HasParallelEdgesDirected, HasParallelEdgesUndirected};

/// A graph whose nodes may be removed.
pub trait NodeRemovable: Data {
    type Output;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::Output;
}

impl<N, E, Ty, Ix> NodeRemovable for StableGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Output = Option<Self::NodeWeight>;
    fn remove_node(&mut self, node: Self::NodeId) -> Option<Self::NodeWeight> {
        self.remove_node(node)
    }
}

impl<N, E, Ty, Ix> NodeRemovable for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Output = Option<Self::NodeWeight>;
    fn remove_node(&mut self, node: Self::NodeId) -> Option<Self::NodeWeight> {
        self.remove_node(node)
    }
}

impl<N, E, Ty> NodeRemovable for GraphMap<N, E, Ty>
where
    N: NodeTrait,
    Ty: EdgeType,
{
    type Output = bool;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::Output {
        self.remove_node(node)
    }
}

impl<N, E, Ty: EdgeType, Null: Nullable<Wrapped = E>, Ix: IndexType> NodeRemovable
    for MatrixGraph<N, E, Ty, Null, Ix>
{
    type Output = Option<Self::NodeWeight>;
    fn remove_node(&mut self, node: Self::NodeId) -> Self::Output {
        for n in self.node_identifiers() {
            if node == n {
                return Some(self.remove_node(node));
            }
        }
        None
    }
}
