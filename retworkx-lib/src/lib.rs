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

//! # retworkx-lib
//!
//! retworkx-lib is a graph algorithm crate built on top of petgraph.
//! it offers Petgraph graph implementation generic set of functions
//! to run different graph algorithms that are used in the larger retworkx
//! project.
//!
//! ## Usage
//!
//! First add this crate to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! retworkx-lib = "0.11"
//! ```
//!
//! Then in your code, it may be used something like this:
//!
//! ```rust
//! use retworkx_lib::petgraph;
//! use retworkx_lib::centrality::betweenness_centrality;
//!
//! let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
//!     (1, 2), (2, 3), (3, 4), (1, 4)
//! ]);
//! // Calculate the betweeness centrality
//! let output = betweenness_centrality(&g, false, false, 200);
//! assert_eq!(
//!     vec![Some(0.0), Some(0.5), Some(0.5), Some(0.5), Some(0.5)],
//!     output
//! );
//! ```
//!
//! ## Algorithm Modules
//!
//! The crate is organized into
//!
//! * [`centrality`](./centrality/index.html)
//! * [`max_weight_matching`](./max_weight_matching/index.html)
//! * [`shortest_path`](./shortest_path/index.html)
//!
//! ## Release Notes
//!
//! The release notes for retworkx-lib are included as part of the retworkx
//! documentation which is hosted at:
//!
//! <https://qiskit.org/documentation/retworkx/release_notes.html>
//!

/// Module for centrality algorithms
pub mod centrality;
/// Module for depth first search edge methods
pub mod dfs_edges;
/// Module for maximum weight matching algorithmss
pub mod max_weight_matching;
/// Modules for shortest path algorithms
pub mod shortest_path;
// These modules define additional data structures
/// This module contains the DictMap type which is a combination of IndexMap
/// and ahash which is used as a return type for retworkx for compatibility
/// with Python's dict which preserves insertion order
pub mod dictmap;
pub mod min_scored;

// re-export petgraph so there is a consistent version available to users and
// then only need to require retworkx-lib in their dependencies
pub use petgraph;
