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

//! # rustworkx-core
//!
//! rustworkx-core is a graph algorithm crate built on top of [`petgraph`]. It offers
//! a set of functions that are used in the larger rustworkx project but
//! implemented in a generic manner for use by downstream rust projects.
//!
//! ## Usage
//!
//! First add this crate to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! rustworkx-core = "0.11"
//! ```
//!
//! Then in your code, it may be used something like this:
//!
//! ```rust
//! use rustworkx_core::petgraph;
//! use rustworkx_core::centrality::betweenness_centrality;
//!
//! let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
//!     (1, 2), (2, 3), (3, 4), (1, 4)
//! ]);
//! // Calculate the betweenness centrality
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
//! * [`connectivity`](./connectivity/index.html)
//! * [`max_weight_matching`](./max_weight_matching/index.html)
//! * [`shortest_path`](./shortest_path/index.html)
//! * [`traversal`](./traversal/index.html)
//!
//! ## Release Notes
//!
//! The release notes for rustworkx-core are included as part of the rustworkx
//! documentation which is hosted at:
//!
//! <https://qiskit.org/documentation/rustworkx/release_notes.html>

use std::convert::Infallible;

/// A convenient type alias that by default assumes no error can happen.
///
/// It can be used to avoid type annotations when the function you want
/// to use needs a callback that returns [`Result`] but in your case no
/// error can happen.
pub type Result<T, E = Infallible> = core::result::Result<T, E>;

/// Module for centrality algorithms.
pub mod centrality;
pub mod connectivity;
/// Module for maximum weight matching algorithms.
pub mod max_weight_matching;
pub mod planar;
pub mod shortest_path;
pub mod traversal;
// These modules define additional data structures
pub mod dictmap;
pub mod distancemap;
mod min_scored;

// re-export petgraph so there is a consistent version available to users and
// then only need to require rustworkx-core in their dependencies
pub use petgraph;
