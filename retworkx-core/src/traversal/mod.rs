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

//! Module for graph traversal algorithms.

mod bfs_visit;
mod dfs_edges;
mod dfs_visit;
mod dijkstra_visit;

pub use bfs_visit::{breadth_first_search, BfsEvent};
pub use dfs_edges::dfs_edges;
pub use dfs_visit::{depth_first_search, DfsEvent};
pub use dijkstra_visit::{dijkstra_search, DijkstraEvent};

/// Return if the expression is a break value, execute the provided statement
/// if it is a prune value.
/// https://github.com/petgraph/petgraph/blob/0.6.0/src/visit/dfsvisit.rs#L27
macro_rules! try_control {
    ($e:expr, $p:stmt) => {
        try_control!($e, $p, ());
    };
    ($e:expr, $p:stmt, $q:stmt) => {
        match $e {
            x => {
                if x.should_break() {
                    return x;
                } else if x.should_prune() {
                    $p
                } else {
                    $q
                }
            }
        }
    };
}

use try_control;
