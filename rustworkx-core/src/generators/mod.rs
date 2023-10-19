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

//! This module contains generator functions for building graphs

mod barbell_graph;
mod binomial_tree_graph;
mod complete_graph;
mod cycle_graph;
mod full_rary_tree_graph;
mod grid_graph;
mod heavy_hex_graph;
mod heavy_square_graph;
mod hexagonal_lattice_graph;
mod lollipop_graph;
mod path_graph;
mod petersen_graph;
mod random_graph;
mod star_graph;

mod utils;

use std::{error::Error, fmt};

/// Error returned by generator functions when the input arguments are an
/// invalid combination (such as missing required options).
#[derive(Debug, PartialEq, Eq)]
pub struct InvalidInputError;

impl Error for InvalidInputError {}

impl fmt::Display for InvalidInputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid inputs received.")
    }
}

pub use barbell_graph::barbell_graph;
pub use binomial_tree_graph::binomial_tree_graph;
pub use complete_graph::complete_graph;
pub use cycle_graph::cycle_graph;
pub use full_rary_tree_graph::full_rary_tree_graph;
pub use grid_graph::grid_graph;
pub use heavy_hex_graph::heavy_hex_graph;
pub use heavy_square_graph::heavy_square_graph;
pub use hexagonal_lattice_graph::hexagonal_lattice_graph;
pub use lollipop_graph::lollipop_graph;
pub use path_graph::path_graph;
pub use petersen_graph::petersen_graph;
pub use random_graph::barabasi_albert_graph;
pub use random_graph::gnm_random_graph;
pub use random_graph::gnp_random_graph;
pub use random_graph::random_geometric_graph;
pub use star_graph::star_graph;
