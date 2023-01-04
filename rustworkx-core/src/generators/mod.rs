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

pub use star_graph::star_graph;
