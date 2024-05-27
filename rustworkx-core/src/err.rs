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

//! This module contains common error types and trait impls.

use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub enum ContractError {
    DAGWouldCycle,
}

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractError::DAGWouldCycle => fmt_dag_would_cycle(f),
        }
    }
}

impl Error for ContractError {}

#[derive(Debug)]
pub enum ContractSimpleError<E: Error> {
    DAGWouldCycle,
    MergeError(E),
}

impl<E: Error> Display for ContractSimpleError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractSimpleError::DAGWouldCycle => fmt_dag_would_cycle(f),
            ContractSimpleError::MergeError(ref e) => fmt_merge_error(f, e),
        }
    }
}

impl<E: Error> Error for ContractSimpleError<E> {}

fn fmt_dag_would_cycle(f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "The operation would introduce a cycle.")
}

fn fmt_merge_error<E: Error>(f: &mut Formatter<'_>, inner: &E) -> std::fmt::Result {
    write!(f, "The prov failed with: {:?}", inner)
}

/// Error returned by Layers function when an index is not part of the graph.
#[derive(Debug, PartialEq, Eq)]
pub struct LayersError(pub Option<String>);

impl Error for LayersError {}

impl Display for LayersError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match &self.0 {
            Some(message) => write!(f, "{message}"),
            None => write!(
                f,
                "The provided layer contains an index that is not present in the graph"
            ),
        }
    }
}
