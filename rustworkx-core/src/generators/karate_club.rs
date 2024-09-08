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

use std::hash::Hash;

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, NodeIndexable};

use super::InvalidInputError;

pub fn karate_club<G, T, M>(
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    G::NodeId: Eq + Hash,
{
    let n = 34;
    let m = 78;
    let graph = G::with_capacity(n, m);
    Ok(graph)
}