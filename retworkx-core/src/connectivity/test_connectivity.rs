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

use hashbrown::HashSet;
use petgraph::graph::Graph;
use petgraph::Undirected;

use crate::connected_components;

#[test]
fn test_connected_components() {
    let mut graph = Graph::<(), (), Undirected>::from_edges(&[
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)
    ]);
    let components = connected_components(&mut graph);
    let exp1: HashSet<usize> = [0, 1, 3, 2].iter().cloned().collect();
    let exp2: HashSet<usize> = [7, 5, 4, 6].iter().cloned().collect();
    let expected: Vec<_> = vec![exp1, exp2];
    assert_eq!(expected, components);
}