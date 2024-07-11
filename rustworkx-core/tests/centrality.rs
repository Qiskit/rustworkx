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

use petgraph::visit::Reversed;
use rustworkx_core::centrality::closeness_centrality;
use rustworkx_core::petgraph::graph::{DiGraph, UnGraph};

#[test]
fn test_simple() {
    let g = UnGraph::<i32, ()>::from_edges([(1, 2), (2, 3), (3, 4), (1, 4)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![
            Some(0.0),
            Some(0.5625),
            Some(0.5625),
            Some(0.5625),
            Some(0.5625)
        ],
        c
    );
}

#[test]
fn test_wf_improved() {
    let g = UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![
            Some(1.0 / 4.0),
            Some(3.0 / 8.0),
            Some(3.0 / 8.0),
            Some(1.0 / 4.0),
            Some(2.0 / 9.0),
            Some(1.0 / 3.0),
            Some(2.0 / 9.0)
        ],
        c
    );
    let cwf = closeness_centrality(&g, false);
    assert_eq!(
        vec![
            Some(1.0 / 2.0),
            Some(3.0 / 4.0),
            Some(3.0 / 4.0),
            Some(1.0 / 2.0),
            Some(2.0 / 3.0),
            Some(1.0),
            Some(2.0 / 3.0)
        ],
        cwf
    );
}

#[test]
fn test_digraph() {
    let g = DiGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(vec![Some(0.), Some(1. / 2.), Some(2. / 3.)], c);

    let cr = closeness_centrality(Reversed(&g), true);
    assert_eq!(vec![Some(2. / 3.), Some(1. / 2.), Some(0.)], cr);
}

#[test]
fn test_k5() {
    let g = UnGraph::<i32, ()>::from_edges([
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
        c
    );
}

#[test]
fn test_path() {
    let g = UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(vec![Some(2.0 / 3.0), Some(1.0), Some(2.0 / 3.0)], c);
}
