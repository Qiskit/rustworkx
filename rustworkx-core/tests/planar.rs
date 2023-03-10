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

//! Test module for planar graphs.

use rustworkx_core::petgraph::graph::UnGraph;
use rustworkx_core::planar::is_planar;

#[test]
fn test_simple_planar_graph() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 6),
        (6, 7),
        (7, 1),
        (1, 5),
        (5, 2),
        (2, 4),
        (4, 5),
        (5, 7),
    ]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_planar_grid_3_3_graph() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        // row edges
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        // col edges
        (0, 3),
        (3, 6),
        (1, 4),
        (4, 7),
        (2, 5),
        (5, 8),
    ]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_planar_with_self_loop() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (1, 2),
        (1, 3),
        (1, 5),
        (2, 5),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
    ]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_goldner_harary_planar_graph() {
    // test goldner-harary graph (a maximal planar graph)
    let graph = UnGraph::<(), ()>::from_edges(&[
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 7),
        (1, 8),
        (1, 10),
        (1, 11),
        (2, 3),
        (2, 4),
        (2, 6),
        (2, 7),
        (2, 9),
        (2, 10),
        (2, 11),
        (3, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 8),
        (7, 9),
        (7, 10),
        (8, 10),
        (9, 10),
        (10, 11),
    ]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_multiple_components_planar_graph() {
    let graph = UnGraph::<(), ()>::from_edges(&[(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_planar_multi_graph() {
    let graph = UnGraph::<(), ()>::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2), (2, 0)]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_k3_3_non_planar() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
    ]);
    let res = is_planar(&graph);
    assert_eq!(res, false)
}

#[test]
fn test_k5_non_planar() {
    let graph = UnGraph::<(), ()>::from_edges(&[
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
    let res = is_planar(&graph);
    assert_eq!(res, false)
}

#[test]
fn test_multiple_components_non_planar() {
    let graph = UnGraph::<(), ()>::from_edges(&[
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
        (6, 7),
        (7, 8),
        (8, 6),
    ]);
    let res = is_planar(&graph);
    assert_eq!(res, false)
}

#[test]
fn test_non_planar() {
    // tests a graph that has no subgraph directly isomorphic to K5 or K3_3.
    let graph = UnGraph::<(), ()>::from_edges(&[
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 6),
        (2, 3),
        (3, 5),
        (3, 7),
        (4, 5),
        (4, 6),
        (4, 7),
    ]);
    let res = is_planar(&graph);
    assert_eq!(res, false)
}

#[test]
fn test_planar_graph1() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (3, 10),
        (2, 13),
        (1, 13),
        (7, 11),
        (0, 8),
        (8, 13),
        (0, 2),
        (0, 7),
        (0, 10),
        (1, 7),
    ]);
    let res = is_planar(&graph);
    assert!(res)
}

#[test]
fn test_non_planar_graph2() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (1, 2),
        (4, 13),
        (0, 13),
        (4, 5),
        (7, 10),
        (1, 7),
        (0, 3),
        (2, 6),
        (5, 6),
        (7, 13),
        (4, 8),
        (0, 8),
        (0, 9),
        (2, 13),
        (6, 7),
        (3, 6),
        (2, 8),
    ]);
    let res = is_planar(&graph);
    assert_eq!(res, false)
}

#[test]
fn test_non_planar_graph3() {
    let graph = UnGraph::<(), ()>::from_edges(&[
        (0, 7),
        (3, 11),
        (3, 4),
        (8, 9),
        (4, 11),
        (1, 7),
        (1, 13),
        (1, 11),
        (3, 5),
        (5, 7),
        (1, 3),
        (0, 4),
        (5, 11),
        (5, 13),
    ]);
    let res = is_planar(&graph);
    assert_eq!(res, false)
}
