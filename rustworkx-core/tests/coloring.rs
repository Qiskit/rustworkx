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

//! Test module for coloring algorithms.

use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::Undirected;
use rustworkx_core::dictmap::*;

use rustworkx_core::coloring::greedy_color;

#[test]
fn test_greedy_color_empty_graph() {
    let graph = Graph::<(), (), Undirected>::new_undirected();
    let colors = greedy_color(&graph);
    let expected_colors = DictMap::new();
    assert_eq!(colors, expected_colors);
}

#[test]
fn test_greedy_color_simple_graph() {
    let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
    let colors = greedy_color(&graph);
    let mut expected_colors = DictMap::new();
    expected_colors.insert(NodeIndex::new(0), 0);
    expected_colors.insert(NodeIndex::new(1), 1);
    expected_colors.insert(NodeIndex::new(2), 1);
    assert_eq!(colors, expected_colors);
}

#[test]
fn test_greedy_color_simple_graph_large_degree() {
    let graph =
        Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]);
    let colors = greedy_color(&graph);
    let mut expected_colors = DictMap::new();
    expected_colors.insert(NodeIndex::new(0), 0);
    expected_colors.insert(NodeIndex::new(1), 1);
    expected_colors.insert(NodeIndex::new(2), 1);
    assert_eq!(colors, expected_colors);
}
