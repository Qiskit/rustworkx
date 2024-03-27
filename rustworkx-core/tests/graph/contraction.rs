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

use ahash::HashSet;
use hashbrown::HashMap;
use petgraph::data::Build;
use petgraph::visit::{
    Data, EdgeCount, EdgeRef, GraphBase, IntoEdgeReferences, IntoNodeIdentifiers,
};
use rustworkx_core::err::ContractError;
use rustworkx_core::graph::directed::*;
use rustworkx_core::graph::NodeRemovable;
use std::convert::Infallible;
use std::fmt::Debug;
use std::hash::Hash;

mod graph_map {
    use petgraph::prelude::*;
    type G = DiGraphMap<char, usize>;

    common_test!(test_cycle_check_enabled, G);
    common_test!(test_cycle_check_disabled, G);
    common_test!(test_empty_nodes, G);
    common_test!(test_unknown_nodes, G);
    common_test!(test_cycle_path_len_gt_1, G);
    common_test!(test_multiple_paths_would_cycle, G);
    common_test!(test_replace_node_no_neighbors, G);
    common_test!(test_keep_edges_multigraph, G);
    common_test!(test_collapse_parallel_edges, G);
    common_test!(test_replace_all_nodes, G);
}

mod stable_graph {
    use petgraph::prelude::*;
    type G = StableDiGraph<char, usize>;

    common_test!(test_cycle_check_enabled, G);
    common_test!(test_cycle_check_disabled, G);
    common_test!(test_empty_nodes, G);
    common_test!(test_unknown_nodes, G);
    common_test!(test_cycle_path_len_gt_1, G);
    common_test!(test_multiple_paths_would_cycle, G);
    common_test!(test_replace_node_no_neighbors, G);
    common_test!(test_keep_edges_multigraph, G);
    common_test!(test_collapse_parallel_edges, G);
    common_test!(test_replace_all_nodes, G);
}

///          ┌─┐                         ┌─┐
///        ┌─┤a│               ┌─────────┤m│
///        │ └─┘               │         └▲┘
///       ┌▼┐                 ┌▼┐         │
///       │b│          ───►   │b├─────────┘
///       └┬┘                 └─┘
///        │  ┌─┐
///        └─►┤c│
///           └─┘
pub fn test_cycle_check_enabled<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    G::NodeId: Debug,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    dag.add_edge(a, b, 1);
    dag.add_edge(b, c, 2);
    let result = dag.contract_nodes([a, c], 'm', true);
    match result.expect_err("Cycle should cause return error.") {
        ContractError::DAGWouldCycle => (),
    }
}

fn test_cycle_check_disabled<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    G::NodeId: Debug,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    dag.add_edge(a, b, 1);
    dag.add_edge(b, c, 2);
    let result = dag.contract_nodes([a, c], 'm', false);
    result.expect("No error should be raised for a cycle when cycle check is disabled.");
}

fn test_empty_nodes<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    G::NodeId: Debug,
{
    let mut dag = G::default();
    dag.contract_nodes([], 'm', false).unwrap();
    assert_eq!(dag.node_count(), 1);
}

fn test_unknown_nodes<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>
        + NodeRemovable,
    G::NodeId: Debug + Copy,
{
    let mut dag = G::default();

    // A -> B -> C
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');

    dag.add_edge(a, b, 1);
    dag.add_edge(b, c, 2);

    // Leave just A.
    dag.remove_node(b);
    dag.remove_node(c);

    // Replacement should ignore the unknown nodes, making
    // the behavior equivalent to adding a new node in
    // this case.
    dag.contract_nodes([b, c], 'm', false).unwrap();
    assert_eq!(dag.node_count(), 2);
}

///           ┌─┐              ┌─┐
///        ┌4─┤a├─1┐           │m├──1───┐
///        │  └─┘  │           └▲┘      │
///       ┌▼┐     ┌▼┐           │      ┌▼┐
///       │d│     │b│   ───►    │      │b│
///       └▲┘     └┬┘           │      └┬┘
///        │  ┌─┐  2            │  ┌─┐  2
///        └3─┤c│◄─┘            └3─┤c│◄─┘
///           └─┘                  └─┘
fn test_cycle_path_len_gt_1<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>
        + NodeRemovable,
    G::NodeId: Debug + Copy,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    let d = dag.add_node('d');
    dag.add_edge(a, b, 1);
    dag.add_edge(b, c, 2);
    dag.add_edge(c, d, 3);
    dag.add_edge(a, d, 4);

    dag.contract_nodes([a, d], 'm', true)
        .expect_err("Cycle should be detected.");
}

///           ┌─┐     ┌─┐                  ┌─┐     ┌─┐
///        ┌3─┤c│     │e├─5┐            ┌──┤c│     │e├──┐
///        │  └▲┘     └▲┘  │            │  └▲┘     └▲┘  │
///       ┌▼┐  2  ┌─┐  4  ┌▼┐           │   2  ┌─┐  4   │
///       │d│  └──┤b├──┘  │f│   ───►    │   └──┤b├──┘   │
///       └─┘     └▲┘     └─┘           3      └▲┘      5
///                1                    │       1       │
///               ┌┴┐                   │      ┌┴┐      │
///               │a│                   └─────►│m│◄─────┘
///               └─┘                          └─┘
fn test_multiple_paths_would_cycle<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId>
        + Data<EdgeWeight = G::EdgeWeight>
        + IntoEdgeReferences
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug + Copy,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    let d = dag.add_node('d');
    let e = dag.add_node('e');
    let f = dag.add_node('f');

    dag.add_edge(a, b, 1);
    dag.add_edge(b, c, 2);
    dag.add_edge(c, d, 3);
    dag.add_edge(b, e, 4);
    dag.add_edge(e, f, 5);

    let result = dag.contract_nodes([a, d, f], 'm', true);
    match result.expect_err("Cycle should cause return error.") {
        ContractError::DAGWouldCycle => (),
    }

    // Proceed, ignoring cycles.
    dag.contract_nodes([a, d, f], 'm', false)
        .expect("Contraction should be allowed without cycle check.");

    let edge_refs: Vec<_> = dag.edge_references().collect();
    assert_eq!(edge_refs.len(), 5, "Missing expected edge!");

    // Build up a map of node weight to node ID and ensure
    // IDs cross reference as expected between edges.
    let mut seen = HashMap::new();
    for edge_ref in edge_refs.into_iter() {
        match (edge_ref.source(), edge_ref.target(), edge_ref.weight()) {
            (m, b, 1) => {
                assert_eq!(*seen.entry('m').or_insert(m), m);
                assert_eq!(*seen.entry('b').or_insert(b), b);
            }
            (b, c, 2) => {
                assert_eq!(*seen.entry('b').or_insert(b), b);
                assert_eq!(*seen.entry('c').or_insert(c), c);
            }
            (c, m, 3) => {
                assert_eq!(*seen.entry('c').or_insert(c), c);
                assert_eq!(*seen.entry('m').or_insert(m), m);
            }
            (b, e, 4) => {
                assert_eq!(*seen.entry('b').or_insert(b), b);
                assert_eq!(*seen.entry('e').or_insert(e), e);
            }
            (e, m, 5) => {
                assert_eq!(*seen.entry('e').or_insert(e), e);
                assert_eq!(*seen.entry('m').or_insert(m), m);
            }
            (_, _, w) => panic!("Unexpected edge weight: {}", w),
        }
    }

    assert_eq!(seen.len(), 4, "Missing expected node!");
}

fn test_replace_node_no_neighbors<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    G::NodeId: Debug,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    dag.contract_nodes([a], 'm', true).unwrap();
    assert_eq!(dag.node_count(), 1);
}

///          ┌─┐            ┌─┐
///        ┌─┤a│◄┐        ┌─┤a│◄┐
///        │ └─┘ │        │ └─┘ │
///        1     2   ──►  1     2
///       ┌▼┐   ┌┴┐       │ ┌─┐ │
///       │b│   │c│       └►│m├─┘
///       └─┘   └─┘         └─┘
fn test_keep_edges_multigraph<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected<Error = ContractError>,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId>
        + Data<EdgeWeight = G::EdgeWeight>
        + IntoEdgeReferences
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug + Copy,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');

    dag.add_edge(a, b, 1);
    dag.add_edge(c, a, 2);

    let result = dag.contract_nodes([b, c], 'm', true);
    match result.expect_err("Cycle should cause return error.") {
        ContractError::DAGWouldCycle => (),
    }

    // Proceed, ignoring cycles.
    let m = dag
        .contract_nodes([b, c], 'm', false)
        .expect("Contraction should be allowed without cycle check.");

    assert_eq!(dag.node_count(), 2);

    let edges: HashSet<_> = dag
        .edge_references()
        .map(|e| (e.source(), e.target(), *e.weight()))
        .collect();
    let expected = HashSet::from_iter([(a, m, 1), (m, a, 2)]);
    assert_eq!(edges, expected);
}

/// Parallel edges are collapsed using weight_combo_fn.
///           ┌─┐               ┌─┐
///           │a│               │a│
///        ┌──┴┬┴──┐            └┬┘
///        1   2   3             6
///       ┌▼┐ ┌▼┐ ┌▼┐           ┌▼┐
///       │b│ │c│ │d│   ──►     │m│
///       └┬┘ └┬┘ └┬┘           └┬┘
///        4   5   6             15
///        └──►▼◄──┘            ┌▼┐
///           │e│               │e│
///           └─┘               └─┘
fn test_collapse_parallel_edges<G>()
where
    G: Default + Data<NodeWeight = char, EdgeWeight = usize> + Build + ContractNodesSimpleDirected,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId>
        + Data<EdgeWeight = G::EdgeWeight>
        + IntoEdgeReferences
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug + Copy,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    let d = dag.add_node('d');
    let e = dag.add_node('e');

    dag.add_edge(a, b, 1);
    dag.add_edge(a, c, 2);
    dag.add_edge(a, d, 3);
    dag.add_edge(b, e, 4);
    dag.add_edge(c, e, 5);
    dag.add_edge(d, e, 6);

    let m = dag
        .contract_nodes_simple([b, c, d], 'm', true, |w1, w2| {
            Ok::<usize, Infallible>(w1 + w2)
        })
        .unwrap();

    assert_eq!(dag.node_count(), 3);

    let edges: HashSet<_> = dag
        .edge_references()
        .map(|e| (e.source(), e.target(), *e.weight()))
        .collect();
    let expected = HashSet::from_iter([(a, m, 6), (m, e, 15)]);
    assert_eq!(edges, expected);
}

fn test_replace_all_nodes<G>()
where
    G: Default
        + Data<NodeWeight = char, EdgeWeight = usize>
        + Build
        + ContractNodesDirected
        + EdgeCount,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId>
        + Data<EdgeWeight = G::EdgeWeight>
        + IntoEdgeReferences
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug + Copy,
{
    let mut dag = G::default();
    let a = dag.add_node('a');
    let b = dag.add_node('b');
    let c = dag.add_node('c');
    let d = dag.add_node('d');
    let e = dag.add_node('e');

    dag.add_edge(a, b, 1);
    dag.add_edge(a, c, 2);
    dag.add_edge(a, d, 3);
    dag.add_edge(b, e, 4);
    dag.add_edge(c, e, 5);
    dag.add_edge(d, e, 6);

    dag.contract_nodes(dag.node_identifiers().collect::<Vec<_>>(), 'm', true)
        .unwrap();

    assert_eq!(dag.node_count(), 1);
    assert_eq!(dag.edge_count(), 0);
}
