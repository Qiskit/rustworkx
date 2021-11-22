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

extern crate retworkx_core;

use hashbrown::HashMap;

use std::cell::RefCell;

use retworkx_core::petgraph::graph::node_index as n;
use retworkx_core::petgraph::graph::EdgeReference;
use retworkx_core::petgraph::prelude::*;
use retworkx_core::petgraph::visit::Control;

use retworkx_core::traversal::{dijkstra_search, DijkstraEvent};

fn decr(x: &mut usize) {
    if *x > 0 {
        *x -= 1;
    }
}

#[test]
fn dijkstra_test() {
    let rules: Vec<Vec<usize>> =
        vec![vec![1, 2], vec![3], vec![1], vec![4], vec![5]];

    let gr: DiGraph<(), Option<usize>> = DiGraph::from_edges(&[
        (0, 1, None),
        (0, 2, None),
        (1, 3, Some(0)),
        (2, 3, Some(0)),
        (3, 6, Some(1)),
        (1, 4, Some(2)),
        (4, 5, Some(3)),
        (5, 6, Some(4)),
    ]);

    let mut predecessor = HashMap::new();
    let shared_scores_map: RefCell<HashMap<usize, usize>> =
        RefCell::new(HashMap::new());

    let mut num_nodes_remain_for_rule: Vec<_> =
        rules.iter().map(|r| r.len()).collect();

    let start = 0;
    let goal = 6;

    let res = dijkstra_search(
        &gr,
        Some(n(start)),
        |edge: EdgeReference<'_, Option<usize>>| -> Result<usize, ()> {
            match edge.weight() {
                Some(ri) => {
                    let source = edge.source().index();
                    let scores_map = shared_scores_map.borrow();
                    let tot_cost: usize =
                        rules[*ri].iter().map(|nx| scores_map[nx]).sum();
                    Ok(tot_cost - scores_map[&source])
                }
                None => Ok(1),
            }
        },
        |event| -> Control<usize> {
            match event {
                DijkstraEvent::<NodeIndex, _, _>::Discover(v, cost) => {
                    let v = v.index();
                    let mut scores_map = shared_scores_map.borrow_mut();
                    scores_map.insert(v, cost);

                    if v == goal {
                        return Control::Break(v);
                    }
                }
                DijkstraEvent::ExamineEdge(_, _, Some(ri)) => {
                    if let Some(x) = num_nodes_remain_for_rule.get_mut(*ri) {
                        decr(x);
                        if *x > 0 {
                            return Control::Prune;
                        }
                    }
                }
                DijkstraEvent::EdgeRelaxed(u, v, _) => {
                    predecessor.insert(v, u);
                }
                _ => {}
            }
            Control::Continue
        },
    );
    res.unwrap();

    let scores_map = shared_scores_map.borrow();
    assert_eq!(scores_map[&goal], 1);
    println!("{:?}", scores_map);
}
