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

use crate::petgraph::algo::Measure;
use crate::petgraph::graph::{DiGraph, NodeIndex};
use crate::petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable};

use std::cmp::Ordering;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::{f32, thread};
// use rand::Rng;

///////////
//             SimplePath
// This is Structure which saves all the context about graph & iterating attributes
// Score : the total weightage of the current shortest path
// Path: Shorted path caulculated in this iteration
// index: used for iterating over edged to delete one and calculate path once again
// Source: to store the start point
// Target: to store goal of path
// graph: store the path to be used
// unique_path: stores all unique_paths to verify that we are not returning same path again after random edge removal
// switch : used for switching to next shortest path in case for one all possible paths are generated.
/////////////

#[derive(Debug, Clone)]
pub struct SimplePath {
    pub Score: f32,
    pub Path: Vec<NodeIndex>,
    index: usize,
    source: NodeIndex,
    target: NodeIndex,
    graph: DiGraph<(), f32>,
    unique_paths: Vec<Vec<NodeIndex>>,
    switch: usize,
}

impl SimplePath {
    fn new(
        graph: &mut DiGraph<(), f32>,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Option<SimplePath> {
	let s = SimplePath {
                    switch: 0,
                    unique_paths: vec![],
                    Score: 0.0,
                    Path: vec!(),
                    index: 0,
                    source: source,
                    target: target,
		    graph: graph.clone(),
        };
        return Some(s);

    }
}

impl Iterator for SimplePath {
    type Item = SimplePath;
    fn next(&mut self) -> Option<Self::Item> {
        let mut graph = self.graph.clone();
        if self.unique_paths.len() == 0 {
            let sim_path = get_simple_path(& graph, self);
            match sim_path {
                None => {
                    self.index = self.index + 1;
                    return self.next();
                }
                _ => {
                    return sim_path;
                }
            }
            
        }

        let mut simple_graph = &self.unique_paths[self.switch];
        let mut index: usize = self.index;

        if index + 1 == simple_graph.len() {
            if self.switch < self.unique_paths.len() + 1 {
                self.switch = self.switch + 1;
                simple_graph = &self.unique_paths[self.switch];
                self.index = 0;
                index = 0;
            } else {
                return None;
            }
        }

        let edge = graph.find_edge(simple_graph[index], simple_graph[index + 1]);
        match edge {
            Some(edge) => {
                let (s, t) = (simple_graph[index], simple_graph[index + 1]);
                let Some(weight) = graph.edge_weight(edge) else {
                    return None;
                };
                let weight = *weight;
                graph.remove_edge(edge);

                let sim_path = get_simple_path(& graph, self);
                graph.add_edge(s, t, weight);

                match sim_path {
                    None => {
                        self.index = self.index + 1;
                        return self.next();
                    }
                    _ => {
                        return sim_path;
                    }
                }
            }
            None => {
                self.index = self.index + 1;
                return self.next();
            }
        }
    }
}
// The code provides the shortest distance cost of all Nodes from the start Node
#[derive(Copy, Clone, Debug)]
struct MinScored<K, T>(pub K, pub T);

impl<K: PartialOrd, T> PartialEq for MinScored<K, T> {
    #[inline]
    fn eq(&self, other: &MinScored<K, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K: PartialOrd, T> Eq for MinScored<K, T> {}

impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    #[inline]
    fn partial_cmp(&self, other: &MinScored<K, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    #[inline]
    fn cmp(&self, other: &MinScored<K, T>) -> Ordering {
        let a = &self.0;
        let b = &other.0;
        if a == b {
            Ordering::Equal
        } else if a < b {
            Ordering::Greater
        } else if a > b {
            Ordering::Less
        } else if a.ne(a) && b.ne(b) {
            // these are the NaN cases
            Ordering::Equal
        } else if a.ne(a) {
            // Order NaN less, so that it is last in the MinScore order
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

// This is mutation of petgraph dijkastra to get full path between source to target
fn dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F,
) -> (HashMap<G::NodeId, K>, HashMap<G::NodeId, Vec<G::NodeId>>)
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
    <G>::NodeId: Debug,
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    let mut tracing: HashMap<G::NodeId, Vec<G::NodeId>> = HashMap::new();
    visit_next.push(MinScored(zero_score, start));
    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        if goal.as_ref() == Some(&node) {
            break;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        *ent.into_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        let Some(v) = tracing.get_mut(&next) else {
                            tracing.insert(next, vec![]);
                            todo!()
                        };
                        v.push(node);
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    tracing.insert(next, vec![]);
                    if tracing.contains_key(&node) {
                        let Some(previous_path) = tracing.get(&node) else {
                            todo!()
                        };
                        let old_v = previous_path.clone();
                        let Some(v) = tracing.get_mut(&next) else {
                            todo!()
                        };
                        for path in old_v {
                            v.push(path);
                        }
                    }

                    let Some(v) = tracing.get_mut(&next) else {
                        todo!()
                    };
                    v.push(node);
                }
            }
        }
        visited.visit(node);
    }

    (scores, tracing)
}

// This function is private to this module, will call Dijkstra algo to get the possible path & Scores & returns a SimplePath as return value

fn get_simple_path(graph: & DiGraph<(), f32>, s: &mut SimplePath) -> Option<SimplePath> {
    let (score, mut path) = dijkstra(&*graph, s.source, Some(s.target), |e| *e.weight());
    let mut score_target: f32 = 0.0;
    let mut unique_paths = s.unique_paths.clone();

    if score.contains_key(&s.target) {
        score_target = *score.get(&s.target).expect("Error");
    }
    for (node, paths) in &mut path {
        if *node == s.target {
            paths.push(*node);
            let contains_target = unique_paths.iter().any(|v| *v == paths.to_vec());
            if !contains_target {
                unique_paths.push(paths.to_vec());
                let s = SimplePath {
                    switch: s.switch,
                    unique_paths: unique_paths,
                    Score: score_target,
                    Path: paths.to_vec(),
                    index: s.index + 1,
                    source: s.source,
                    target: s.target,
                    graph: graph.clone(),
                };
                return Some(s);
            }
        }
    }
    None
}

// This function call get_simple_path for each graph after removing one of the edges in between.

// -------------------------------------------
// INPUTS
// -------------------------------------------
// you can call the function with Input Graph, Source Node, Target Node
// Create a SimplePath instance as -
//       let path = SimplePath::new(&mut graph,source,target);
//       Then iterate over it, as path.next() .
// The Return type is a Option, so you have to handle the None Part as the End of Iterator
////////////////////////////////////////////////

//////////////////////////////////////////////
//  Testing Main function
// fn main() {
//    	let mut graph = DiGraph::new();
//    	let nodes: Vec<NodeIndex> = (0..10000).map(|_| graph.add_node(())).collect();
//    	let mut rng = rand::thread_rng();
//      for _ in 0..50000 { // Adjust the number of edges as desired
//        let a = rng.gen_range(0..nodes.len());
//        let b = rng.gen_range(0..nodes.len());
//        let weight = rng.gen_range(1..100); // Random weight between 1 and 100
//        if a != b { // Prevent self-loops
//            graph.add_edge(nodes[a], nodes[b], weight as f32);
//        }
//      }
//      let source = nodes[10];
//      let target = nodes[800];
//      let mut result =  SimplePath::new(&mut graph,source,target);
//
//      while result.is_some()  {
//        let mut result_new  = result.expect("REASON").next();
//        if result_new.is_none() {
//            break;
//        }
//        println!("New Path & Score {:#?}, {:#?}",result_new.clone().unwrap().Score, result_new.clone().unwrap().Path);
//        result = result_new;
//      }
//    }

// ----------------------------------------------
// OUTPUT
// ----------------------------------------------
//  The function simple_paths_generator will return the Vector of Type SimplePath, which is a structure which contains { Score, Path }
//
//  Example :
//New Path & Score 614.0, [
//    NodeIndex(10),
//    NodeIndex(2636),
//    NodeIndex(8612),
//    NodeIndex(7513),
//    NodeIndex(800),
//]
//New Path & Score 675.0, [
//    NodeIndex(10),
//    NodeIndex(2636),
//    NodeIndex(8612),
//    NodeIndex(7513),
//    NodeIndex(5367),
//    NodeIndex(6520),
//    NodeIndex(5590),
//    NodeIndex(5745),
//    NodeIndex(2596),
//    NodeIndex(4981),
//    NodeIndex(2837),
//    NodeIndex(6319),
//    NodeIndex(4025),
//    NodeIndex(5631),
//    NodeIndex(6935),
//    NodeIndex(2784),
//    NodeIndex(800),
//]
