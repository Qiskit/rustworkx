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

use crate::petgraph::graph::{DiGraph, NodeIndex};
use crate::petgraph::algo::Measure;
use crate::petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable};

use std::collections::{HashMap, BinaryHeap};
use std::{f32,thread};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::hash::Hash;
use std::cmp::Ordering;
use std::fmt::Debug;
// use rand::Rng;



#[derive(Debug,Clone)]
pub struct  SimplePath {
    Score: f32,
    Path : Vec<NodeIndex>,
}

// The code provides the shortest distance cost of all Nodes from the start Node 
#[derive(Copy, Clone, Debug)] struct MinScored<K, T>(pub K, pub T);

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

) -> (HashMap<G::NodeId, K>,HashMap<G::NodeId, Vec<G::NodeId>>)
where
   G: IntoEdges + Visitable,
   G::NodeId: Eq + Hash,
   F: FnMut(G::EdgeRef) -> K,
   K: Measure + Copy,<G>::NodeId: Debug
{
   let mut visited = graph.visit_map();
   let mut scores = HashMap::new();
   let mut visit_next = BinaryHeap::new();
   let zero_score = K::default();
   scores.insert(start, zero_score );
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
                       let Some(v) = tracing.get_mut(&next) else { tracing.insert(next,vec!()) ; todo!() };
                       v.push(node);
                   }
               }
               Vacant(ent) => {
                        ent.insert(next_score);
                        visit_next.push(MinScored(next_score, next));
                        tracing.insert(next,vec!());
                        if tracing.contains_key(&node) {
                            let Some(previous_path) = tracing.get(&node) else { todo!() };
                            let old_v = previous_path.clone();
                            let Some(v) = tracing.get_mut(&next) else { todo!() };
                            for path in old_v {
                                 v.push(path);
                            }
                        }
                        
                        let Some(v) = tracing.get_mut(&next) else { todo!() };
                        v.push(node);
               }
           }

       }
       visited.visit(node);
       
   }
   (scores,tracing)
}

// This function is private to this module, will call Dijkstra algo to get all possible path & Scores & returns a Simple{Score, Path} as return value

fn get_simple_path(graph: & DiGraph<(), f32>, source : NodeIndex, target : NodeIndex) -> Option<SimplePath>{
    let (score,mut path)=dijkstra(&*graph, source, Some(target),|e| *e.weight()) ;
    let mut score_target :f32 = 0.0;
    if score.contains_key(&target) {
        score_target = *score.get(&target).expect("Error");
    }
    for (node,paths) in &mut path{
        if *node == target {
            paths.push(*node);
            let s = SimplePath{ Score: score_target, Path: paths.to_vec()} ;
            return Some(s); 
        }
    }
    None
}

// This function call get_simple_path for each graph after removing one of the edges in between.

pub fn simple_paths_generator(graph: &mut DiGraph<(), f32>, source : NodeIndex, target : NodeIndex ) -> Vec<SimplePath>   {  
    let mut result : Vec<SimplePath> = Vec::new();
    let mut count =0;
    let mut threads = vec!();
    for edge in graph.edge_indices() {
        if count == 0 {
            let  value_graph = graph.clone();
            let t1 = thread::spawn( move ||  
            { 
                get_simple_path(&value_graph, source, target)
            });
            threads.push(t1);
        }
        
        if let Some((s,t)) = graph.edge_endpoints(edge) {
            if s >= source {
                let Some(weight) = graph.edge_weight(edge) else {panic!("No weigh found")};
                let weight = *weight;
                graph.remove_edge(edge);
                let  value_graph = graph.clone();
                let t1 = thread::spawn(  move ||  { 
                    get_simple_path(&value_graph, source, target)
                 });
                threads.push(t1);
                graph.add_edge(s,t,weight);
   
            }
        }
        count=count+1;
    }
    for t in threads {

        match t.join() {
            Ok(Some(path)) => {
                let contains_target = result.iter().any(|v| v.Path == path.Path.to_vec());
                if ! contains_target {   
                    let s = SimplePath{ Score: path.Score, Path: path.Path.to_vec()} ;
                    result.push(s);
                }


            },
            _ => {}  ,
        }
    }

    result
}

// -------------------------------------------
// INPUTS
// -------------------------------------------
// you can call the function with Input Graph, Source Node, Target Node
// path_finder(&mut graph,source,target);

//  Testing Main function
//  fn main() {
//    	let mut graph = DiGraph::new();
//    	let nodes: Vec<NodeIndex> = (0..1000).map(|_| graph.add_node(())).collect();
//    	let mut rng = rand::thread_rng();
//      for _ in 0..5000 { // Adjust the number of edges as desired
//        let a = rng.gen_range(0..nodes.len());
//        let b = rng.gen_range(0..nodes.len());
//        let weight = rng.gen_range(1..100); // Random weight between 1 and 100
//        if a != b { // Prevent self-loops
//            graph.add_edge(nodes[a], nodes[b], weight as f32);
//        }
//      }
//      let source = nodes[10];
//      let target = nodes[880];
//      let result =  simple_paths_generator(&mut graph,source,target);
//      println!("{:#?}",result) ;
//   }

// ----------------------------------------------
// OUTPUT
// ----------------------------------------------
//  The function simple_paths_generator will return the Vector of Type SimplePath, which is a structure which contains { Score, Path }
//
//  Example : 
//   [
//    SimplePath {
//        Score: 154.0,
//        Path: [
//            NodeIndex(10),
//            NodeIndex(49),
//            NodeIndex(844),
//            NodeIndex(83),
//            NodeIndex(879),
//            NodeIndex(477),
//            NodeIndex(530),
//            NodeIndex(318),
//            NodeIndex(179),
//            NodeIndex(433),
//            NodeIndex(466),
//            NodeIndex(629),
//            NodeIndex(880),
//        ],
//    },
//    SimplePath {
//        Score: 154.0,
//        Path: [
//            NodeIndex(10),
//            NodeIndex(871),
//            NodeIndex(792),
//            NodeIndex(449),
//            NodeIndex(356),
//            NodeIndex(169),
//            NodeIndex(457),
//            NodeIndex(642),
//            NodeIndex(588),
//            NodeIndex(189),
//            NodeIndex(629),
//            NodeIndex(880),
//        ],
//    },
//   ]
