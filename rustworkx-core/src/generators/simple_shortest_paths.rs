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

use crate::petgraph::algo::{Measure};
use crate::petgraph::graph::{Graph, NodeIndex};
use crate::petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable, IntoEdgeReferences};
use crate::petgraph::EdgeType;

use std::cmp::Ordering;
use std::vec::IntoIter;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BinaryHeap, HashMap};

use std::fmt::Debug;
use std::hash::Hash;
use std::f32;
use crate::min_scored::MinScored;


// Call dijkastra to get shortest path for a graph, by ignore some edges from the all unique_shortest_paths.
// Returns the score of new path, & the full Path

fn get_simple_paths<T,P,N>(graph: &mut Graph<T, P, N>,source:NodeIndex,target: NodeIndex,unique_paths: &mut Vec<Vec<NodeIndex>>,ignore_edges: &mut Vec<(NodeIndex, NodeIndex)>, index: &mut usize, switch: &mut usize) -> Option<(P,Vec<NodeIndex>)>  
where N: EdgeType,
P: Copy + std::cmp::PartialOrd + std::default::Default  + std::ops::Add<Output = P>  + std::fmt::Debug {
    if *index != 0 || *switch != 0 {

        let mut path = &unique_paths[*switch];
        if *index  >= path.len() -1{
            if *switch < unique_paths.len() - 1 {
                *switch = *switch + 1;
                path = &unique_paths[*switch];
                *index = 1;
            } else {
                return None;
            }
        }
        ignore_edges.push((path[*index-1], path[*index]));
      }
    
  
    let (score, path) = dijkstra(&*graph,source, Some(target), |e| *e.weight(), ignore_edges.clone());
    let mut score_target= P::default();
    let mut paths :Vec<NodeIndex> = vec!();

    if score.contains_key(&target) {
        score_target = *score.get(&target).expect("Error");
    }
    
    if path.contains_key(&target)  {
        paths.push(target);
        let mut node = &target;
        loop { 
            let pre_node = path.get(node).expect("Error");
            paths.push(*pre_node);
            // if you have reached to source from target , then exit, no need to backtrack
            if *pre_node == source { 
                break;
            }
            node = pre_node;
            }
    }

    if paths.len() == 0 {
        *index = *index + 1;
        return get_simple_paths(graph,source,target,unique_paths,ignore_edges,index,switch);
    }
    paths.reverse();

    let contains_target = unique_paths.iter().any(|v| *v == paths.to_vec());
    if !contains_target {
        unique_paths.push(paths.clone());
        *index = *index +1;
        return Some((score_target,paths.clone()));
    } else {
        *index = *index + 1;
        return get_simple_paths(graph,source,target,unique_paths,ignore_edges,index,switch);
    }

}

// This is mutation of petgraph dijkastra to get full path between source to target and to ignore some edges while computing shorest path.

fn dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F, ignore_edges : Vec<(G::NodeId,G::NodeId)>
) -> (HashMap<G::NodeId, K>, HashMap<G::NodeId,G::NodeId>)
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
    <G>::NodeId: Debug, <G as IntoEdgeReferences>::EdgeRef: PartialEq
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    let mut tracing: HashMap<G::NodeId, G::NodeId> = HashMap::new();
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
            let edge_to_check = (node,next);
            if ignore_edges.iter().any(|&edge| edge == edge_to_check ){
                continue;
            }
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        *ent.into_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        if let Some(v) = tracing.get_mut(&next) {
                            *v = node;
                        } else {
                            tracing.insert(next, node);
                            todo!()
                        };
                        
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    tracing.insert(next, node);
                }
            }
        }
        visited.visit(node);
    }
    (scores, tracing)
}

// This is the public function to call all possible paths, then N shortest out of them.


pub fn get_shortest_paths<P,N,T>(graph: &mut Graph<T, P, N>,source: NodeIndex,target: NodeIndex,shortest_path_get : usize) -> IntoIter<Option<Vec< NodeIndex>>>
where N: EdgeType,
P: Copy + std::cmp::PartialOrd  + std::default::Default  + std::ops::Add<Output = P>  + std::fmt::Debug{
    let mut scores :Vec<P> = vec!();
    let mut paths :Vec<Option<Vec<NodeIndex>>> = vec!();
    let mut shortest_paths :Vec<Option<Vec<NodeIndex>>> = vec!();
    let mut ignore_edges : Vec<(NodeIndex, NodeIndex)> =vec!();
    
    let mut index : usize = 0;
    let mut switch : usize =0 ;
    let mut unique_paths: Vec<Vec<NodeIndex>> = vec!();
    while let Some((score,path)) =  get_simple_paths(graph,source,target,&mut unique_paths,&mut ignore_edges,&mut index,&mut switch) {
        scores.push(score);
        paths.push(Some(path));
    }
    for i in 0..scores.len(){
        let mut min_score_index :usize = i;
        for j in i+1..scores.len(){
            if scores[j] < scores[min_score_index] {
                min_score_index =j;
            }
        }
        shortest_paths.push(paths[min_score_index].clone());
        if i == shortest_path_get -1  {
            break;
        }
    }
   // println!("Scores & Paths {:#?} {:#?}", scores, paths);
    return shortest_paths.into_iter()
}



// -------------------------------------------
// TEST CASES
// -------------------------------------------

//////////////////////////////////////////////
//  Testing  function

#[cfg(test)]
mod tests {
    use crate::get_shortest_paths;
    use petgraph::Graph;
    use petgraph::graph::DiGraph;

    #[test]
    fn test_shortest_paths() {
        
        let mut g = Graph::new_undirected();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let f = g.add_node("F");
        g.add_edge(a, b, 7);
        g.add_edge(c, a, 9);
        g.add_edge(a, d, 14);
        g.add_edge(b, c, 10);
        g.add_edge(d, c, 2);
        g.add_edge(d, e, 9);
        g.add_edge(b, f, 15);
        g.add_edge(c, f, 11);
        g.add_edge(e, f, 6);
      let source = a;
      let target = f;
      let mut path1 = [
        a,
        c,
        f,
      ];

      let path2 = [
        a,
        b,
        f,
       ];
    
      for path in get_shortest_paths( &mut g,source,target,2){
        match path {
        Some(p) => assert_eq!(p, path1),
        None => panic!("Not matched"),
        }
        path1 = path2;
      }

}
}

