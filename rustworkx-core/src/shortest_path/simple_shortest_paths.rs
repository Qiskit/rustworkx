use crate::petgraph::algo::{Measure};
use crate::petgraph::graph::{Graph, Node, NodeIndex};
use crate::petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable, IntoEdgeReferences};
use crate::petgraph::EdgeType;
use min_scored::MinScored;

use std::cmp::Ordering;
use std::vec::IntoIter;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BinaryHeap, HashMap};

use std::fmt::Debug;
use std::hash::Hash;
use std::f32;
use rand::Rng;


/// This is mutation of petgraph dijkastra to get all possible parents_nodes for a target instead of shortest only
/// Returns the default score Hashmap also a set of (parent_node, node, score) vector.

fn dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F, 
) -> (HashMap<G::NodeId, K>, Vec<(G::NodeId, G::NodeId,K)>)
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
    let mut tracing: Vec<(G::NodeId, G::NodeId,K)> = vec!();
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
                        tracing.push((next, node,next_score));
                        }
                     else {
                        tracing.push((next, node, next_score));
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    tracing.push((next, node,next_score));
                }
            }
        }
        visited.visit(node);
    }
    (scores, tracing)
}

/// To return next possible target for the path.
/// if all possible nodes are traced, then it picks the shortest endpoint and pick the nodes of next nodes.

fn get_smallest_k_element<P>(scores : & HashMap<NodeIndex,P> , visited : &mut Vec<NodeIndex>) -> 
Option<NodeIndex> where P: Copy + std::cmp::PartialOrd  + std::default::Default  + std::ops::Add<Output = P>  + std::fmt::Debug{
    if scores.len() == 1 {
        for (node,_score) in scores {
            return Some(node.clone());
        }
    }
    else {
        let mut score_vec: Vec<_> = scores.iter().collect();
        score_vec.sort_by(|&(_, &score1), &(_, &score2)| {
            score1.partial_cmp(&score2).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut count = 0;
        for (node,_score) in &score_vec {
            if ! visited.contains(node) {
                visited.push(**node);
                return Some(**node); 
            }  
            count = count + 1;
            if count == score_vec.len() {
                return Some(*score_vec[0].0);
            }
        }

    }
    return None;
}

/// pubic function to get all possible paths
/// The dijkastra returns values like -
///  (parent_node, node , total_score_to_reach_<node>_from_root_through_<parent_node>)
/// Using these values for each <node>  we extract and store a Hashmap of <parent_node ,score>.
/// then we backtrack from target till source to get the path. by picking one of the parent_nodes.
/// the last step is done for all possible paths.
/// For visiting all nodes of a parent_node, it only consider the shortest path of the <node> from <aprent_node> -> <node> .

pub fn get_shortest_paths<P,N,T>(graph: &mut Graph<T, P, N>,source: NodeIndex,target: NodeIndex, number_of_paths: Option<i32>) -> IntoIter<Option<Vec< NodeIndex>>>
where N: EdgeType,
P: Copy + std::cmp::PartialOrd  + std::default::Default  + std::ops::Add<Output = P>  + std::fmt::Debug {

    let mut final_paths : HashMap<NodeIndex, HashMap<NodeIndex, P>> = HashMap::new();
    let mut all_paths : HashMap<Vec<NodeIndex>, P> = HashMap::new();
    let mut visited : Vec<NodeIndex> = vec!();
    let mut shortest_paths : Vec<Option<Vec<NodeIndex>>>  = vec!();
    let mut paths : Vec<NodeIndex> = vec!();

    let (score_new, path) = dijkstra(&*graph,source, Some(target), |e| *e.weight());
    if ! score_new.contains_key(&target) {
        shortest_paths.push(None);
        return shortest_paths.into_iter();
    }

    for (node ,next,score) in path {
        if let Some(v) = final_paths.get_mut(&node) {
            v.insert(next,score);
        } else {
            final_paths.insert(node,HashMap::new());
            let Some(v) = final_paths.get_mut(&node)else { todo!() };
            v.insert(next, score);
        };
    }
    
    loop {
        paths.push(target);
        let mut node=target ;
        let mut total_score_path: P = P::default();
        loop { 
            let pre_node = get_smallest_k_element(&final_paths[&node] ,&mut visited);
            match pre_node {
            Some(s) => {
                paths.push(s);
                let edge = graph.find_edge(pre_node.expect("REASON"),node);
                let mut weight : P = P::default();
                match edge {
                    Some(edge) => {
                        weight = *graph.edge_weight(edge).unwrap();
                    },
                    None => {},

                };

                total_score_path = total_score_path + weight;
                if source == s { 
                    // If you have already reach to source from target, the path is complete.
                    break;
                }
                node = s;
                }
            None => {
                break;
            }
            }
            
        }
        paths.reverse();
        
        if all_paths.contains_key(&paths) {
            break;
        }
        all_paths.insert(paths,total_score_path);
        paths = vec!();
        total_score_path = P::default();
   }  
    
    let mut score_vec: Vec<_> = all_paths.iter().collect::<Vec<_>>().clone();
    score_vec.sort_by(|&(_, &score1), &(_, &score2)| {
        score1.partial_cmp(&score2).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut count = 0;
    let mut total_paths = 0;
    match number_of_paths {
        Some(K) => total_paths = K,
        None => total_paths = score_vec.len() as i32,

    }

    for (k,v ) in score_vec {
        // println!("Path {:#?} Score {:#?}",k,v);
        shortest_paths.push(Some(k.clone()));
        count = count +1;
        if count >= total_paths {
            break;
        }
    }

    return shortest_paths.into_iter();
}

///  Test Function
///  the graph can Directed or Undirected.
///  weight is must , pass 1 as weight for each edge if no weight is there for the graph.
///  It verifies all 2 paths generated for the graph.

#[cfg(test)]
mod tests {
    use crate::get_shortest_paths;
    use petgraph::Graph;

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
        g.add_edge(a, c, 9);
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
    
      // You should pass None for parameter number_of_paths if you want all possible shortest paths
      
      for path in get_shortest_paths( &mut g,source,target,Some(2)){
        println!("{:#?}",path);
       match path {
        Some(p) => assert_eq!(p, path1),
        None => panic!("Not matched"),
        }
        path1 = path2;
      }

}
}

