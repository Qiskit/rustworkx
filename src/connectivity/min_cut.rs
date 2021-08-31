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

use std::cmp::Ordering;
use std::ops::{Add, AddAssign};

use num_traits::Zero;

use priority_queue::PriorityQueue;

use petgraph::stable_graph::{NodeIndex, StableUnGraph};
use petgraph::visit::EdgeRef;

/// `Score<K>` holds a score `K` for use with a `PriorityHeap`.
///
/// **Note:** `Score` implements a total order (`Ord`), so that it is
/// possible to use float types as scores.
#[derive(Clone, Copy, PartialEq)]
pub struct Score<K>(pub K);

impl<K: Add<Output = K>> Add for Score<K> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Score(self.0 + rhs.0)
    }
}

impl<K: AddAssign> AddAssign for Score<K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl<K: Zero> Zero for Score<K> {
    fn zero() -> Self {
        Score(K::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<K: PartialOrd> PartialOrd for Score<K> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: PartialOrd> Eq for Score<K> {}
impl<K: PartialOrd> Ord for Score<K> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Order NaN less, so that it is last in the Score.
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

fn zip<T, U>(a: Option<T>, b: Option<U>) -> Option<(T, U)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn stoer_wagner_phase<E>(
    graph: &StableUnGraph<Vec<NodeIndex>, E>,
) -> Option<((NodeIndex, NodeIndex), E)>
where
    E: Copy + Ord + Zero + AddAssign,
{
    let mut pq = PriorityQueue::<NodeIndex, E, ahash::RandomState>::from(
        graph
            .node_indices()
            .map(|nx| (nx, E::zero()))
            .collect::<Vec<(NodeIndex, E)>>(),
    );

    let mut cut_w = None;
    let (mut s, mut t) = (None, None);
    while let Some((nx, nx_val)) = pq.pop() {
        s = t;
        t = Some(nx);
        cut_w = Some(nx_val);
        for edge in graph.edges(nx) {
            pq.change_priority_by(&edge.target(), |x| {
                *x += *edge.weight();
            })
        }
    }

    zip(zip(s, t), cut_w)
}

pub fn stoer_wagner_min_cut<E>(
    mut graph: StableUnGraph<Vec<NodeIndex>, E>,
) -> Option<(E, Vec<NodeIndex>)>
where
    E: Copy + Ord + Zero + AddAssign,
{
    let (mut min_cut, mut min_cut_val) = (None, None);
    for _ in 1..graph.node_count() {
        if let Some(((s, t), cut_w)) = stoer_wagner_phase(&graph) {
            if min_cut_val.is_none() || Some(cut_w) < min_cut_val {
                min_cut = graph.node_weight(t).cloned();
                min_cut_val = Some(cut_w);
            }
            // now merge nodes ``s`` and  ``t``.
            let edges = graph
                .edges(t)
                .map(|edge| (s, edge.target(), *edge.weight()))
                .collect::<Vec<(NodeIndex, NodeIndex, E)>>();
            for (source, target, weight) in edges {
                graph.add_edge(source, target, weight);
            }
            let mut cluster_t = graph.remove_node(t).unwrap();
            graph.node_weight_mut(s).unwrap().append(&mut cluster_t);
        }
    }

    zip(min_cut_val, min_cut)
}
