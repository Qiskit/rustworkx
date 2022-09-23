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
use std::hash::Hash;
use std::vec::IntoIter;

use hashbrown::{hash_map::Entry, HashMap};
use petgraph::{
    visit::{
        EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdges, IntoNodeIdentifiers, NodeCount,
        Visitable,
    },
    Undirected,
};

use crate::traversal::{depth_first_search, DfsEvent};

type Edge<G> = (<G as GraphBase>::NodeId, <G as GraphBase>::NodeId);

fn modify_if_min<K, V>(xs: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: Ord + Copy,
{
    xs.entry(key).and_modify(|e| {
        if val < *e {
            *e = val;
        }
    });
}

fn edges_filtered_and_sorted_by<G, P, F, K>(
    graph: G,
    a: G::NodeId,
    filter: P,
    compare: F,
) -> IntoIter<Edge<G>>
where
    G: IntoEdges,
    P: Fn(&Edge<G>) -> bool,
    F: Fn(&Edge<G>) -> K,
    K: Ord,
{
    let mut edges = graph
        .edges(a)
        .filter_map(|edge| {
            let e = (edge.source(), edge.target());
            if filter(&e) {
                Some(e)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(compare);
    // Remove parallel edges since they do *not* affect whether a graph is planar.
    edges.dedup();
    edges.into_iter()
}

fn is_target<G: GraphBase>(edge: Option<&Edge<G>>, v: G::NodeId) -> Option<&Edge<G>> {
    edge.filter(|e| e.1 == v)
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Interval<T> {
    inner: Option<(T, T)>,
}

impl<T> Default for Interval<T> {
    fn default() -> Self {
        Interval { inner: None }
    }
}

impl<T> Interval<T> {
    fn new(low: T, high: T) -> Self {
        Interval {
            inner: Some((low, high)),
        }
    }

    fn is_empty(&self) -> bool {
        self.inner.is_none()
    }

    fn unwrap(self) -> (T, T) {
        self.inner.unwrap()
    }

    fn low(&self) -> Option<&T> {
        match self.inner {
            Some((ref low, _)) => Some(low),
            None => None,
        }
    }

    fn high(&self) -> Option<&T> {
        match self.inner {
            Some((_, ref high)) => Some(high),
            None => None,
        }
    }

    fn as_ref(&mut self) -> Option<&(T, T)> {
        self.inner.as_ref()
    }

    fn as_mut(&mut self) -> Option<&mut (T, T)> {
        self.inner.as_mut()
    }

    fn as_mut_low(&mut self) -> Option<&mut T> {
        match self.inner {
            Some((ref mut low, _)) => Some(low),
            None => None,
        }
    }
}

impl<T> Interval<(T, T)>
where
    T: Copy + Hash + Eq,
{
    /// Returns ``true`` if the interval conflicts with ``edge``.
    fn conflict<G>(&self, lr_state: &LRState<G>, edge: Edge<G>) -> bool
    where
        G: GraphBase<NodeId = T>,
    {
        match self.inner {
            Some((_, ref h)) => lr_state.lowpt.get(h) > lr_state.lowpt.get(&edge),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct ConflictPair<T> {
    left: Interval<T>,
    right: Interval<T>,
}

impl<T> Default for ConflictPair<T> {
    fn default() -> Self {
        ConflictPair {
            left: Interval::default(),
            right: Interval::default(),
        }
    }
}

impl<T> ConflictPair<T> {
    fn new(left: Interval<T>, right: Interval<T>) -> Self {
        ConflictPair { left, right }
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right)
    }

    fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }
}

impl<T> ConflictPair<(T, T)>
where
    T: Copy + Hash + Eq,
{
    /// Returns the lowest low point of a conflict pair.
    fn lowest<G>(&self, lr_state: &LRState<G>) -> usize
    where
        G: GraphBase<NodeId = T>,
    {
        match (self.left.low(), self.right.low()) {
            (Some(l_low), Some(r_low)) => lr_state.lowpt[l_low].min(lr_state.lowpt[r_low]),
            (Some(l_low), None) => lr_state.lowpt[l_low],
            (None, Some(r_low)) => lr_state.lowpt[r_low],
            (None, None) => std::usize::MAX,
        }
    }
}

enum Sign {
    Plus,
    Minus,
}

/// Similar to ``DfsEvent`` plus an extra event ``FinishEdge``
/// that indicates that we have finished processing an edge.
enum LRTestDfsEvent<N> {
    Finish(N),
    TreeEdge(N, N),
    BackEdge(N, N),
    FinishEdge(N, N),
}

// An error: graph is *not* planar.
struct NonPlanar {}

struct LRState<G: GraphBase>
where
    G::NodeId: Hash + Eq,
{
    graph: G,
    /// roots of the DFS forest.
    roots: Vec<G::NodeId>,
    /// distnace from root.
    height: HashMap<G::NodeId, usize>,
    /// parent edge.
    eparent: HashMap<G::NodeId, Edge<G>>,
    /// height of lowest return point.
    lowpt: HashMap<Edge<G>, usize>,
    /// height of next-to-lowest return point. Only used to check if an edge is chordal.
    lowpt_2: HashMap<Edge<G>, usize>,
    /// next back edge in traversal with lowest return point.
    lowpt_edge: HashMap<Edge<G>, Edge<G>>,
    /// proxy for nesting order ≺ given by twice lowpt (plus 1 if chordal).
    nesting_depth: HashMap<Edge<G>, usize>,
    /// stack for conflict pairs.
    stack: Vec<ConflictPair<Edge<G>>>,
    /// marks the top conflict pair when an edge was pushed in the stack.
    stack_emarker: HashMap<Edge<G>, ConflictPair<Edge<G>>>,
    /// edge relative to which side is defined.
    eref: HashMap<Edge<G>, Edge<G>>,
    /// side of edge, or modifier for side of reference edge.
    side: HashMap<Edge<G>, Sign>,
}

impl<G> LRState<G>
where
    G: GraphBase + NodeCount + EdgeCount + IntoEdges + Visitable,
    G::NodeId: Hash + Eq,
{
    fn new(graph: G) -> Self {
        let num_nodes = graph.node_count();
        let num_edges = graph.edge_count();

        LRState {
            graph,
            roots: Vec::new(),
            height: HashMap::with_capacity(num_nodes),
            eparent: HashMap::with_capacity(num_edges),
            lowpt: HashMap::with_capacity(num_edges),
            lowpt_2: HashMap::with_capacity(num_edges),
            lowpt_edge: HashMap::with_capacity(num_edges),
            nesting_depth: HashMap::with_capacity(num_edges),
            stack: Vec::new(),
            stack_emarker: HashMap::with_capacity(num_edges),
            eref: HashMap::with_capacity(num_edges),
            side: graph
                .edge_references()
                .map(|e| ((e.source(), e.target()), Sign::Plus))
                .collect(),
        }
    }

    fn lr_orientation_visitor(&mut self, event: DfsEvent<G::NodeId, &G::EdgeWeight>) {
        match event {
            DfsEvent::Discover(v, _) => {
                if let Entry::Vacant(entry) = self.height.entry(v) {
                    entry.insert(0);
                    self.roots.push(v);
                }
            }
            DfsEvent::TreeEdge(v, w, _) => {
                let ei = (v, w);
                let v_height = self.height[&v];
                let w_height = v_height + 1;

                self.eparent.insert(w, ei);
                self.height.insert(w, w_height);
                // now initialize low points.
                self.lowpt.insert(ei, v_height);
                self.lowpt_2.insert(ei, w_height);
            }
            DfsEvent::BackEdge(v, w, _) => {
                // do *not* consider ``(v, w)`` as a back edge if ``(w, v)`` is a tree edge.
                if Some(&(w, v)) != self.eparent.get(&v) {
                    let ei = (v, w);
                    self.lowpt.insert(ei, self.height[&w]);
                    self.lowpt_2.insert(ei, self.height[&v]);
                }
            }
            DfsEvent::Finish(v, _) => {
                for edge in self.graph.edges(v) {
                    let w = edge.target();
                    let ei = (v, w);

                    // determine nesting depth.
                    let low = match self.lowpt.get(&ei) {
                        Some(val) => *val,
                        None =>
                        // if ``lowpt`` does *not* contain edge ``(v, w)``, it means
                        // that it's *not* a tree or a back edge so we skip it since
                        // it's oriented in the reverse direction.
                        {
                            continue
                        }
                    };

                    if self.lowpt_2[&ei] < self.height[&v] {
                        // if it's chordal, add one.
                        self.nesting_depth.insert(ei, 2 * low + 1);
                    } else {
                        self.nesting_depth.insert(ei, 2 * low);
                    }

                    // update lowpoints of parent edge.
                    if let Some(e_par) = self.eparent.get(&v) {
                        match self.lowpt[&ei].cmp(&self.lowpt[e_par]) {
                            Ordering::Less => {
                                self.lowpt_2
                                    .insert(*e_par, self.lowpt[e_par].min(self.lowpt_2[&ei]));
                                self.lowpt.insert(*e_par, self.lowpt[&ei]);
                            }
                            Ordering::Greater => {
                                modify_if_min(&mut self.lowpt_2, *e_par, self.lowpt[&ei]);
                            }
                            _ => {
                                let val = self.lowpt_2[&ei];
                                modify_if_min(&mut self.lowpt_2, *e_par, val);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn lr_testing_visitor(&mut self, event: LRTestDfsEvent<G::NodeId>) -> Result<(), NonPlanar> {
        match event {
            LRTestDfsEvent::TreeEdge(v, w) => {
                let ei = (v, w);
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(ei, last);
                }
            }
            LRTestDfsEvent::BackEdge(v, w) => {
                let ei = (v, w);
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(ei, last);
                }
                self.lowpt_edge.insert(ei, ei);
                let c_pair = ConflictPair::new(Interval::default(), Interval::new(ei, ei));
                self.stack.push(c_pair);
            }
            LRTestDfsEvent::FinishEdge(v, w) => {
                let ei = (v, w);
                if self.lowpt[&ei] < self.height[&v] {
                    // ei has return edge
                    let e_par = self.eparent[&v];
                    let val = self.lowpt_edge[&ei];

                    match self.lowpt_edge.entry(e_par) {
                        Entry::Occupied(_) => {
                            self.add_constraints(ei, e_par)?;
                        }
                        Entry::Vacant(o) => {
                            o.insert(val);
                        }
                    }
                }
            }
            LRTestDfsEvent::Finish(v) => {
                if let Some(&e) = self.eparent.get(&v) {
                    let u = e.0;
                    self.remove_back_edges(u);

                    // side of ``e = (u, v)` is side of a highest return edge
                    if self.lowpt[&e] < self.height[&u] {
                        if let Some(top) = self.stack.last() {
                            let e_high = match (top.left.high(), top.right.high()) {
                                (Some(hl), Some(hr)) => {
                                    if self.lowpt[hl] > self.lowpt[hr] {
                                        hl
                                    } else {
                                        hr
                                    }
                                }
                                (Some(hl), None) => hl,
                                (None, Some(hr)) => hr,
                                _ => {
                                    // Otherwise ``top`` would be empty, but we don't push
                                    // empty conflict pairs in stack.
                                    unreachable!()
                                }
                            };
                            self.eref.insert(e, *e_high);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn until_top_of_stack_hits_emarker(&mut self, ei: Edge<G>) -> Option<ConflictPair<Edge<G>>> {
        if let Some(&c_pair) = self.stack.last() {
            if self.stack_emarker[&ei] != c_pair {
                return self.stack.pop();
            }
        }

        None
    }

    fn until_top_of_stack_is_conflicting(&mut self, ei: Edge<G>) -> Option<ConflictPair<Edge<G>>> {
        if let Some(c_pair) = self.stack.last() {
            if c_pair.left.conflict(self, ei) || c_pair.right.conflict(self, ei) {
                return self.stack.pop();
            }
        }

        None
    }

    /// Unify intervals ``pi``, ``qi``.
    ///
    /// Interval ``qi`` must be non - empty and contain edges
    /// with smaller lowpt than interval ``pi``.
    fn union_intervals(&mut self, pi: &mut Interval<Edge<G>>, qi: Interval<Edge<G>>) {
        match pi.as_mut_low() {
            Some(p_low) => {
                let (q_low, q_high) = qi.unwrap();
                self.eref.insert(*p_low, q_high);
                *p_low = q_low;
            }
            None => {
                *pi = qi;
            }
        }
    }

    /// Adding constraints associated with edge ``ei``.
    fn add_constraints(&mut self, ei: Edge<G>, e: Edge<G>) -> Result<(), NonPlanar> {
        let mut c_pair = ConflictPair::<Edge<G>>::default();

        // merge return edges of ei into ``c_pair.right``.
        while let Some(mut q_pair) = self.until_top_of_stack_hits_emarker(ei) {
            if !q_pair.left.is_empty() {
                q_pair.swap();

                if !q_pair.left.is_empty() {
                    return Err(NonPlanar {});
                }
            }

            // We call unwrap since ``q_pair`` was in stack and
            // ``q_pair.right``, ``q_pair.left`` can't be both empty
            // since we don't push empty conflict pairs in stack.
            let qr_low = q_pair.right.low().unwrap();
            if self.lowpt[qr_low] > self.lowpt[&e] {
                // merge intervals
                self.union_intervals(&mut c_pair.right, q_pair.right);
            } else {
                // make consinsent
                self.eref.insert(*qr_low, self.lowpt_edge[&e]);
            }
        }

        // merge conflicting return edges of e1, . . . , ei−1 into ``c_pair.left``.
        while let Some(mut q_pair) = self.until_top_of_stack_is_conflicting(ei) {
            if q_pair.right.conflict(self, ei) {
                q_pair.swap();

                if q_pair.right.conflict(self, ei) {
                    return Err(NonPlanar {});
                }
            }

            // merge interval below lowpt(ei) into ``c_pair.right``.
            if let Some((qr_low, qr_high)) = q_pair.right.as_ref() {
                if let Some(pr_low) = c_pair.right.as_mut_low() {
                    self.eref.insert(*pr_low, *qr_high);
                    *pr_low = *qr_low;
                }
            };
            self.union_intervals(&mut c_pair.left, q_pair.left);
        }

        if !c_pair.is_empty() {
            self.stack.push(c_pair);
        }

        Ok(())
    }

    fn until_lowest_top_of_stack_has_height(
        &mut self,
        v: G::NodeId,
    ) -> Option<ConflictPair<Edge<G>>> {
        if let Some(c_pair) = self.stack.last() {
            if c_pair.lowest(self) == self.height[&v] {
                return self.stack.pop();
            }
        }

        None
    }

    fn follow_eref_until_is_target(&self, edge: Edge<G>, v: G::NodeId) -> Option<Edge<G>> {
        let mut res = Some(&edge);
        while let Some(b) = is_target::<G>(res, v) {
            res = self.eref.get(b);
        }

        res.copied()
    }

    /// Trim back edges ending at parent v.
    fn remove_back_edges(&mut self, v: G::NodeId) {
        // drop entire conflict pairs.
        while let Some(c_pair) = self.until_lowest_top_of_stack_has_height(v) {
            if let Some(pl_low) = c_pair.left.low() {
                self.side.insert(*pl_low, Sign::Minus);
            }
        }

        // one more conflict pair to consider.
        if let Some(mut c_pair) = self.stack.pop() {
            // trim left interval.
            if let Some((pl_low, pl_high)) = c_pair.left.as_mut() {
                match self.follow_eref_until_is_target(*pl_high, v) {
                    Some(val) => {
                        *pl_high = val;
                    }
                    None => {
                        // just emptied.
                        // We call unwrap since right interval cannot be empty for otherwise
                        // the entire conflict pair had been removed.
                        let pr_low = c_pair.right.low().unwrap();
                        self.eref.insert(*pl_low, *pr_low);
                        self.side.insert(*pl_low, Sign::Minus);
                        c_pair.left = Interval::default();
                    }
                }
            }

            // trim right interval
            if let Some((pr_low, ref mut pr_high)) = c_pair.right.as_mut() {
                match self.follow_eref_until_is_target(*pr_high, v) {
                    Some(val) => {
                        *pr_high = val;
                    }
                    None => {
                        // just emptied.
                        // We call unwrap since left interval cannot be empty for otherwise
                        // the entire conflict pair had been removed.
                        let pl_low = c_pair.left.low().unwrap();
                        self.eref.insert(*pr_low, *pl_low);
                        self.side.insert(*pr_low, Sign::Minus);
                        c_pair.right = Interval::default();
                    }
                };
            }

            if !c_pair.is_empty() {
                self.stack.push(c_pair);
            }
        }
    }
}

/// Visits the DFS - oriented tree that we have pre-computed
/// and stored in ``lr_state``. We traverse the edges of
/// a node in nesting depth order. Events are emitted at points
/// of interest and should be handled by ``visitor``.
fn lr_visit_ordered_dfs_tree<G, F, E>(
    lr_state: &mut LRState<G>,
    v: G::NodeId,
    mut visitor: F,
) -> Result<(), E>
where
    G: GraphBase + IntoEdges,
    G::NodeId: Hash + Eq,
    F: FnMut(&mut LRState<G>, LRTestDfsEvent<G::NodeId>) -> Result<(), E>,
{
    let mut stack: Vec<(G::NodeId, IntoIter<Edge<G>>)> = vec![(
        v,
        edges_filtered_and_sorted_by(
            lr_state.graph,
            v,
            // if ``lowpt`` does *not* contain edge ``e = (v, w)``, it means
            // that it's *not* a tree or a back edge so we skip it since
            // it's oriented in the reverse direction.
            |e| lr_state.lowpt.contains_key(e),
            // we sort edges based on nesting depth order.
            |e| lr_state.nesting_depth[e],
        ),
    )];

    while let Some(elem) = stack.last_mut() {
        let v = elem.0;
        let adjacent_edges = &mut elem.1;
        let mut next = None;

        for (v, w) in adjacent_edges {
            if Some(&(v, w)) == lr_state.eparent.get(&w) {
                // tree edge
                visitor(lr_state, LRTestDfsEvent::TreeEdge(v, w))?;
                next = Some(w);
                break;
            } else {
                // back edge
                visitor(lr_state, LRTestDfsEvent::BackEdge(v, w))?;
                visitor(lr_state, LRTestDfsEvent::FinishEdge(v, w))?;
            }
        }

        match next {
            Some(w) => stack.push((
                w,
                edges_filtered_and_sorted_by(
                    lr_state.graph,
                    w,
                    |e| lr_state.lowpt.contains_key(e),
                    |e| lr_state.nesting_depth[e],
                ),
            )),
            None => {
                stack.pop();
                visitor(lr_state, LRTestDfsEvent::Finish(v))?;
                if let Some(&(u, v)) = lr_state.eparent.get(&v) {
                    visitor(lr_state, LRTestDfsEvent::FinishEdge(u, v))?;
                }
            }
        }
    }

    Ok(())
}

/// Check if an undirected graph is planar.
///
/// A graph is planar iff it can be drawn in a plane without any edge
/// intersections.
///
/// The planarity check algorithm  is based on the
/// Left-Right Planarity Test:
///
/// [`Ulrik Brandes:  The Left-Right Planarity Test (2009)`](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.9208)
///
/// # Example:
/// ```rust
/// use rustworkx_core::petgraph::graph::UnGraph;
/// use rustworkx_core::planar::is_planar;
///
/// let grid = UnGraph::<(), ()>::from_edges(&[
///    // row edges
///    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
///    // col edges
///    (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),
/// ]);
/// assert!(is_planar(&grid))
/// ```
pub fn is_planar<G>(graph: G) -> bool
where
    G: GraphProp<EdgeType = Undirected>
        + NodeCount
        + EdgeCount
        + IntoEdges
        + IntoNodeIdentifiers
        + Visitable,
    G::NodeId: Hash + Eq,
{
    let mut state = LRState::new(graph);

    // Dfs orientation phase
    depth_first_search(graph, graph.node_identifiers(), |event| {
        state.lr_orientation_visitor(event)
    });

    // Left - Right partition.
    for v in state.roots.clone() {
        let res = lr_visit_ordered_dfs_tree(&mut state, v, |state, event| {
            state.lr_testing_visitor(event)
        });
        if res.is_err() {
            return false;
        }
    }

    true
}
