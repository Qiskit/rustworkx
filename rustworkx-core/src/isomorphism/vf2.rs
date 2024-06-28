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

#![allow(clippy::too_many_arguments)]
// This module was originally forked from petgraph's isomorphism module @ v0.5.0
// to handle PyDiGraph inputs instead of petgraph's generic Graph. However it has
// since diverged significantly from the original petgraph implementation.

use std::cmp::{Ordering, Reverse};
use std::iter::Iterator;
use std::marker;
use std::marker::PhantomData;

use hashbrown::HashMap;
use crate::dictmap::*;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::{Data, EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges, IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph::EdgeType;
use petgraph::{Directed, Incoming, Outgoing, Undirected};
use petgraph::data::{Build, Create, DataMap};

use rayon::slice::ParallelSliceMut;


/// Returns `true` if we can map every element of `xs` to a unique
/// element of `ys` while using `matcher` func to compare two elements.
fn is_subset<T: Copy, F>(xs: &[T], ys: &[T], matcher: F) -> bool
where
    F: Fn(T, T) -> bool,
{
    let mut valid = vec![true; ys.len()];
    for &a in xs {
        let mut found = false;
        for (&b, free) in ys.iter().zip(valid.iter_mut()) {
            if *free && matcher(a, b) {
                found = true;
                *free = false;
                break;
            }
        }

        if !found {
            return false;
        }
    }

    true
}

#[inline]
fn sorted<N: std::cmp::PartialOrd>(x: &mut (N, N)) {
    let (a, b) = x;
    if b < a {
        std::mem::swap(a, b)
    }
}

/// Returns the adjacency matrix of a graph as a dictionary
/// with `(i, j)` entry equal to number of edges from node `i` to node `j`.
fn adjacency_matrix<G>(
    graph: &G,
) -> HashMap<(NodeIndex, NodeIndex), usize> where
    G: GraphProp + GraphBase<NodeId = NodeIndex> + EdgeCount + IntoEdgeReferences
{
    let mut matrix = HashMap::with_capacity(graph.edge_count());
    for edge in graph.edge_references() {
        let mut item = (edge.source(), edge.target());
        if !graph.is_directed() {
            sorted(&mut item);
        }
        let entry = matrix.entry(item).or_insert(0);
        *entry += 1;
    }
    matrix
}

/// Returns the number of edges from node `a` to node `b`.
fn edge_multiplicity<G>(
    graph: &G,
    matrix: &HashMap<(NodeIndex, NodeIndex), usize>,
    a: NodeIndex,
    b: NodeIndex,
) -> usize where
    G: GraphProp + GraphBase<NodeId = NodeIndex>
{
    let mut item = (a, b);
    if !graph.is_directed() {
        sorted(&mut item);
    }
    *matrix.get(&item).unwrap_or(&0)
}

/// Nodes `a`, `b` are adjacent if the number of edges
/// from node `a` to node `b` is greater than `val`.
fn is_adjacent<G>(
    graph: &G,
    matrix: &HashMap<(NodeIndex, NodeIndex), usize>,
    a: NodeIndex,
    b: NodeIndex,
    val: usize,
) -> bool where G: GraphProp + GraphBase<NodeId = NodeIndex> {
    edge_multiplicity(graph, matrix, a, b) >= val
}

trait NodeSorter<'a, G>
where
    G: GraphBase<NodeId = NodeIndex> + DataMap + NodeCount + EdgeCount + IntoEdgeReferences,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
{
    type OutputGraph: GraphBase<NodeId = NodeIndex> + Create + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>;

    fn sort(&self, _: &'a G) -> Vec<NodeIndex>;

    fn reorder(
        &self,
        graph: &'a G,
    ) -> (Self::OutputGraph, HashMap<usize, usize>) {
        let order = self.sort(graph);

        let mut new_graph = Self::OutputGraph::with_capacity(graph.node_count(), graph.edge_count());
        let mut id_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(graph.node_count());
        for node_index in order {
            let node_data = graph.node_weight(node_index).unwrap();
            let new_index = new_graph.add_node(node_data.clone());
            id_map.insert(node_index, new_index);
        }
        for edge in graph.edge_references() {
            let edge_w = edge.weight();
            let p_index = id_map[&edge.source()];
            let c_index = id_map[&edge.target()];
            new_graph.add_edge(p_index, c_index, edge_w.clone());
        }
        (
            new_graph,
            id_map.iter().map(|(k, v)| (v.index(), k.index())).collect(),
        )
    }
}

/// Sort nodes based on node ids.
struct DefaultIdSorter<G> {
    _phantom: PhantomData<G>,
}

impl<G> DefaultIdSorter<G> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData::default(),
        }
    }
}

impl<'a, G, GS> NodeSorter<'a, G> for DefaultIdSorter<GS>
where
    G: GraphBase<NodeId = NodeIndex> + DataMap + NodeCount + EdgeCount + IntoEdgeReferences + IntoNodeIdentifiers,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
    GS: GraphBase<NodeId = NodeIndex> + Create + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>,
{
    type OutputGraph = GS;
    fn sort(&self, graph: &'a G) -> Vec<NodeIndex> {
        graph.node_identifiers().collect()
    }
}

/// Sort nodes based on VF2++ heuristic.
struct Vf2ppSorter<G> {
    _phantom: PhantomData<G>,
}

impl<G> Vf2ppSorter<G> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData::default(),
        }
    }
}

impl<'a, G, GS> NodeSorter<'a, G> for Vf2ppSorter<GS>
where
    G: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + NodeCount + NodeIndexable + EdgeCount + IntoEdgeReferences + IntoNodeIdentifiers + IntoNeighborsDirected,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
    GS: GraphBase<NodeId = NodeIndex> + Create + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>,
{
    type OutputGraph = GS;
    fn sort(&self, graph: &'a G) -> Vec<NodeIndex> {
        let n = graph.node_bound();

        let dout: Vec<usize> = (0..n)
            .map(|idx| {
                graph
                    .neighbors_directed(graph.from_index(idx), Outgoing)
                    .count()
            })
            .collect();

        let mut din: Vec<usize> = vec![0; n];
        if graph.is_directed() {
            din = (0..n)
                .map(|idx| {
                    graph
                        .neighbors_directed(graph.from_index(idx), Incoming)
                        .count()
                })
                .collect();
        }

        let mut conn_in: Vec<usize> = vec![0; n];
        let mut conn_out: Vec<usize> = vec![0; n];

        let mut order: Vec<NodeIndex> = Vec::with_capacity(n);

        // Process BFS level
        let mut process = |mut vd: Vec<usize>| -> Vec<usize> {
            // repeatedly bring largest element in front.
            for i in 0..vd.len() {
                let (index, &item) = vd[i..]
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &node)| {
                        (
                            conn_in[node],
                            dout[node],
                            conn_out[node],
                            din[node],
                            Reverse(node),
                        )
                    })
                    .unwrap();

                vd.swap(i, i + index);
                order.push(NodeIndex::new(item));

                for neigh in graph.neighbors_directed(graph.from_index(item), Outgoing) {
                    conn_in[graph.to_index(neigh)] += 1;
                }

                if graph.is_directed() {
                    for neigh in graph.neighbors_directed(graph.from_index(item), Incoming) {
                        conn_out[graph.to_index(neigh)] += 1;
                    }
                }
            }
            vd
        };

        let mut seen: Vec<bool> = vec![false; n];

        // Create BFS Tree from root and process each level.
        let mut bfs_tree = |root: usize| {
            if seen[root] {
                return;
            }

            let mut next_level: Vec<usize> = Vec::new();

            seen[root] = true;
            next_level.push(root);
            while !next_level.is_empty() {
                let this_level = next_level;
                let this_level = process(this_level);

                next_level = Vec::new();
                for bfs_node in this_level {
                    for neighbor in graph.neighbors_directed(graph.from_index(bfs_node), Outgoing) {
                        let neigh = graph.to_index(neighbor);
                        if !seen[neigh] {
                            seen[neigh] = true;
                            next_level.push(neigh);
                        }
                    }
                }
            }
        };

        let mut sorted_nodes: Vec<usize> = graph.node_identifiers().map(|node| node.index()).collect();
        sorted_nodes.par_sort_by_key(|&node| (dout[node], din[node], Reverse(node)));
        sorted_nodes.reverse();

        for node in sorted_nodes {
            bfs_tree(node);
        }

        order
    }
}

#[derive(Debug)]
struct Vf2State<'a, G>
{
    graph: &'a G,
    /// The current mapping M(s) of nodes from G0 → G1 and G1 → G0,
    /// NodeIndex::end() for no mapping.
    mapping: Vec<NodeIndex>,
    /// out[i] is non-zero if i is in either M_0(s) or Tout_0(s)
    /// These are all the next vertices that are not mapped yet, but
    /// have an outgoing edge from the mapping.
    out: Vec<usize>,
    /// ins[i] is non-zero if i is in either M_0(s) or Tin_0(s)
    /// These are all the incoming vertices, those not mapped yet, but
    /// have an edge from them into the mapping.
    /// Unused if graph is undirected -- it's identical with out in that case.
    ins: Vec<usize>,
    out_size: usize,
    ins_size: usize,
    adjacency_matrix: HashMap<(NodeIndex, NodeIndex), usize>,
    generation: usize,
    _etype: marker::PhantomData<Directed>,
}

impl<'a, G> Vf2State<'a, G>
where
    G: GraphProp + GraphBase<NodeId = NodeIndex> + NodeCount + EdgeCount + IntoNeighborsDirected + IntoEdgeReferences,
{
    pub fn new(graph: &'a G) -> Self {
        let c0 = graph.node_count();
        let is_directed = graph.is_directed();
        let adjacency_matrix = adjacency_matrix(&graph);
        Vf2State {
            graph,
            mapping: vec![NodeIndex::end(); c0],
            out: vec![0; c0],
            ins: vec![0; c0 * (is_directed as usize)],
            out_size: 0,
            ins_size: 0,
            adjacency_matrix,
            generation: 0,
            _etype: marker::PhantomData,
        }
    }

    /// Return **true** if we have a complete mapping
    pub fn is_complete(&self) -> bool {
        self.generation == self.mapping.len()
    }

    /// Add mapping **from** <-> **to** to the state.
    pub fn push_mapping(&mut self, from: NodeIndex, to: NodeIndex) {
        self.generation += 1;
        let s = self.generation;
        self.mapping[from.index()] = to;
        // update T0 & T1 ins/outs
        // T0out: Node in G0 not in M0 but successor of a node in M0.
        // st.out[0]: Node either in M0 or successor of M0
        for ix in self.graph.neighbors(from) {
            if self.out[ix.index()] == 0 {
                self.out[ix.index()] = s;
                self.out_size += 1;
            }
        }
        if self.graph.is_directed() {
            for ix in self.graph.neighbors_directed(from, Incoming) {
                if self.ins[ix.index()] == 0 {
                    self.ins[ix.index()] = s;
                    self.ins_size += 1;
                }
            }
        }
    }

    /// Restore the state to before the last added mapping
    pub fn pop_mapping(&mut self, from: NodeIndex) {
        let s = self.generation;
        self.generation -= 1;

        // undo (n, m) mapping
        self.mapping[from.index()] = NodeIndex::end();

        // unmark in ins and outs
        for ix in self.graph.neighbors(from) {
            if self.out[ix.index()] == s {
                self.out[ix.index()] = 0;
                self.out_size -= 1;
            }
        }
        if self.graph.is_directed() {
            for ix in self.graph.neighbors_directed(from, Incoming) {
                if self.ins[ix.index()] == s {
                    self.ins[ix.index()] = 0;
                    self.ins_size -= 1;
                }
            }
        }
    }

    /// Find the next (least) node in the Tout set.
    pub fn next_out_index(&self, from_index: usize) -> Option<usize> {
        self.out[from_index..]
            .iter()
            .enumerate()
            .find(move |&(index, elt)| {
                *elt > 0 && self.mapping[from_index + index] == NodeIndex::end()
            })
            .map(|(index, _)| index)
    }

    /// Find the next (least) node in the Tin set.
    pub fn next_in_index(&self, from_index: usize) -> Option<usize> {
        self.ins[from_index..]
            .iter()
            .enumerate()
            .find(move |&(index, elt)| {
                *elt > 0 && self.mapping[from_index + index] == NodeIndex::end()
            })
            .map(|(index, _)| index)
    }

    /// Find the next (least) node in the N - M set.
    pub fn next_rest_index(&self, from_index: usize) -> Option<usize> {
        self.mapping[from_index..]
            .iter()
            .enumerate()
            .find(|&(_, elt)| *elt == NodeIndex::end())
            .map(|(index, _)| index)
    }
}

pub struct NoSemanticMatch;

pub trait NodeMatcher<G0: GraphBase, G1: GraphBase> {
    fn enabled() -> bool;
    fn eq(&mut self, _g0: &G0, _g1: &G1, _n0: G0::NodeId, _n1: G1::NodeId) -> bool;
}

impl<G0: GraphBase, G1: GraphBase> NodeMatcher<G0, G1> for NoSemanticMatch {
    #[inline]
    fn enabled() -> bool {
        false
    }
    #[inline]
    fn eq(&mut self, _g0: &G0, _g1: &G1, _n0: G0::NodeId, _n1: G1::NodeId) -> bool {
        true
    }
}

impl<G0, G1, F> NodeMatcher<G0, G1> for F
    where
        G0: GraphBase + DataMap,
        G1: GraphBase + DataMap,
        F: FnMut(&G0::NodeWeight, &G1::NodeWeight) -> bool,
{
    #[inline]
    fn enabled() -> bool {
        true
    }
    #[inline]
    fn eq(&mut self, g0: &G0, g1: &G1, n0: G0::NodeId, n1: G1::NodeId) -> bool {
        if let (Some(x), Some(y)) = (g0.node_weight(n0), g1.node_weight(n1)) {
            self(x, y)
        } else {
            false
        }
    }
}

pub trait EdgeMatcher<G0: GraphBase, G1: GraphBase> {
    fn enabled() -> bool;
    fn eq(
        &mut self,
        _g0: &G0,
        _g1: &G1,
        e0: (G0::NodeId, G0::NodeId),
        e1: (G1::NodeId, G1::NodeId),
    ) -> bool;
}

impl<G0: GraphBase, G1: GraphBase> EdgeMatcher<G0, G1> for NoSemanticMatch {
    #[inline]
    fn enabled() -> bool {
        false
    }
    #[inline]
    fn eq(
        &mut self,
        _g0: &G0,
        _g1: &G1,
        _e0: (G0::NodeId, G0::NodeId),
        _e1: (G1::NodeId, G1::NodeId),
    ) -> bool {
        true
    }
}

impl<G0, G1, F> EdgeMatcher<G0, G1> for F
    where
        G0: GraphBase + DataMap + IntoEdgesDirected,
        G1: GraphBase + DataMap + IntoEdgesDirected,
        F: FnMut(&G0::EdgeWeight, &G1::EdgeWeight) -> bool,
{
    #[inline]
    fn enabled() -> bool {
        true
    }
    #[inline]
    fn eq(
        &mut self,
        g0: &G0,
        g1: &G1,
        e0: (G0::NodeId, G0::NodeId),
        e1: (G1::NodeId, G1::NodeId),
    ) -> bool {
        let w0 = g0
            .edges_directed(e0.0, Outgoing)
            .find(|edge| edge.target() == e0.1)
            .and_then(|edge| g0.edge_weight(edge.id()));
        let w1 = g1
            .edges_directed(e1.0, Outgoing)
            .find(|edge| edge.target() == e1.1)
            .and_then(|edge| g1.edge_weight(edge.id()));
        if let (Some(x), Some(y)) = (w0, w1) {
            self(x, y)
        } else {
            false
        }
    }
}

/// [Graph] Return `true` if the graphs `g0` and `g1` are (sub) graph isomorphic.
///
/// Using the VF2 algorithm, examining both syntactic and semantic
/// graph isomorphism (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
pub fn is_isomorphic<'a, G0, G1, NM, EM>(
    g0: &'a G0,
    g1: &'a G1,
    node_match: NM,
    edge_match: EM,
    id_order: bool,
    ordering: Ordering,
    induced: bool,
    call_limit: Option<usize>,
) -> bool
    where
        G0: GraphProp + GraphBase<NodeId = NodeIndex> + Data + NodeCount + EdgeCount + IntoNeighborsDirected + IntoEdgeReferences,
        G1: GraphProp + GraphBase<NodeId = NodeIndex> + Data + NodeCount + EdgeCount + IntoNeighborsDirected + IntoEdgeReferences,
        NM: NodeMatcher<G0, G1>,
        EM: EdgeMatcher<G0, G1>,
{
    if (g0.node_count().cmp(&g1.node_count()).then(ordering) != ordering)
        || (g0.edge_count().cmp(&g1.edge_count()).then(ordering) != ordering)
    {
        return false;
    }

    let mut vf2 = Vf2Algorithm::new(
        g0, g1, node_match, edge_match, id_order, ordering, induced, call_limit,
    );
    if vf2.next().is_some() {
        return true;
    }
    false
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum OpenList {
    Out,
    In,
    Other,
}

#[derive(Clone, PartialEq, Debug)]
enum Frame<N: marker::Copy> {
    Outer,
    Inner { nodes: [N; 2], open_list: OpenList },
    Unwind { nodes: [N; 2], open_list: OpenList },
}

struct Vf2Algorithm<'a, G0, G1, NM, EM>
where
    G0: GraphBase,
    G1: GraphBase,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    st: (Vf2State<'a, G0>, Vf2State<'a, G1>),
    node_match: NM,
    edge_match: EM,
    ordering: Ordering,
    induced: bool,
    node_map_g0: HashMap<usize, usize>,
    node_map_g1: HashMap<usize, usize>,
    stack: Vec<Frame<NodeIndex>>,
    call_limit: Option<usize>,
    _counter: usize,
}

impl<'a, G0, G1, NM, EM> Vf2Algorithm<'a, G0, G1, NM, EM>
where
    G0: GraphProp + GraphBase<NodeId = NodeIndex> + Data + NodeCount + EdgeCount + IntoNeighborsDirected + IntoEdgeReferences,
    G1: GraphProp + GraphBase<NodeId = NodeIndex> + Data + NodeCount + EdgeCount + IntoNeighborsDirected + IntoEdgeReferences,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    pub fn new(
        g0: &'a G0,
        g1: &'a G1,
        node_match: NM,
        edge_match: EM,
        id_order: bool,
        ordering: Ordering,
        induced: bool,
        call_limit: Option<usize>,
    ) -> Self {
        let (g0, node_map_g0) = if id_order {
            DefaultIdSorter::<G0>::new().reorder(g0)
        } else {
            Vf2ppSorter::<G0>::new().reorder(g0)
        };

        let (g1, node_map_g1) = if id_order {
            DefaultIdSorter::<G1>::new().reorder(g1)
        } else {
            Vf2ppSorter::<G1>::new().reorder(g1)
        };

        let st = (Vf2State::new(g0), Vf2State::new(g1));
        Vf2Algorithm {
            st,
            node_match,
            edge_match,
            ordering,
            induced,
            node_map_g0,
            node_map_g1,
            stack: vec![Frame::Outer],
            call_limit,
            _counter: 0,
        }
    }

    fn mapping(&self) -> DictMap<usize, usize> {
        let mut mapping: DictMap<usize, usize> = DictMap::new();
        self.st.1
            .mapping
            .iter()
            .enumerate()
            .for_each(|(index, val)| {
                mapping.insert(self.node_map_g0[&val.index()], self.node_map_g1[&index]);
            });

        mapping
    }

    fn next_candidate(st: &mut (Vf2State<'a, G0>, Vf2State<'a, G1>)) -> Option<(NodeIndex, NodeIndex, OpenList)> {
        // Try the out list
        let mut to_index = st.1.next_out_index(0);
        let mut from_index = None;
        let mut open_list = OpenList::Out;

        if to_index.is_some() {
            from_index = st.0.next_out_index(0);
            open_list = OpenList::Out;
        }
        // Try the in list
        if to_index.is_none() || from_index.is_none() {
            to_index = st.1.next_in_index(0);

            if to_index.is_some() {
                from_index = st.0.next_in_index(0);
                open_list = OpenList::In;
            }
        }
        // Try the other list -- disconnected graph
        if to_index.is_none() || from_index.is_none() {
            to_index = st.1.next_rest_index(0);
            if to_index.is_some() {
                from_index = st.0.next_rest_index(0);
                open_list = OpenList::Other;
            }
        }
        match (from_index, to_index) {
            (Some(n), Some(m)) => Some((NodeIndex::new(n), NodeIndex::new(m), open_list)),
            // No more candidates
            _ => None,
        }
    }

    fn next_from_ix(
        st: &mut (Vf2State<'a, G0>, Vf2State<'a, G1>),
        nx: NodeIndex,
        open_list: OpenList,
    ) -> Option<NodeIndex> {
        // Find the next node index to try on the `from` side of the mapping
        let start = nx.index() + 1;
        let cand0 = match open_list {
            OpenList::Out => st.0.next_out_index(start),
            OpenList::In => st.0.next_in_index(start),
            OpenList::Other => st.0.next_rest_index(start),
        }
        .map(|c| c + start); // compensate for start offset.
        match cand0 {
            None => None, // no more candidates
            Some(ix) => {
                debug_assert!(ix >= start);
                Some(NodeIndex::new(ix))
            }
        }
    }

    fn pop_state(st: &mut (Vf2State<'a, G0>, Vf2State<'a, G1>), nodes: [NodeIndex; 2]) {
        // Restore state.
        st.0.pop_mapping(nodes[0]);
        st.1.pop_mapping(nodes[1]);
    }

    fn push_state(st: &mut (Vf2State<'a, G0>, Vf2State<'a, G1>), nodes: [NodeIndex; 2]) {
        // Add mapping nx <-> mx to the state
        st.0.push_mapping(nodes[0], nodes[1]);
        st.1.push_mapping(nodes[1], nodes[0]);
    }

    fn is_feasible(
        st: &mut (Vf2State<'a, G0>, Vf2State<'a, G1>),
        nodes: [NodeIndex; 2],
        node_match: &mut NM,
        edge_match: &mut EM,
        ordering: Ordering,
        induced: bool,
    ) -> bool {
        // Check syntactic feasibility of mapping by ensuring adjacencies
        // of nx map to adjacencies of mx.
        //
        // nx == map to => mx
        //
        // R_succ
        //
        // Check that every neighbor of nx is mapped to a neighbor of mx,
        // then check the reverse, from mx to nx. Check that they have the same
        // count of edges.
        //
        // Note: We want to check the lookahead measures here if we can,
        // R_out: Equal for G0, G1: Card(Succ(G, n) ^ Tout); for both Succ and Pred
        // R_in: Same with Tin
        // R_new: Equal for G0, G1: Ñ n Pred(G, n); both Succ and Pred,
        //      Ñ is G0 - M - Tin - Tout
        let end = NodeIndex::end();
        let mut succ_count = [0, 0];
        for n_neigh in st.0.graph.neighbors(nodes[0]) {
            succ_count[0] += 1;
            if !induced && 0 == 0 {
                continue;
            }
            // handle the self loop case; it's not in the mapping (yet)
            let m_neigh = if nodes[0] != n_neigh {
                st.0.mapping[n_neigh.index()]
            } else {
                nodes[1]
            };
            if m_neigh == end {
                continue;
            }
            let val =
                edge_multiplicity(&st.0.graph, &st.0.adjacency_matrix, nodes[0], n_neigh);

            let has_edge = is_adjacent(
                &st.1.graph,
                &st.1.adjacency_matrix,
                nodes[1],
                m_neigh,
                val,
            );
            if !has_edge {
                return false;
            }
        }

        for n_neigh in st.1.graph.neighbors(nodes[1]) {
            succ_count[1] += 1;
            if !induced && 1 == 0 {
                continue;
            }
            // handle the self loop case; it's not in the mapping (yet)
            let m_neigh = if nodes[1] != n_neigh {
                st.1.mapping[n_neigh.index()]
            } else {
                nodes[0]
            };
            if m_neigh == end {
                continue;
            }
            let val =
                edge_multiplicity(&st.1.graph, &st.1.adjacency_matrix, nodes[1], n_neigh);

            let has_edge = is_adjacent(
                &st.0.graph,
                &st.0.adjacency_matrix,
                nodes[0],
                m_neigh,
                val,
            );
            if !has_edge {
                return false;
            }
        }
        if succ_count[0].cmp(&succ_count[1]).then(ordering) != ordering {
            return false;
        }
        // R_pred
        if st.0.graph.is_directed() {
            let mut pred_count = [0, 0];
            for n_neigh in st.0.graph.neighbors_directed(nodes[0], Incoming) {
                pred_count[0] += 1;
                if !induced && 0 == 0 {
                    continue;
                }
                // the self loop case is handled in outgoing
                let m_neigh = st.0.mapping[n_neigh.index()];
                if m_neigh == end {
                    continue;
                }
                let val =
                    edge_multiplicity(&st.0.graph, &st.0.adjacency_matrix, n_neigh, nodes[0]);

                let has_edge = is_adjacent(
                    &st.1.graph,
                    &st.1.adjacency_matrix,
                    m_neigh,
                    nodes[1],
                    val,
                );
                if !has_edge {
                    return false;
                }
            }

            for n_neigh in st.1.graph.neighbors_directed(nodes[1], Incoming) {
                pred_count[1] += 1;
                if !induced && 1 == 0 {
                    continue;
                }
                // the self loop case is handled in outgoing
                let m_neigh = st.1.mapping[n_neigh.index()];
                if m_neigh == end {
                    continue;
                }
                let val =
                    edge_multiplicity(&st.1.graph, &st.1.adjacency_matrix, n_neigh, nodes[1]);

                let has_edge = is_adjacent(
                    &st.0.graph,
                    &st.0.adjacency_matrix,
                    m_neigh,
                    nodes[0],
                    val,
                );
                if !has_edge {
                    return false;
                }
            }
            if pred_count[0].cmp(&pred_count[1]).then(ordering) != ordering {
                return false;
            }
        }
        macro_rules! field {
            ($x:ident,     0) => {
                $x.0
            };
            ($x:ident,     1) => {
                $x.1
            };
            ($x:ident, 1 - 0) => {
                $x.1
            };
            ($x:ident, 1 - 1) => {
                $x.0
            };
        }
        macro_rules! rule {
            ($arr:ident, $j:tt, $dir:expr) => {{
                let mut count = 0;
                for n_neigh in field!(st, $j).graph.neighbors_directed(nodes[$j], $dir) {
                    let index = n_neigh.index();
                    if field!(st, $j).$arr[index] > 0 && st.$j.mapping[index] == end {
                        count += 1;
                    }
                }
                count
            }};
        }
        // R_out
        if rule!(out, 0, Outgoing)
            .cmp(&rule!(out, 1, Outgoing))
            .then(ordering)
            != ordering
        {
            return false;
        }
        if st.0.graph.is_directed()
            && rule!(out, 0, Incoming)
                .cmp(&rule!(out, 1, Incoming))
                .then(ordering)
                != ordering
        {
            return false;
        }
        // R_in
        if st.0.graph.is_directed() {
            if rule!(ins, 0, Outgoing)
                .cmp(&rule!(ins, 1, Outgoing))
                .then(ordering)
                != ordering
            {
                return false;
            }

            if rule!(ins, 0, Incoming)
                .cmp(&rule!(ins, 1, Incoming))
                .then(ordering)
                != ordering
            {
                return false;
            }
        }
        // R_new
        if induced {
            let mut new_count = [0, 0];
            for n_neigh in st.0.graph.neighbors(nodes[0]) {
                let index = n_neigh.index();
                if st.0.out[index] == 0 && (st.0.ins.is_empty() || st.0.ins[index] == 0) {
                    new_count[0] += 1;
                }
            }
            for n_neigh in st.1.graph.neighbors(nodes[1]) {
                let index = n_neigh.index();
                if st.1.out[index] == 0 && (st.1.ins.is_empty() || st.1.ins[index] == 0) {
                    new_count[1] += 1;
                }
            }
            if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
                return false;
            }
            if st.0.graph.is_directed() {
                let mut new_count = [0, 0];
                for n_neigh in st.0.graph.neighbors_directed(nodes[0], Incoming) {
                    let index = n_neigh.index();
                    if st.0.out[index] == 0 && st.0.ins[index] == 0 {
                        new_count[0] += 1;
                    }
                }
                for n_neigh in st.1.graph.neighbors_directed(nodes[1], Incoming) {
                    let index = n_neigh.index();
                    if st.1.out[index] == 0 && st.1.ins[index] == 0 {
                        new_count[1] += 1;
                    }
                }
                if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
                    return false;
                }
            }
        }
        // semantic feasibility: compare associated data for nodes
        if node_match.enabled()
            && !node_match.eq(st.0.graph, st.1.graph, nodes[0], nodes[1])
        {
            return false;
        }
        // semantic feasibility: compare associated data for edges
        if edge_match.enabled() {
            let matcher =
                |a: (NodeIndex, (NodeIndex, NodeIndex)), b: (NodeIndex, (NodeIndex, NodeIndex))| -> bool {
                    let (nx, n_edge) = a;
                    let (mx, m_edge) = b;
                    if nx == mx && edge_match.eq(st.0.graph, st.1.graph, n_edge, m_edge)? {
                        return true;
                    }
                    false
                };

            // outgoing edges
            if induced {
                let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.0
                    .graph
                    .edges(nodes[0])
                    .filter_map(|edge| {
                        let n_neigh = edge.target();
                        let m_neigh = if nodes[0] != n_neigh {
                            st.0.mapping[n_neigh.index()]
                        } else {
                            nodes[1]
                        };
                        if m_neigh == end {
                            return None;
                        }
                        Some((m_neigh, (edge.source(), edge.target())))
                    })
                    .collect();

                let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                    .graph
                    .edges(nodes[1])
                    .map(|edge| (edge.target(), (edge.source(), edge.target())))
                    .collect();

                if !is_subset(&e_first, &e_second, matcher)? {
                    return false;
                };

                let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                    .graph
                    .edges(nodes[1])
                    .filter_map(|edge| {
                        let n_neigh = edge.target();
                        let m_neigh = if nodes[1] != n_neigh {
                            st.1.mapping[n_neigh.index()]
                        } else {
                            nodes[0]
                        };
                        if m_neigh == end {
                            return None;
                        }
                        Some((m_neigh, (edge.source(), edge.target())))
                    })
                    .collect();

                let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.0
                    .graph
                    .edges(nodes[0])
                    .map(|edge| (edge.target(), (edge.source(), edge.target())))
                    .collect();

                if !is_subset(&e_first, &e_second, matcher)? {
                    return false;
                };
            } else {
                let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                    .graph
                    .edges(nodes[1])
                    .filter_map(|edge| {
                        let n_neigh = edge.target();
                        let m_neigh = if nodes[1] != n_neigh {
                            st.1.mapping[n_neigh.index()]
                        } else {
                            nodes[0]
                        };
                        if m_neigh == end {
                            return None;
                        }
                        Some((m_neigh, (edge.source(), edge.target())))
                    })
                    .collect();

                let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.0
                    .graph
                    .edges(nodes[0])
                    .map(|edge| (edge.target(), (edge.source(), edge.target())))
                    .collect();

                if !is_subset(&e_first, &e_second, matcher)? {
                    return false;
                };
            }

            // incoming edges
            if st.0.graph.is_directed() {
                if induced {
                    let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.0
                        .graph
                        .edges_directed(nodes[0], Incoming)
                        .filter_map(|edge| {
                            let n_neigh = edge.source();
                            let m_neigh = if nodes[0] != n_neigh {
                                st.0.mapping[n_neigh.index()]
                            } else {
                                nodes[1]
                            };
                            if m_neigh == end {
                                return None;
                            }
                            Some((m_neigh, (edge.source(), edge.target())))
                        })
                        .collect();

                    let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                        .graph
                        .edges_directed(nodes[1], Incoming)
                        .map(|edge| (edge.source(), (edge.source(), edge.target())))
                        .collect();

                    if !is_subset(&e_first, &e_second, matcher)? {
                        return false;
                    };

                    let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                        .graph
                        .edges_directed(nodes[1], Incoming)
                        .filter_map(|edge| {
                            let n_neigh = edge.source();
                            let m_neigh = if nodes[1] != n_neigh {
                                st.1.mapping[n_neigh.index()]
                            } else {
                                nodes[0]
                            };
                            if m_neigh == end {
                                return None;
                            }
                            Some((m_neigh, (edge.source(), edge.target())))
                        })
                        .collect();

                    let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                        .graph
                        .edges_directed(nodes[0], Incoming)
                        .map(|edge| (edge.source(), (edge.source(), edge.target())))
                        .collect();

                    if !is_subset(&e_first, &e_second, matcher)? {
                        return false;
                    };
                } else {
                    let e_first: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.1
                        .graph
                        .edges_directed(nodes[1], Incoming)
                        .filter_map(|edge| {
                            let n_neigh = edge.source();
                            let m_neigh = if nodes[1] != n_neigh {
                                st.1.mapping[n_neigh.index()]
                            } else {
                                nodes[0]
                            };
                            if m_neigh == end {
                                return None;
                            }
                            Some((m_neigh, (edge.source(), edge.target())))
                        })
                        .collect();

                    let e_second: Vec<(NodeIndex, (NodeIndex, NodeIndex))> = st.0
                        .graph
                        .edges_directed(nodes[0], Incoming)
                        .map(|edge| (edge.source(), (edge.source(), edge.target())))
                        .collect();

                    if !is_subset(&e_first, &e_second, matcher)? {
                        return false;
                    };
                }
            }
        }
        true
    }

    /// Return Some(mapping) if isomorphism is decided, else None.
    fn next(&mut self) -> Option<DictMap<usize, usize>> {
        if (self.st.0
            .graph
            .node_count()
            .cmp(&self.st.1.graph.node_count())
            .then(self.ordering)
            != self.ordering)
            || (self.st.0
                .graph
                .edge_count()
                .cmp(&self.st.1.graph.edge_count())
                .then(self.ordering)
                != self.ordering)
        {
            return None;
        }

        // A "depth first" search of a valid mapping from graph 1 to graph 2

        // F(s, n, m) -- evaluate state s and add mapping n <-> m

        // Find least T1out node (in st.out[1] but not in M[1])
        while let Some(frame) = self.stack.pop() {
            match frame {
                Frame::Unwind {
                    nodes,
                    open_list: ol,
                } => {
                    Vf2Algorithm::<'a, G0, G1, NM, EM>::pop_state(&mut self.st, nodes);

                    match Vf2Algorithm::<'a, G0, G1, NM, EM>::next_from_ix(&mut self.st, nodes[0], ol) {
                        None => continue,
                        Some(nx) => {
                            let f = Frame::Inner {
                                nodes: [nx, nodes[1]],
                                open_list: ol,
                            };
                            self.stack.push(f);
                        }
                    }
                }
                Frame::Outer => match Vf2Algorithm::<'a, G0, G1, NM, EM>::next_candidate(&mut self.st) {
                    None => {
                        if self.st.1.is_complete() {
                            return Some(self.mapping());
                        }
                        continue;
                    }
                    Some((nx, mx, ol)) => {
                        let f = Frame::Inner {
                            nodes: [nx, mx],
                            open_list: ol,
                        };
                        self.stack.push(f);
                    }
                },
                Frame::Inner {
                    nodes,
                    open_list: ol,
                } => {
                    if Vf2Algorithm::<'a, G0, G1, NM, EM>::is_feasible(
                        &mut self.st,
                        nodes,
                        &mut self.node_match,
                        &mut self.edge_match,
                        self.ordering,
                        self.induced,
                    ) {
                        Vf2Algorithm::<'a, G0, G1, NM, EM>::push_state(&mut self.st, nodes);
                        // Check cardinalities of Tin, Tout sets
                        if self.st.0
                            .out_size
                            .cmp(&self.st.1.out_size)
                            .then(self.ordering)
                            == self.ordering
                            && self.st.0
                                .ins_size
                                .cmp(&self.st.1.ins_size)
                                .then(self.ordering)
                                == self.ordering
                        {
                            self._counter += 1;
                            if let Some(limit) = self.call_limit {
                                if self._counter > limit {
                                    return None;
                                }
                            }
                            let f0 = Frame::Unwind {
                                nodes,
                                open_list: ol,
                            };

                            self.stack.push(f0);
                            self.stack.push(Frame::Outer);
                            continue;
                        }
                        Vf2Algorithm::<'a, G0, G1, NM, EM>::pop_state(&mut self.st, nodes);
                    }
                    match Vf2Algorithm::<'a, G0, G1, NM, EM>::next_from_ix(&mut self.st, nodes[0], ol) {
                        None => continue,
                        Some(nx) => {
                            let f = Frame::Inner {
                                nodes: [nx, nodes[1]],
                                open_list: ol,
                            };
                            self.stack.push(f);
                        }
                    }
                }
            }
        }
        None
    }
}
