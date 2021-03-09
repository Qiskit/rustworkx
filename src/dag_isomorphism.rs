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

// This module is a forked version of petgraph's isomorphism module @ 0.5.0.
// It has then been modified to function with PyDiGraph inputs instead of Graph.

use fixedbitset::FixedBitSet;
use std::marker;

use hashbrown::HashMap;

use super::digraph::PyDiGraph;

use pyo3::prelude::*;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::GetAdjacencyMatrix;
use petgraph::{Directed, Incoming};

#[derive(Debug)]
struct Vf2State {
    /// The current mapping M(s) of nodes from G0 → G1 and G1 → G0,
    /// NodeIndex::end() for no mapping.
    mapping: HashMap<NodeIndex, NodeIndex>,
    /// out[i] is non-zero if i is in either M_0(s) or Tout_0(s)
    /// These are all the next vertices that are not mapped yet, but
    /// have an outgoing edge from the mapping.
    out: HashMap<NodeIndex, usize>,
    /// ins[i] is non-zero if i is in either M_0(s) or Tin_0(s)
    /// These are all the incoming vertices, those not mapped yet, but
    /// have an edge from them into the mapping.
    /// Unused if graph is undirected -- it's identical with out in that case.
    ins: HashMap<NodeIndex, usize>,
    out_size: usize,
    ins_size: usize,
    adjacency_matrix: FixedBitSet,
    generation: usize,
    _etype: marker::PhantomData<Directed>,
}

impl Vf2State {
    pub fn new(dag: &PyDiGraph) -> Self {
        let g = &dag.graph;
        let c0 = g.node_count();
        let mut state = Vf2State {
            mapping: HashMap::with_capacity(c0),
            out: HashMap::with_capacity(c0),
            ins: HashMap::with_capacity(c0 * (g.is_directed() as usize)),
            out_size: 0,
            ins_size: 0,
            adjacency_matrix: g.adjacency_matrix(),
            generation: 0,
            _etype: marker::PhantomData,
        };
        for index in g.node_indices() {
            state.mapping.insert(index, NodeIndex::end());
            state.out.insert(index, 0);
            state.ins.insert(index, 0);
        }
        state
    }

    /// Return **true** if we have a complete mapping
    pub fn is_complete(&self) -> bool {
        self.generation == self.mapping.len()
    }

    /// Add mapping **from** <-> **to** to the state.
    pub fn push_mapping(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        dag: &PyDiGraph,
    ) {
        let g = &dag.graph;
        self.generation += 1;
        let s = self.generation;
        self.mapping.insert(from, to);
        // update T0 & T1 ins/outs
        // T0out: Node in G0 not in M0 but successor of a node in M0.
        // st.out[0]: Node either in M0 or successor of M0
        for ix in g.neighbors(from) {
            if self.out[&ix] == 0 {
                self.out.insert(ix, s);
                self.out_size += 1;
            }
        }
        if g.is_directed() {
            for ix in g.neighbors_directed(from, Incoming) {
                if self.ins[&ix] == 0 {
                    self.ins.insert(ix, s);
                    self.ins_size += 1;
                }
            }
        }
    }

    /// Restore the state to before the last added mapping
    pub fn pop_mapping(&mut self, from: NodeIndex, dag: &PyDiGraph) {
        let g = &dag.graph;
        let s = self.generation;
        self.generation -= 1;

        // undo (n, m) mapping
        self.mapping.insert(from, NodeIndex::end());

        // unmark in ins and outs
        for ix in g.neighbors(from) {
            if self.out[&ix] == s {
                self.out.insert(ix, 0);
                self.out_size -= 1;
            }
        }
        if g.is_directed() {
            for ix in g.neighbors_directed(from, Incoming) {
                if self.ins[&ix] == s {
                    self.ins.insert(ix, 0);
                    self.ins_size -= 1;
                }
            }
        }
    }

    /// Find the next (least) node in the Tout set.
    pub fn next_out_index(&self, from_index: usize) -> Option<usize> {
        self.out
            .iter()
            .filter(|&(node, elt)| {
                (*node).index() >= from_index
                    && *elt > 0
                    && self.mapping[node] == NodeIndex::end()
            })
            .min_by_key(|&(&node, _)| node.index())
            .map(|(&node, _)| node.index())
    }

    /// Find the next (least) node in the Tin set.
    pub fn next_in_index(&self, from_index: usize) -> Option<usize> {
        self.ins
            .iter()
            .filter(|&(node, elt)| {
                (*node).index() >= from_index
                    && *elt > 0
                    && self.mapping[node] == NodeIndex::end()
            })
            .min_by_key(|&(&node, _)| node.index())
            .map(|(&node, _)| node.index())
    }

    /// Find the next (least) node in the N - M set.
    pub fn next_rest_index(&self, from_index: usize) -> Option<usize> {
        self.mapping
            .iter()
            .filter(|&(&node, elt)| {
                node.index() >= from_index && *elt == NodeIndex::end()
            })
            .min_by_key(|&(&node, _)| node.index())
            .map(|(&node, _)| node.index())
    }
}

/// [Graph] Return `true` if the graphs `g0` and `g1` are isomorphic.
///
/// Using the VF2 algorithm, only matching graph syntactically (graph
/// structure).
///
/// The graphs should not be multigraphs.
///
/// **Reference**
///
/// * Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, Mario Vento;
///   *A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs*
pub fn is_isomorphic(dag0: &PyDiGraph, dag1: &PyDiGraph) -> PyResult<bool> {
    let g0 = &dag0.graph;
    let g1 = &dag1.graph;
    if g0.node_count() != g1.node_count() || g0.edge_count() != g1.edge_count()
    {
        return Ok(false);
    }

    let mut st = [Vf2State::new(dag0), Vf2State::new(dag1)];
    let res = try_match(
        &mut st,
        dag0,
        dag1,
        &mut NoSemanticMatch,
        &mut NoSemanticMatch,
    )?;
    Ok(res.unwrap_or(false))
}

/// [Graph] Return `true` if the graphs `g0` and `g1` are isomorphic.
///
/// Using the VF2 algorithm, examining both syntactic and semantic
/// graph isomorphism (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
pub fn is_isomorphic_matching<F, G>(
    dag0: &PyDiGraph,
    dag1: &PyDiGraph,
    mut node_match: F,
    mut edge_match: G,
) -> PyResult<bool>
where
    F: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
    G: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
{
    let g0 = &dag0.graph;
    let g1 = &dag1.graph;
    if g0.node_count() != g1.node_count() || g0.edge_count() != g1.edge_count()
    {
        return Ok(false);
    }

    let mut st = [Vf2State::new(dag0), Vf2State::new(dag1)];
    let res = try_match(&mut st, dag0, dag1, &mut node_match, &mut edge_match)?;
    Ok(res.unwrap_or(false))
}

trait SemanticMatcher<T> {
    fn enabled() -> bool;
    fn eq(&mut self, _: &T, _: &T) -> PyResult<bool>;
}

struct NoSemanticMatch;

impl<T> SemanticMatcher<T> for NoSemanticMatch {
    #[inline]
    fn enabled() -> bool {
        false
    }
    #[inline]
    fn eq(&mut self, _: &T, _: &T) -> PyResult<bool> {
        Ok(true)
    }
}

impl<T, F> SemanticMatcher<T> for F
where
    F: FnMut(&T, &T) -> PyResult<bool>,
{
    #[inline]
    fn enabled() -> bool {
        true
    }
    #[inline]
    fn eq(&mut self, a: &T, b: &T) -> PyResult<bool> {
        let res = self(a, b)?;
        Ok(res)
    }
}

/// Return Some(bool) if isomorphism is decided, else None.
fn try_match<F, G>(
    mut st: &mut [Vf2State; 2],
    dag0: &PyDiGraph,
    dag1: &PyDiGraph,
    node_match: &mut F,
    edge_match: &mut G,
) -> PyResult<Option<bool>>
where
    F: SemanticMatcher<PyObject>,
    G: SemanticMatcher<PyObject>,
{
    let g0 = &dag0.graph;
    let g1 = &dag1.graph;
    if st[0].is_complete() {
        return Ok(Some(true));
    }
    let dag = [dag0, dag1];
    let g = [g0, g1];
    let graph_indices = 0..2;
    let end = NodeIndex::end();

    // A "depth first" search of a valid mapping from graph 1 to graph 2

    // F(s, n, m) -- evaluate state s and add mapping n <-> m

    // Find least T1out node (in st.out[1] but not in M[1])
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

    let next_candidate =
        |st: &mut [Vf2State; 2]| -> Option<(NodeIndex, NodeIndex, OpenList)> {
            let mut to_index;
            let mut from_index = None;
            let mut open_list = OpenList::Out;
            // Try the out list
            to_index = st[1].next_out_index(0);

            if to_index.is_some() {
                from_index = st[0].next_out_index(0);
                open_list = OpenList::Out;
            }
            // Try the in list
            if to_index.is_none() || from_index.is_none() {
                to_index = st[1].next_in_index(0);

                if to_index.is_some() {
                    from_index = st[0].next_in_index(0);
                    open_list = OpenList::In;
                }
            }
            // Try the other list -- disconnected graph
            if to_index.is_none() || from_index.is_none() {
                to_index = st[1].next_rest_index(0);
                if to_index.is_some() {
                    from_index = st[0].next_rest_index(0);
                    open_list = OpenList::Other;
                }
            }
            match (from_index, to_index) {
                (Some(n), Some(m)) => {
                    Some((NodeIndex::new(n), NodeIndex::new(m), open_list))
                }
                // No more candidates
                _ => None,
            }
        };
    let next_from_ix = |st: &mut [Vf2State; 2],
                        nx: NodeIndex,
                        open_list: OpenList|
     -> Option<NodeIndex> {
        // Find the next node index to try on the `from` side of the mapping
        let start = nx.index() + 1;
        let cand0 = match open_list {
            OpenList::Out => st[0].next_out_index(start),
            OpenList::In => st[0].next_in_index(start),
            OpenList::Other => st[0].next_rest_index(start),
        };
        match cand0 {
            None => None, // no more candidates
            Some(ix) => {
                debug_assert!(ix >= start);
                Some(NodeIndex::new(ix))
            }
        }
    };
    //fn pop_state(nodes: [NodeIndex<Ix>; 2]) {
    let pop_state = |st: &mut [Vf2State; 2], nodes: [NodeIndex; 2]| {
        // Restore state.
        for j in graph_indices.clone() {
            st[j].pop_mapping(nodes[j], dag[j]);
        }
    };
    //fn push_state(nodes: [NodeIndex<Ix>; 2]) {
    let push_state = |st: &mut [Vf2State; 2], nodes: [NodeIndex; 2]| {
        // Add mapping nx <-> mx to the state
        for j in graph_indices.clone() {
            st[j].push_mapping(nodes[j], nodes[1 - j], dag[j]);
        }
    };
    //fn is_feasible(nodes: [NodeIndex<Ix>; 2]) -> bool {
    let mut is_feasible = |st: &mut [Vf2State; 2],
                           nodes: [NodeIndex; 2]|
     -> PyResult<bool> {
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
        // last attempt to add these did not speed up any of the testcases
        let mut succ_count = [0, 0];
        for j in graph_indices.clone() {
            for n_neigh in g[j].neighbors(nodes[j]) {
                succ_count[j] += 1;
                // handle the self loop case; it's not in the mapping (yet)
                let m_neigh = if nodes[j] != n_neigh {
                    st[j].mapping[&n_neigh]
                } else {
                    nodes[1 - j]
                };
                if m_neigh == end {
                    continue;
                }
                let has_edge = dag[1 - j].is_adjacent(
                    &st[1 - j].adjacency_matrix,
                    nodes[1 - j],
                    m_neigh,
                );
                if !has_edge {
                    return Ok(false);
                }
            }
        }
        if succ_count[0] != succ_count[1] {
            return Ok(false);
        }
        // R_pred
        if g[0].is_directed() {
            let mut pred_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors_directed(nodes[j], Incoming) {
                    pred_count[j] += 1;
                    // the self loop case is handled in outgoing
                    let m_neigh = st[j].mapping[&n_neigh];
                    if m_neigh == end {
                        continue;
                    }
                    let has_edge = dag[1 - j].is_adjacent(
                        &st[1 - j].adjacency_matrix,
                        m_neigh,
                        nodes[1 - j],
                    );
                    if !has_edge {
                        return Ok(false);
                    }
                }
            }
            if pred_count[0] != pred_count[1] {
                return Ok(false);
            }
        }
        // semantic feasibility: compare associated data for nodes
        let match_result = node_match.eq(&g[0][nodes[0]], &g[1][nodes[1]])?;
        if F::enabled() && !match_result {
            return Ok(false);
        }
        // semantic feasibility: compare associated data for edges
        if G::enabled() {
            // outgoing edges
            for j in graph_indices.clone() {
                let mut edges = g[j].neighbors(nodes[j]).detach();
                while let Some((n_edge, n_neigh)) = edges.next(g[j]) {
                    // handle the self loop case; it's not in the mapping (yet)
                    let m_neigh = if nodes[j] != n_neigh {
                        st[j].mapping[&n_neigh]
                    } else {
                        nodes[1 - j]
                    };
                    if m_neigh == end {
                        continue;
                    }
                    match g[1 - j].find_edge(nodes[1 - j], m_neigh) {
                        Some(m_edge) => {
                            let match_result = edge_match
                                .eq(&g[j][n_edge], &g[1 - j][m_edge])?;
                            if !match_result {
                                return Ok(false);
                            }
                        }
                        None => unreachable!(), // covered by syntactic check
                    }
                }
            }
            // incoming edges
            if g[0].is_directed() {
                for j in graph_indices.clone() {
                    let mut edges =
                        g[j].neighbors_directed(nodes[j], Incoming).detach();
                    while let Some((n_edge, n_neigh)) = edges.next(g[j]) {
                        // the self loop case is handled in outgoing
                        let m_neigh = st[j].mapping[&n_neigh];
                        if m_neigh == end {
                            continue;
                        }
                        match g[1 - j].find_edge(m_neigh, nodes[1 - j]) {
                            Some(m_edge) => {
                                let match_result = edge_match
                                    .eq(&g[j][n_edge], &g[1 - j][m_edge])?;
                                if !match_result {
                                    return Ok(false);
                                }
                            }
                            None => unreachable!(), // covered by syntactic check
                        }
                    }
                }
            }
        }
        Ok(true)
    };
    let mut stack: Vec<Frame<NodeIndex>> = vec![Frame::Outer];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Unwind {
                nodes,
                open_list: ol,
            } => {
                pop_state(&mut st, nodes);

                match next_from_ix(&mut st, nodes[0], ol) {
                    None => continue,
                    Some(nx) => {
                        let f = Frame::Inner {
                            nodes: [nx, nodes[1]],
                            open_list: ol,
                        };
                        stack.push(f);
                    }
                }
            }
            Frame::Outer => match next_candidate(&mut st) {
                None => continue,
                Some((nx, mx, ol)) => {
                    let f = Frame::Inner {
                        nodes: [nx, mx],
                        open_list: ol,
                    };
                    stack.push(f);
                }
            },
            Frame::Inner {
                nodes,
                open_list: ol,
            } => {
                let feasible = is_feasible(&mut st, nodes)?;
                if feasible {
                    push_state(&mut st, nodes);
                    if st[0].is_complete() {
                        return Ok(Some(true));
                    }
                    // Check cardinalities of Tin, Tout sets
                    if st[0].out_size == st[1].out_size
                        && st[0].ins_size == st[1].ins_size
                    {
                        let f0 = Frame::Unwind {
                            nodes,
                            open_list: ol,
                        };
                        stack.push(f0);
                        stack.push(Frame::Outer);
                        continue;
                    }
                    pop_state(&mut st, nodes);
                }
                match next_from_ix(&mut st, nodes[0], ol) {
                    None => continue,
                    Some(nx) => {
                        let f = Frame::Inner {
                            nodes: [nx, nodes[1]],
                            open_list: ol,
                        };
                        stack.push(f);
                    }
                }
            }
        }
    }
    Ok(None)
}
