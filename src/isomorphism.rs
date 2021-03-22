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

use hashbrown::{HashMap, HashSet};

use super::NodesRemoved;

use pyo3::prelude::*;

use petgraph::stable_graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::{
    EdgeRef, GetAdjacencyMatrix, IntoEdgeReferences, NodeIndexable,
};
use petgraph::EdgeType;
use petgraph::{Directed, Incoming, Outgoing};

type Graph<Ty> = StableGraph<PyObject, PyObject, Ty>;

// NOTE: assumes contiguous node ids.
pub trait NodeSorter<Ty>
where
    Ty: EdgeType,
{
    fn sort(&self, _: &Graph<Ty>) -> Vec<usize>;

    fn reorder(&self, py: Python, graph: &Graph<Ty>) -> Graph<Ty> {
        let order = self.sort(graph);

        let mut new_graph = Graph::<Ty>::default();
        let mut id_map: Vec<usize> = vec![0; graph.node_count()];
        for node in order {
            let node_index = graph.from_index(node);
            let node_data = graph.node_weight(node_index).unwrap();
            let new_index = new_graph.add_node(node_data.clone_ref(py));
            id_map[node] = graph.to_index(new_index);
        }
        for edge in graph.edge_references() {
            let edge_w = edge.weight();
            let p = id_map[graph.to_index(edge.source())];
            let c = id_map[graph.to_index(edge.target())];
            let p_index = graph.from_index(p);
            let c_index = graph.from_index(c);
            new_graph.add_edge(p_index, c_index, edge_w.clone_ref(py));
        }

        new_graph
    }
}

struct SimpleSorter;

impl<Ty> NodeSorter<Ty> for SimpleSorter
where
    Ty: EdgeType,
{
    fn sort(&self, graph: &Graph<Ty>) -> Vec<usize> {
        let n: usize = graph.node_count();
        (0..n).collect()
    }
}

pub struct Vf2ppSorter;

impl<Ty> NodeSorter<Ty> for Vf2ppSorter
where
    Ty: EdgeType,
{
    fn sort(&self, graph: &Graph<Ty>) -> Vec<usize> {
        let n = graph.node_count();

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

        let mut order: Vec<usize> = Vec::with_capacity(n);

        // Process BFS level
        let mut process = |mut vd: Vec<usize>| -> Vec<usize> {
            // repeatedly bring largest element in front.
            for i in 0..vd.len() {
                let (index, &item) = vd[i..]
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &node)| {
                        (conn_in[node], dout[node], conn_out[node], din[node])
                    })
                    .unwrap();

                vd.swap(i, i + index);
                order.push(item);

                for neigh in
                    graph.neighbors_directed(graph.from_index(item), Outgoing)
                {
                    conn_in[graph.to_index(neigh)] += 1;
                }

                if graph.is_directed() {
                    for neigh in graph
                        .neighbors_directed(graph.from_index(item), Incoming)
                    {
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

            let mut next_level: HashSet<usize> = HashSet::new();

            seen[root] = true;
            next_level.insert(root);
            while !next_level.is_empty() {
                let this_level: Vec<usize> =
                    next_level.iter().map(|&node| node).collect();
                let this_level = process(this_level);

                next_level = HashSet::new();
                for bfs_node in this_level {
                    for neighbor in
                        graph.neighbors_undirected(graph.from_index(bfs_node))
                    {
                        let neigh = graph.to_index(neighbor);
                        if !seen[neigh] {
                            seen[neigh] = true;
                            next_level.insert(neigh);
                        }
                    }
                }
            }
        };

        let mut sorted_nodes: Vec<usize> = (0..n).collect();
        sorted_nodes.sort_unstable_by_key(|&node| (dout[node], din[node]));
        sorted_nodes.reverse();

        for node in sorted_nodes {
            bfs_tree(node);
        }

        order
    }
}

impl<'a, Ty> NodesRemoved for &'a Graph<Ty>
where
    Ty: EdgeType,
{
    fn nodes_removed(&self) -> bool {
        self.node_bound() != self.node_count()
    }
}

#[derive(Debug)]
struct Vf2State<'a, Ty>
where
    Ty: EdgeType,
{
    graph: &'a Graph<Ty>,
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
    adjacency_matrix: FixedBitSet,
    generation: usize,
    _etype: marker::PhantomData<Directed>,
}

impl<'a, Ty> Vf2State<'a, Ty>
where
    Ty: EdgeType,
{
    pub fn new(g: &'a Graph<Ty>) -> Self {
        let c0 = g.node_count();
        Vf2State {
            graph: g,
            mapping: vec![NodeIndex::end(); c0],
            out: vec![0; c0],
            ins: vec![0; c0 * (g.is_directed() as usize)],
            out_size: 0,
            ins_size: 0,
            adjacency_matrix: g.adjacency_matrix(),
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

fn reindex_graph<Ty>(py: Python, graph: &Graph<Ty>) -> Graph<Ty>
where
    Ty: EdgeType,
{
    // NOTE: this is a hacky workaround to handle non-contiguous node ids in
    // VF2. The code which was forked from petgraph was written assuming the
    // Graph type and not StableGraph so it makes an implicit assumption on
    // node_bound() == node_count() which isn't true with removals on
    // StableGraph. This compacts the node ids as a workaround until VF2State
    // and try_match can be rewitten to handle this (and likely contributed
    // upstream to petgraph too).
    let mut new_graph = Graph::<Ty>::default();
    let mut id_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    for node_index in graph.node_indices() {
        let node_data = graph.node_weight(node_index).unwrap();
        let new_index = new_graph.add_node(node_data.clone_ref(py));
        id_map.insert(node_index, new_index);
    }
    for edge in graph.edge_references() {
        let edge_w = edge.weight();
        let p_index = id_map[&edge.source()];
        let c_index = id_map[&edge.target()];
        new_graph.add_edge(p_index, c_index, edge_w.clone_ref(py));
    }

    new_graph
}

/// [Graph] Return `true` if the graphs `g0` and `g1` are isomorphic.
///
/// Using the VF2 algorithm, examining both syntactic and semantic
/// graph isomorphism (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
pub fn is_isomorphic<Ty, F, G>(
    py: Python,
    g0: &Graph<Ty>,
    g1: &Graph<Ty>,
    mut node_match: Option<F>,
    mut edge_match: Option<G>,
) -> PyResult<bool>
where
    Ty: EdgeType,
    F: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
    G: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
{
    let inner_temp_g0: Graph<Ty>;
    let inner_temp_g1: Graph<Ty>;
    let g0_out = if g0.nodes_removed() {
        inner_temp_g0 = reindex_graph(py, g0);
        &inner_temp_g0
    } else {
        g0
    };
    let g1_out = if g1.nodes_removed() {
        inner_temp_g1 = reindex_graph(py, g1);
        &inner_temp_g1
    } else {
        g1
    };
    let g0 = &Vf2ppSorter.reorder(py, g0_out);
    let g1 = &Vf2ppSorter.reorder(py, g1_out);
    if g0.node_count() != g1.node_count() || g0.edge_count() != g1.edge_count()
    {
        return Ok(false);
    }

    let mut st = [Vf2State::new(g0), Vf2State::new(g1)];
    let res = try_match(&mut st, g0, g1, &mut node_match, &mut edge_match)?;
    Ok(res.unwrap_or(false))
}

trait SemanticMatcher<T> {
    fn enabled(&self) -> bool;
    fn eq(&mut self, _: &T, _: &T) -> PyResult<bool>;
}

struct NoSemanticMatch;

impl<T> SemanticMatcher<T> for NoSemanticMatch {
    #[inline]
    fn enabled(&self) -> bool {
        false
    }
    #[inline]
    fn eq(&mut self, _: &T, _: &T) -> PyResult<bool> {
        Ok(true)
    }
}

impl<T, F> SemanticMatcher<T> for Option<F>
where
    F: FnMut(&T, &T) -> PyResult<bool>,
{
    #[inline]
    fn enabled(&self) -> bool {
        self.is_some()
    }
    #[inline]
    fn eq(&mut self, a: &T, b: &T) -> PyResult<bool> {
        let res = (self.as_mut().unwrap())(a, b)?;
        Ok(res)
    }
}

/// Return Some(bool) if isomorphism is decided, else None.
fn try_match<Ty, F, G>(
    mut st: &mut [Vf2State<Ty>; 2],
    g0: &Graph<Ty>,
    g1: &Graph<Ty>,
    node_match: &mut F,
    edge_match: &mut G,
) -> PyResult<Option<bool>>
where
    Ty: EdgeType,
    F: SemanticMatcher<PyObject>,
    G: SemanticMatcher<PyObject>,
{
    if st[0].is_complete() {
        return Ok(Some(true));
    }

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
        |st: &mut [Vf2State<'_, Ty>; 2]| -> Option<(NodeIndex, NodeIndex, OpenList)> {
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
    let next_from_ix = |st: &mut [Vf2State<'_, Ty>; 2],
                        nx: NodeIndex,
                        open_list: OpenList|
     -> Option<NodeIndex> {
        // Find the next node index to try on the `from` side of the mapping
        let start = nx.index() + 1;
        let cand0 = match open_list {
            OpenList::Out => st[0].next_out_index(start),
            OpenList::In => st[0].next_in_index(start),
            OpenList::Other => st[0].next_rest_index(start),
        }
        .map(|c| c + start); // compensate for start offset.
        match cand0 {
            None => None, // no more candidates
            Some(ix) => {
                debug_assert!(ix >= start);
                Some(NodeIndex::new(ix))
            }
        }
    };
    //fn pop_state(nodes: [NodeIndex<Ix>; 2]) {
    let pop_state = |st: &mut [Vf2State<'_, Ty>; 2], nodes: [NodeIndex; 2]| {
        // Restore state.
        for j in graph_indices.clone() {
            st[j].pop_mapping(nodes[j]);
        }
    };
    //fn push_state(nodes: [NodeIndex<Ix>; 2]) {
    let push_state = |st: &mut [Vf2State<'_, Ty>; 2], nodes: [NodeIndex; 2]| {
        // Add mapping nx <-> mx to the state
        for j in graph_indices.clone() {
            st[j].push_mapping(nodes[j], nodes[1 - j]);
        }
    };
    //fn is_feasible(nodes: [NodeIndex<Ix>; 2]) -> bool {
    let mut is_feasible = |st: &mut [Vf2State<'_, Ty>; 2],
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
                    st[j].mapping[n_neigh.index()]
                } else {
                    nodes[1 - j]
                };
                if m_neigh == end {
                    continue;
                }
                let has_edge = g[1 - j].is_adjacent(
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
                    let m_neigh = st[j].mapping[n_neigh.index()];
                    if m_neigh == end {
                        continue;
                    }
                    let has_edge = g[1 - j].is_adjacent(
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
        // R_out
        let mut out_count = [0, 0];
        for j in graph_indices.clone() {
            for n_neigh in g[j].neighbors(nodes[j]) {
                let index = n_neigh.index();
                if st[j].out[index] > 0 && st[j].mapping[index] == end {
                    out_count[j] += 1;
                }
            }
        }
        if out_count[0] != out_count[1] {
            return Ok(false);
        }
        if g[0].is_directed() {
            let mut out_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors_directed(nodes[j], Incoming) {
                    let index = n_neigh.index();
                    if st[j].out[index] > 0 && st[j].mapping[index] == end {
                        out_count[j] += 1;
                    }
                }
            }
            if out_count[0] != out_count[1] {
                return Ok(false);
            }
        }
        // R_in
        if g[0].is_directed() {
            let mut in_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors(nodes[j]) {
                    let index = n_neigh.index();
                    if st[j].ins[index] > 0 && st[j].mapping[index] == end {
                        in_count[j] += 1;
                    }
                }
            }
            if in_count[0] != in_count[1] {
                return Ok(false);
            }
            if g[0].is_directed() {
                let mut in_count = [0, 0];
                for j in graph_indices.clone() {
                    for n_neigh in g[j].neighbors_directed(nodes[j], Incoming) {
                        let index = n_neigh.index();
                        if st[j].ins[index] > 0 && st[j].mapping[index] == end {
                            in_count[j] += 1;
                        }
                    }
                }
                if in_count[0] != in_count[1] {
                    return Ok(false);
                }
            }
        }
        // R_new
        let mut new_count = [0, 0];
        for j in graph_indices.clone() {
            for n_neigh in g[j].neighbors(nodes[j]) {
                let index = n_neigh.index();
                if st[j].out[index] == 0
                    && (st[j].ins.len() == 0 || st[j].ins[index] == 0)
                {
                    new_count[j] += 1;
                }
            }
        }
        if new_count[0] != new_count[1] {
            return Ok(false);
        }
        if g[0].is_directed() {
            let mut new_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors_directed(nodes[j], Incoming) {
                    let index = n_neigh.index();
                    if st[j].out[index] == 0 && st[j].ins[index] == 0 {
                        new_count[j] += 1;
                    }
                }
            }
            if new_count[0] != new_count[1] {
                return Ok(false);
            }
        }
        // semantic feasibility: compare associated data for nodes
        if node_match.enabled()
            && !node_match.eq(&g[0][nodes[0]], &g[1][nodes[1]])?
        {
            return Ok(false);
        }
        // semantic feasibility: compare associated data for edges
        if edge_match.enabled() {
            // outgoing edges
            for j in graph_indices.clone() {
                let mut edges = g[j].neighbors(nodes[j]).detach();
                while let Some((n_edge, n_neigh)) = edges.next(g[j]) {
                    // handle the self loop case; it's not in the mapping (yet)
                    let m_neigh = if nodes[j] != n_neigh {
                        st[j].mapping[n_neigh.index()]
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
                        let m_neigh = st[j].mapping[n_neigh.index()];
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
