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
// This module is a forked version of petgraph's isomorphism module @ 0.5.0.
// It has then been modified to function with PyDiGraph inputs instead of Graph.

use fixedbitset::FixedBitSet;
use std::cmp::Ordering;
use std::iter::FromIterator;
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

use rayon::slice::ParallelSliceMut;

type StablePyGraph<Ty> = StableGraph<PyObject, PyObject, Ty>;

// NOTE: assumes contiguous node ids.
trait NodeSorter<Ty>
where
    Ty: EdgeType,
{
    fn sort(&self, _: &StablePyGraph<Ty>) -> Vec<usize>;

    fn reorder(
        &self,
        py: Python,
        graph: &StablePyGraph<Ty>,
        mut mapping: Option<&mut HashMap<usize, usize>>,
    ) -> StablePyGraph<Ty> {
        let order = self.sort(graph);

        let mut new_graph = StablePyGraph::<Ty>::default();
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
        if mapping.is_some() {
            for (old_value, new_index) in id_map.iter().enumerate() {
                mapping.as_mut().unwrap().insert(*new_index, old_value);
            }
        }
        new_graph
    }
}

struct Vf2ppSorter;

impl<Ty> NodeSorter<Ty> for Vf2ppSorter
where
    Ty: EdgeType,
{
    fn sort(&self, graph: &StablePyGraph<Ty>) -> Vec<usize> {
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
                let this_level = Vec::from_iter(next_level);
                let this_level = process(this_level);

                next_level = HashSet::new();
                for bfs_node in this_level {
                    for neighbor in graph.neighbors_directed(
                        graph.from_index(bfs_node),
                        Outgoing,
                    ) {
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
        sorted_nodes.par_sort_unstable_by_key(|&node| (dout[node], din[node]));
        sorted_nodes.reverse();

        for node in sorted_nodes {
            bfs_tree(node);
        }

        order
    }
}

impl<'a, Ty> NodesRemoved for &'a StablePyGraph<Ty>
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
    graph: &'a StablePyGraph<Ty>,
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
    pub fn new(g: &'a StablePyGraph<Ty>) -> Self {
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

fn reindex_graph<Ty>(
    py: Python,
    graph: &StablePyGraph<Ty>,
) -> (StablePyGraph<Ty>, HashMap<usize, usize>)
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
    let mut new_graph = StablePyGraph::<Ty>::default();
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

    (
        new_graph,
        id_map.iter().map(|(k, v)| (v.index(), k.index())).collect(),
    )
}

trait SemanticMatcher<T> {
    fn enabled(&self) -> bool;
    fn eq(&mut self, _: &T, _: &T) -> PyResult<bool>;
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

/// [Graph] Return `true` if the graphs `g0` and `g1` are (sub) graph isomorphic.
///
/// Using the VF2 algorithm, examining both syntactic and semantic
/// graph isomorphism (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
#[allow(clippy::too_many_arguments)]
pub fn is_isomorphic<Ty, F, G>(
    py: Python,
    g0: &StablePyGraph<Ty>,
    g1: &StablePyGraph<Ty>,
    mut node_match: Option<F>,
    mut edge_match: Option<G>,
    id_order: bool,
    ordering: Ordering,
    induced: bool,
    mut mapping: Option<&mut HashMap<usize, usize>>,
) -> PyResult<bool>
where
    Ty: EdgeType,
    F: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
    G: FnMut(&PyObject, &PyObject) -> PyResult<bool>,
{
    let mut inner_temp_g0: StablePyGraph<Ty>;
    let mut inner_temp_g1: StablePyGraph<Ty>;
    let mut node_map_g0: Option<HashMap<usize, usize>>;
    let mut node_map_g1: Option<HashMap<usize, usize>>;
    let g0_out = if g0.nodes_removed() {
        let res = reindex_graph(py, g0);
        inner_temp_g0 = res.0;
        node_map_g0 = Some(res.1);
        &inner_temp_g0
    } else {
        node_map_g0 = None;
        g0
    };
    let g1_out = if g1.nodes_removed() {
        let res = reindex_graph(py, g1);
        inner_temp_g1 = res.0;
        node_map_g1 = Some(res.1);
        &inner_temp_g1
    } else {
        node_map_g1 = None;
        g1
    };

    if (g0_out.node_count().cmp(&g1_out.node_count()).then(ordering)
        != ordering)
        || (g0_out.edge_count().cmp(&g1_out.edge_count()).then(ordering)
            != ordering)
    {
        return Ok(false);
    }

    let g0 = if !id_order {
        inner_temp_g0 = if mapping.is_some() {
            let mut vf2pp_map: HashMap<usize, usize> =
                HashMap::with_capacity(g0_out.node_count());
            let temp = Vf2ppSorter.reorder(py, g0_out, Some(&mut vf2pp_map));
            match node_map_g0 {
                Some(ref mut g0_map) => {
                    for (_, old_index) in vf2pp_map.iter_mut() {
                        *old_index = g0_map[old_index];
                    }
                    *g0_map = vf2pp_map;
                }
                None => node_map_g0 = Some(vf2pp_map),
            };
            temp
        } else {
            Vf2ppSorter.reorder(py, g0_out, None)
        };
        &inner_temp_g0
    } else {
        g0_out
    };

    let g1 = if !id_order {
        inner_temp_g1 = if mapping.is_some() {
            let mut vf2pp_map: HashMap<usize, usize> =
                HashMap::with_capacity(g1_out.node_count());
            let temp = Vf2ppSorter.reorder(py, g1_out, Some(&mut vf2pp_map));
            match node_map_g1 {
                Some(ref mut g1_map) => {
                    for (_, old_index) in vf2pp_map.iter_mut() {
                        *old_index = g1_map[old_index];
                    }
                    *g1_map = vf2pp_map;
                }
                None => node_map_g1 = Some(vf2pp_map),
            };
            temp
        } else {
            Vf2ppSorter.reorder(py, g1_out, None)
        };
        &inner_temp_g1
    } else {
        g1_out
    };

    let mut st = [Vf2State::new(g0), Vf2State::new(g1)];
    let res = try_match(
        &mut st,
        g0,
        g1,
        &mut node_match,
        &mut edge_match,
        ordering,
        induced,
    )?;
    if mapping.is_some() && res == Some(true) {
        for (index, val) in st[1].mapping.iter().enumerate() {
            match node_map_g1 {
                Some(ref g1_map) => {
                    let node_index = g1_map[&index];
                    match node_map_g0 {
                        Some(ref g0_map) => mapping
                            .as_mut()
                            .unwrap()
                            .insert(g0_map[&val.index()], node_index),
                        None => mapping
                            .as_mut()
                            .unwrap()
                            .insert(val.index(), node_index),
                    };
                }
                None => {
                    match node_map_g0 {
                        Some(ref g0_map) => mapping
                            .as_mut()
                            .unwrap()
                            .insert(g0_map[&val.index()], index),
                        None => {
                            mapping.as_mut().unwrap().insert(val.index(), index)
                        }
                    };
                }
            };
        }
    }
    Ok(res.unwrap_or(false))
}

/// Return Some(bool) if isomorphism is decided, else None.
fn try_match<Ty, F, G>(
    mut st: &mut [Vf2State<Ty>; 2],
    g0: &StablePyGraph<Ty>,
    g1: &StablePyGraph<Ty>,
    node_match: &mut F,
    edge_match: &mut G,
    ordering: Ordering,
    induced: bool,
) -> PyResult<Option<bool>>
where
    Ty: EdgeType,
    F: SemanticMatcher<PyObject>,
    G: SemanticMatcher<PyObject>,
{
    if st[1].is_complete() {
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
                if !induced && j == 0 {
                    continue;
                }
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
        if succ_count[0].cmp(&succ_count[1]).then(ordering) != ordering {
            return Ok(false);
        }
        // R_pred
        if g[0].is_directed() {
            let mut pred_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors_directed(nodes[j], Incoming) {
                    pred_count[j] += 1;
                    if !induced && j == 0 {
                        continue;
                    }
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
            if pred_count[0].cmp(&pred_count[1]).then(ordering) != ordering {
                return Ok(false);
            }
        }
        macro_rules! rule {
            ($arr:ident, $j:expr, $dir:expr) => {{
                let mut count = 0;
                for n_neigh in g[$j].neighbors_directed(nodes[$j], $dir) {
                    let index = n_neigh.index();
                    if st[$j].$arr[index] > 0 && st[$j].mapping[index] == end {
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
            return Ok(false);
        }
        if g[0].is_directed()
            && rule!(out, 0, Incoming)
                .cmp(&rule!(out, 1, Incoming))
                .then(ordering)
                != ordering
        {
            return Ok(false);
        }
        // R_in
        if g[0].is_directed() {
            if rule!(ins, 0, Outgoing)
                .cmp(&rule!(ins, 1, Outgoing))
                .then(ordering)
                != ordering
            {
                return Ok(false);
            }

            if rule!(ins, 0, Incoming)
                .cmp(&rule!(ins, 1, Incoming))
                .then(ordering)
                != ordering
            {
                return Ok(false);
            }
        }
        // R_new
        if induced {
            let mut new_count = [0, 0];
            for j in graph_indices.clone() {
                for n_neigh in g[j].neighbors(nodes[j]) {
                    let index = n_neigh.index();
                    if st[j].out[index] == 0
                        && (st[j].ins.is_empty() || st[j].ins[index] == 0)
                    {
                        new_count[j] += 1;
                    }
                }
            }
            if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
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
                if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
                    return Ok(false);
                }
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
                    if st[1].is_complete() {
                        return Ok(Some(true));
                    }
                    // Check cardinalities of Tin, Tout sets
                    if st[0].out_size.cmp(&st[1].out_size).then(ordering)
                        == ordering
                        && st[0].ins_size.cmp(&st[1].ins_size).then(ordering)
                            == ordering
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
