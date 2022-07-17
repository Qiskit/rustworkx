use hashbrown::hash_map::HashMap;
use hashbrown::HashSet;
use petgraph::graph::Graph;
use petgraph::prelude::*;
use petgraph::visit::NodeIndexable;
use petgraph::Directed;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::StablePyGraph;
use retworkx_core::connectivity::connected_components;
use retworkx_core::planar::lr_planar::{LRState, Sign};

pub type Point = [f64; 2];

#[derive(Debug)]
pub struct CwCcw<T> {
    cw: Option<T>,
    ccw: Option<T>,
}

impl<T> Default for CwCcw<T> {
    fn default() -> Self {
        CwCcw {
            cw: None,
            ccw: None,
        }
    }
}

impl<T> CwCcw<T> {
    fn new(cw: T, ccw: T) -> Self {
        CwCcw {
            cw: Some(cw),
            ccw: Some(ccw),
        }
    }
}

#[derive(Debug)]
pub struct FirstNbr<T> {
    pub first_nbr: Option<T>,
}

impl<T> Default for FirstNbr<T> {
    fn default() -> Self {
        FirstNbr { first_nbr: None }
    }
}

impl<T> FirstNbr<T> {
    fn new(first_nbr: T) -> Self {
        FirstNbr {
            first_nbr: Some(first_nbr),
        }
    }
}

pub struct PlanarEmbedding {
    pub embedding: Graph<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>,
}

impl Default for PlanarEmbedding {
    fn default() -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>::new(),
        }
    }
}
impl PlanarEmbedding {
    pub fn new() -> Self {
        PlanarEmbedding {
            embedding: Graph::<FirstNbr<NodeIndex>, CwCcw<NodeIndex>, Directed>::new(),
        }
    }

    pub fn neighbors_cw_order(&mut self, v: NodeIndex) -> Vec<NodeIndex> {
        let mut nbrs: Vec<NodeIndex> = vec![];
        let first_nbr = self.embedding[v].first_nbr;
        println!("first_nbr in nbrs_cw_order v {:?} f nbr {:?}", v, first_nbr);

        if first_nbr.is_none() {
            // v has no neighbors
            return nbrs;
        }
        let start_node = first_nbr.unwrap();
        println!("start_node in nbrs_cw_order {:?}", start_node);
        nbrs.push(start_node);

        let mut node = self.get_edge_weight(v, start_node, true);
        println!("node {:?} nbrs {:?}", node, nbrs);

        if let Some(mut current_node) = node {
            println!("current_node 1 start {:?} current {:?}", start_node, current_node);
            let mut count = 0;
            while start_node != current_node && count < 20 {
                count += 1;
                println!("begin while - count {:?}", count);
                nbrs.push(current_node);
                node = self.get_edge_weight(v, current_node, true);
                current_node = match node {
                    Some(node) => node,
                    None => break,
                };
                println!("current_node 2 {:?} {:?}", current_node, nbrs);
            }
        }
        println!("nbrs 1 in nbrs_cw_order {:?}", nbrs);

        nbrs
    }

    fn add_half_edge_cw(
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_nbr: Option<NodeIndex>,
    ) {
        println!("IN HALF CW");
        let cw_weight = CwCcw::<NodeIndex>::default();
        self.embedding.add_edge(start_node, end_node, cw_weight);
        if ref_nbr.is_none() {
            println!("ref_nbr None {:?} {:?} {:?}", start_node, end_node, ref_nbr);

            // The start node has no neighbors
            println!("first update {:?} {:?}", start_node, end_node);
            self.update_edge_weight(start_node, end_node, end_node, true);
            println!("second update");
            self.update_edge_weight(start_node, end_node, end_node, false);
            self.embedding[start_node].first_nbr = Some(end_node);
            return;
        }
        // if ref_nbr not in self[start_node] error
        let ref_nbr_node = ref_nbr.unwrap();
        // DEBUG
        if self.embedding.find_edge(start_node, ref_nbr_node).is_none() {
            println!("NO REF NBR in ADD CW {:?} {:?}", start_node, ref_nbr_node);
        }
        let cw_ref = self.get_edge_weight(start_node, ref_nbr_node, true);
        println!("HALF CW before Some(cw_ref_node) start, end, ref, cw_ref {:?} {:?} {:?} {:?}", 
            start_node, end_node, ref_nbr, cw_ref);

        if let Some(cw_ref_node) = cw_ref {
            println!("HALF CW Some(cw_ref_node) {:?} {:?}", cw_ref, cw_ref_node);
            // Alter half-edge data structures
            self.update_edge_weight(start_node, ref_nbr_node, end_node, true);
            self.update_edge_weight(start_node, end_node, cw_ref_node, true);
            self.update_edge_weight(start_node, cw_ref_node, end_node, false);
            self.update_edge_weight(start_node, end_node, ref_nbr_node, false);
        }
    }

    fn add_half_edge_ccw(
        &mut self,
        start_node: NodeIndex,
        end_node: NodeIndex,
        ref_nbr: Option<NodeIndex>,
    ) {
        println!("IN HALF CCW");
        if ref_nbr.is_none() {
            // Start node has no neighbors
            let cw_weight = CwCcw::<NodeIndex>::default();
            self.embedding.add_edge(start_node, end_node, cw_weight);
            self.update_edge_weight(start_node, end_node, end_node, true);
            self.update_edge_weight(start_node, end_node, end_node, false);
            self.embedding[start_node].first_nbr = Some(end_node);
        } else {
            let ref_nbr_node = ref_nbr.unwrap();
            let ccw_ref_node = self.get_edge_weight(start_node, ref_nbr_node, false);
            self.add_half_edge_cw(start_node, end_node, ccw_ref_node);
            if ref_nbr == self.embedding[start_node].first_nbr {
                // Update first neighbor
                self.embedding[start_node].first_nbr = Some(end_node);
            }
        }
    }

    fn add_half_edge_first(&mut self, start_node: NodeIndex, end_node: NodeIndex) {
        let ref_node: Option<NodeIndex> = if self.embedding.node_bound() >= start_node.index()
            && self.embedding[start_node].first_nbr.is_some()
        {
            self.embedding[start_node].first_nbr
        } else {
            None
        };
        println!("in half first: start_node, end_node, ref_node {:?} {:?} {:?}", start_node, end_node, ref_node);
        self.add_half_edge_ccw(start_node, end_node, ref_node);
    }

    fn next_face_half_edge(&self, v: NodeIndex, w: NodeIndex) -> (NodeIndex, NodeIndex) {
        //println!("next v {:?} w {:?}", v, w);
        let new_node = self.get_edge_weight(w, v, false);
        println!("new node {:?}", new_node);
        // FIX THIS
        //
        //
        if new_node.is_none() {
            println!("NEW NODE NONE in next_face {:?} {:?} {:?}", new_node, v, w);
            panic!("HELP!"); //return (w, v);
        }
        (w, new_node.unwrap())
    }

    fn update_edge_weight(&mut self, v: NodeIndex, w: NodeIndex, new_node: NodeIndex, cw: bool) {
        let found_edge = self.embedding.find_edge(v, w);
        if found_edge.is_none() {
            return;
        }
        let found_weight = self.embedding.edge_weight_mut(found_edge.unwrap()); //.unwrap();
        if found_weight.is_none() {
            return;
        }

        if cw {
            found_weight.unwrap().cw = Some(new_node);
        } else {
            found_weight.unwrap().ccw = Some(new_node);
        }
    }

    fn get_edge_weight(&self, v: NodeIndex, w: NodeIndex, cw: bool) -> Option<NodeIndex> {
        let found_edge = self.embedding.find_edge(v, w);
        if found_edge.is_none() {
            println!("GET EDGE find edge is none {:?}", found_edge);
            return None;
        }
        println!("GET EDGE find edge{:?}", found_edge);
        let found_weight = self.embedding.edge_weight(found_edge.unwrap()); //.unwrap();
                                                                            //println!("after found weight {:?}", found_weight);
        if found_weight.is_none() {
            println!("GET EDGE Weight is none {:?}", found_weight);
            return None;
        }

        if cw {
            found_weight.unwrap().cw
        } else {
            found_weight.unwrap().ccw
        }
    }

    fn connect_components(&mut self, v: NodeIndex, w: NodeIndex) {
        self.add_half_edge_first(v, w);
        self.add_half_edge_first(w, v);
    }
}

pub fn create_embedding(
    planar_emb: &mut PlanarEmbedding,
    lr_state: &mut LRState<&StablePyGraph<Undirected>>,
) {
    // ********** DEBUG
    for node in lr_state.dir_graph.node_indices() {
        for edge in lr_state.dir_graph.edges(node) {
            println!("Edge {:?}, {:?}", edge.source(), edge.target());
        }
    }
    // ********** DEBUG END

    let mut ordered_adjs: Vec<Vec<NodeIndex>> = Vec::new();

    //let mut nesting_depth: HashMap<(NodeIndex, NodeIndex), isize> =
    //    HashMap::with_capacity(lr_state.nesting_depth.len());

    // Create the adjacency list for each node
    for v in lr_state.dir_graph.node_indices() {
        ordered_adjs.push(lr_state.dir_graph.edges(v).map(|e| e.target()).collect());

        // Add empty FirstNbr to the embedding
        let first_nbr = FirstNbr::<NodeIndex>::default();
        planar_emb.embedding.add_node(first_nbr);
    }

    // Sort the adjacency list using nesting depth as sort order
    for (v, adjs) in ordered_adjs.iter_mut().enumerate() {
        adjs.par_sort_by_key(|x| lr_state.nesting_depth[&(NodeIndex::new(v), *x)]);
    }

    for v in lr_state.dir_graph.node_indices() {
        // Change the sign for nesting_depth
        for e in lr_state.dir_graph.edges(v) {
            let edge: (NodeIndex, NodeIndex) = (e.source(), e.target());
            let signed_depth: isize = lr_state.nesting_depth[&edge] as isize;
            let signed_side = if lr_state.side.contains_key(&edge)
                && sign(edge, &mut lr_state.eref, &mut lr_state.side) == Sign::Minus
            {
                -1
            } else {
                1
            };
            lr_state.nesting_depth.insert(edge, signed_depth * signed_side);
        }
    }
    // ********** DEBUG
    for x in &ordered_adjs {
        println!("ordered {:?}", x);
    }
    // ********** DEBUG END

    // ********** DEBUG
    for x in &lr_state.nesting_depth {
        println!("nesting {:?}", x);
    }
    // ********** DEBUG END

    // Sort the adjacency list using nesting depth as sort order
    for (v, adjs) in ordered_adjs.iter_mut().enumerate() {
        adjs.par_sort_by_key(|x| lr_state.nesting_depth[&(NodeIndex::new(v), *x)]);
    }

    // ********** DEBUG
    for x in &ordered_adjs {
        println!("ordered 2222 {:?}", x);
    }
    // ********** DEBUG END

    // Add the initial half edge cw to the embedding using the ordered adjacency list
    for v in lr_state.dir_graph.node_indices() {
        let mut prev_node: Option<NodeIndex> = None;
        for w in &ordered_adjs[v.index()] {
            planar_emb.add_half_edge_cw(v, *w, prev_node);
            prev_node = Some(*w)
        }
    }

    // ********** DEBUG
    // ********** DEBUG
    for v in planar_emb.embedding.node_indices() {
        for w in &ordered_adjs[v.index()] {
            println!(
                "22222 embedding node v {:?} first_nbr {:?} cw(v,w) {:?} ccw(v,w) {:?} cw(w,v) {:?} ccw(w,v) {:?}",
                v,
                planar_emb.embedding[v],
                planar_emb.get_edge_weight(v, *w, true),
                planar_emb.get_edge_weight(v, *w, false),
                planar_emb.get_edge_weight(*w, v, true),
                planar_emb.get_edge_weight(*w, v, false),
            );
        }
    }
    // ********** DEBUG END

    println!("lr eparent {:?}", lr_state.eparent);
    // ********** DEBUG END

    println!("adjs len {:?} emb count {:?}", ordered_adjs.len(), planar_emb.embedding.node_count());

    // Start the DFS traversal for the embedding
    let mut left_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut right_ref: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(ordered_adjs.len());
    let mut idx: Vec<usize> = vec![0; ordered_adjs.len()];

    for v in lr_state.roots.iter() {
        // Create the stack with an initial entry of v
        let mut dfs_stack: Vec<NodeIndex> = vec![*v];

        while dfs_stack.len() > 0 {
            println!("top of while {:?}", dfs_stack);
            let v = dfs_stack.pop().unwrap();
            let idx2 = idx[v.index()];
            println!("top of while 2 v {:?} idx2 {:?} stack {:?}", v, idx2, dfs_stack);

            // Iterate over the ordered_adjs starting at the saved index until the end
            for w in ordered_adjs[v.index()][idx2..].iter() {
                idx[v.index()] += 1;

                println!("In dfs v idx[v] w {:?} {:?} {:?}", v, idx[v.index()], *w);
                let ei = (v, *w);
                println!("lr parent {:?}", lr_state.eparent.contains_key(&w));
                if lr_state.eparent.contains_key(&w) {
                    println!("parent w {:?} {:?}", lr_state.eparent[&w], ei);
                }
                if lr_state.eparent.contains_key(&w) && ei == lr_state.eparent[&w] {
                    println!("in ei {:?}", ei);
                    planar_emb.add_half_edge_first(*w, v);
                    left_ref.insert(v, *w);
                    right_ref.insert(v, *w);
                    dfs_stack.push(v);
                    dfs_stack.push(*w);
                    println!("before break dfs_stack {:?}", dfs_stack);
                    break;
                } else {
                    if !lr_state.side.contains_key(&ei) || lr_state.side[&ei] == Sign::Plus {
                        println!("plus cw {:?}", ei);
                        planar_emb.add_half_edge_cw(*w, v, Some(right_ref[w]));
                    } else {
                        println!("minus cw {:?}", ei);
                        planar_emb.add_half_edge_ccw(*w, v, Some(left_ref[w]));
                        left_ref.insert(*w, v);
                    }
                    println!("in else {:?}", ei);
                }
            }
        }
    }

    // ********** DEBUG
    for v in planar_emb.embedding.node_indices() {
        for w in &ordered_adjs[v.index()] {
            println!(
                "33333 embedding node v {:?} w {:?} first_nbr {:?} cw(v,w) {:?} ccw(v,w) {:?} cw(w,v) {:?} ccw(w,v) {:?}",
                v,
                w,
                planar_emb.embedding[v],
                planar_emb.get_edge_weight(v, *w, true),
                planar_emb.get_edge_weight(v, *w, false),
                planar_emb.get_edge_weight(*w, v, true),
                planar_emb.get_edge_weight(*w, v, false),
            );
        }
    }
    // ********** DEBUG END

    fn sign(
        edge: (NodeIndex, NodeIndex),
        eref: &mut HashMap<(NodeIndex, NodeIndex), (NodeIndex, NodeIndex)>,
        side: &mut HashMap<(NodeIndex, NodeIndex), Sign>,
    ) -> Sign {
        // Resolve the relative side of an edge to the absolute side.

        if eref.contains_key(&edge) {
            if side[&edge].clone() == sign(eref[&edge].clone(), eref, side) {
                *side.get_mut(&edge).unwrap() = Sign::Plus;
            } else {
                *side.get_mut(&edge).unwrap() = Sign::Minus;
            }
            eref.remove(&edge);
        }
        side[&edge]
    }
}

/// Once the embedding has been created, triangulate the embedding,
/// create a canonical ordering, and convert the embedding to position
/// coordinates.
pub fn embedding_to_pos(planar_emb: &mut PlanarEmbedding) -> Vec<Point> {
    let mut pos: Vec<Point> = vec![[0.0, 0.0]; planar_emb.embedding.node_count()];
    if planar_emb.embedding.node_count() < 4 {
        let default_pos = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]].to_vec();
        return planar_emb
            .embedding
            .node_indices()
            .map(|n| default_pos[n.index()])
            .collect();
    }
    let outer_face = triangulate_embedding(planar_emb, false);

    let node_list = canonical_ordering(planar_emb, outer_face);

    let mut right_t_child = HashMap::<Option<NodeIndex>, Option<NodeIndex>>::new();
    let mut left_t_child = HashMap::<Option<NodeIndex>, Option<NodeIndex>>::new();
    let mut delta_x = HashMap::<Option<NodeIndex>, isize>::new();
    let mut y_coord = HashMap::<Option<NodeIndex>, isize>::new();

    // Set the coordinates for the first 3 nodes.
    let v1 = node_list[0].0;
    let v2 = node_list[1].0;
    let v3 = node_list[2].0;

    delta_x.insert(v1, 0);
    y_coord.insert(v1, 0);
    right_t_child.insert(v1, v3);
    left_t_child.insert(v1, None);

    delta_x.insert(v2, 1);
    y_coord.insert(v2, 0);
    right_t_child.insert(v2, None);
    left_t_child.insert(v2, None);

    delta_x.insert(v3, 1);
    y_coord.insert(v3, 1);
    right_t_child.insert(v3, v2);
    left_t_child.insert(v3, None);

    // Set coordinates for the remaining nodes, adjusting
    // positions along the way as needed.
    for k in 3..node_list.len() {
        let vk = node_list[k].0;
        let contour_nbrs = &node_list[k].1;

        let wp = contour_nbrs[0];
        let wp1 = contour_nbrs[1];
        let wq = contour_nbrs[contour_nbrs.len() - 1];
        let wq1 = contour_nbrs[contour_nbrs.len() - 2];

        let adds_mult_tri = contour_nbrs.len() > 2;

        let mut delta_wp1_plus = 1;
        if delta_x.contains_key(&wp1) {
            delta_wp1_plus = delta_x[&wp1] + 1;
        }
        delta_x.insert(wp1, delta_wp1_plus);

        let mut delta_wq_plus = 1;
        if delta_x.contains_key(&wq) {
            delta_wq_plus = delta_x[&wq] + 1;
        }
        delta_x.insert(wq, delta_wq_plus);

        let delta_x_wp_wq = contour_nbrs[1..].iter().map(|x| delta_x[x]).sum::<isize>();

        let y_wp = y_coord[&wp].clone();
        let y_wq = y_coord[&wq].clone();
        delta_x.insert(vk, (delta_x_wp_wq - y_wp + y_wq) / 2 as isize); //y_coord[&wp] + y_coord[&wq]);
        y_coord.insert(vk, (delta_x_wp_wq + y_wp + y_wq) / 2 as isize); //y_coord.cloned()[&wp] + y_coord.cloned()[&wq]);

        //let d_vk = delta_x[&vk].clone();
        delta_x.insert(wq, delta_x_wp_wq - delta_x[&vk]);//d_vk);

        if adds_mult_tri {
            //let delta_wp1_minus = delta_x[&wp1] - delta_x[&vk];
            delta_x.insert(wp1, delta_x[&wp1] - delta_x[&vk]);//delta_wp1_minus);
        }
        right_t_child.insert(wp, vk);
        right_t_child.insert(vk, wq);
        if adds_mult_tri {
            left_t_child.insert(vk, wp1);
            right_t_child.insert(wq1, None);
        } else {
            left_t_child.insert(vk, None);
        }
    }

    // Set the position of the next tree child.
    fn set_position(
        parent: Option<NodeIndex>,
        tree: &HashMap<Option<NodeIndex>, Option<NodeIndex>>,
        remaining_nodes: &mut Vec<Option<NodeIndex>>,
        delta_x: &HashMap<Option<NodeIndex>, isize>,
        y_coord: &HashMap<Option<NodeIndex>, isize>,
        pos: &mut Vec<Point>,
    ) {
        let child = tree[&parent];
        let parent_node_x = pos[parent.unwrap().index()][0];

        if child.is_some() {
            let child_x = parent_node_x + (delta_x[&child] as f64);
            pos[child.unwrap().index()] = [child_x, (y_coord[&child] as f64)];
            remaining_nodes.push(child);
        }
    }

    pos[v1.unwrap().index()] = [0.0, y_coord[&v1] as f64];
    let mut remaining_nodes = vec![v1];

    // Set the positions of all the nodes.
    while remaining_nodes.len() > 0 {
        let parent_node = remaining_nodes.pop().unwrap();
        set_position(
            parent_node,
            &left_t_child,
            &mut remaining_nodes,
            &delta_x,
            &y_coord,
            &mut pos,
        );
        set_position(
            parent_node,
            &right_t_child,
            &mut remaining_nodes,
            &delta_x,
            &y_coord,
            &mut pos,
        );
    }
    pos
}

fn triangulate_embedding(
    planar_emb: &mut PlanarEmbedding,
    fully_triangulate: bool,
) -> Vec<NodeIndex> {
    let component_sets = connected_components(&planar_emb.embedding);
    println!("CONN COMP {:?}", component_sets);

    for i in 0..(component_sets.len() - 1) {
        let v1 = component_sets[i].iter().next().unwrap();
        let v2 = component_sets[i + 1].iter().next().unwrap();
        planar_emb.connect_components(*v1, *v2);
        println!("v1 v2 {:?} {:?}", *v1, *v2);
    }
    let mut outer_face = vec![];
    let mut face_list = vec![];
    let mut edges_counted: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
    println!(" in triangulate component sets{:?}", component_sets);

    for v in planar_emb.embedding.node_indices() {
        for w in planar_emb.neighbors_cw_order(v) {
            let new_face = make_bi_connected(planar_emb, v, w, &mut edges_counted);
            if new_face.len() > 0 {
                face_list.push(new_face.clone());
                if new_face.len() > outer_face.len() {
                    outer_face = new_face;
                }
            }
        }
    }

    println!("FINAL Face list {:?}", face_list);
    for face in face_list {
        println!("face {:?} outer_face {:?}", face, outer_face);
        if face != outer_face || fully_triangulate {
            triangulate_face(planar_emb, face[0], face[1]);
        }
    }
    if fully_triangulate {
        let v1 = outer_face[0];
        let v2 = outer_face[1];
        let v3 = planar_emb.get_edge_weight(v2, v1, false);
        outer_face = vec![v1, v2, v3.unwrap()];
    }
    println!("outer_face {:?}", outer_face);
    outer_face
}

fn make_bi_connected(
    planar_emb: &mut PlanarEmbedding,
    start_node: NodeIndex,
    out_node: NodeIndex,
    edges_counted: &mut HashSet<(NodeIndex, NodeIndex)>,
) -> Vec<NodeIndex> {
    if edges_counted.contains(&(start_node, out_node)) {//|| start_node == out_node {
        println!(
            "biconnect already in start out {:?} {:?}",
            start_node, out_node
        );
        return vec![];
    }
    edges_counted.insert((start_node, out_node));
    //println!("edges counted {:?} {:?} {:?}", start_node, out_node, edges_counted);
    let mut v1 = start_node.clone();
    let mut v2 = out_node.clone();
    let mut face_list: Vec<NodeIndex> = vec![start_node];
    let (_, mut v3) = planar_emb.next_face_half_edge(v1, v2);

    let mut count = 0;
    while (v2 != start_node || v3 != out_node) && count < 20 {
        // && count < 300 {
        if v1 == v2 {
            println!("BICONNECT V1==V2 should raise");
        }
        println!("face_list in while {:?} {:?} {:?}", v2, v3, face_list);

        if face_list.contains(&v2) {
            println!(
                "face_list contains v2 {:?} {:?} {:?}",
                v2, v3, edges_counted
            );
            planar_emb.add_half_edge_cw(v1, v3, Some(v2));
            planar_emb.add_half_edge_ccw(v3, v1, Some(v2));
            edges_counted.insert((v2, v3));
            edges_counted.insert((v3, v1));
            v2 = v1.clone();
        } else {
            println!(
                "face_list not contain v2 {:?} {:?} {:?}",
                v2, v3, edges_counted
            );
            face_list.push(v2);
        }
        v1 = v2.clone();
        println!("2 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);
        (v2, v3) = planar_emb.next_face_half_edge(v2, v3);
        println!("3 edges counted {:?} {:?} {:?}", v2, v3, edges_counted);

        edges_counted.insert((v1, v2));
        count += 1;
    }
    face_list
}

fn triangulate_face(planar_emb: &mut PlanarEmbedding, mut v1: NodeIndex, mut v2: NodeIndex) {
    let (_, mut v3) = planar_emb.next_face_half_edge(v1, v2);
    let (_, mut v4) = planar_emb.next_face_half_edge(v2, v3);
    if v1 == v2 || v1 == v3 {
        return;
    }
    let mut count = 0;
    while v1 != v4 {//} && count < 20 {
        if planar_emb.embedding.contains_edge(v1, v3) {
            (v1, v2, v3) = (v2, v3, v4);
            println!("\nContains EDGE {:?} {:?} {:?} {:?}", v1, v2, v3, v4);
        } else {
            planar_emb.add_half_edge_cw(v1, v3, Some(v2));
            planar_emb.add_half_edge_ccw(v3, v1, Some(v2));
            println!("get cw {:?}", planar_emb.get_edge_weight(v1, v3, true));
            println!("get ccw {:?}", planar_emb.get_edge_weight(v3, v1, false));
            (v1, v2, v3) = (v1, v3, v4);
            println!("\nDont Contains EDGE count {:?} {:?} {:?} {:?} {:?}", count, v1, v2, v3, v4);
        }
        (_, v4) = planar_emb.next_face_half_edge(v2, v3);
        println!("after v4 v2 {:?} v3 {:?} v4 {:?}", v2, v3, v4);
        count += 1;
    }
}

fn canonical_ordering(
    planar_emb: &mut PlanarEmbedding,
    outer_face: Vec<NodeIndex>,
) -> Vec<(Option<NodeIndex>, Vec<Option<NodeIndex>>)> {
    let v1 = outer_face[0];
    let v2 = outer_face[1];
    let mut chords: HashMap<NodeIndex, usize> = HashMap::new();
    let mut marked_nodes: HashSet<NodeIndex> = HashSet::new();
    let mut ready_to_pick: HashSet<NodeIndex> = HashSet::new();

    for node in outer_face.iter() {
        ready_to_pick.insert(*node);
    }

    let mut outer_face_cw_nbr: HashMap<NodeIndex, NodeIndex> =
        HashMap::with_capacity(outer_face.len());
    let mut outer_face_ccw_nbr: HashMap<NodeIndex, NodeIndex> =
        HashMap::with_capacity(outer_face.len());

    let mut prev_nbr = v2.clone();
    for v in outer_face[2..outer_face.len()].iter() {
        outer_face_ccw_nbr.insert(prev_nbr, *v);
        prev_nbr = *v;
    }
    outer_face_ccw_nbr.insert(prev_nbr, v1);
    println!("outer_face_ccw_nbr {:?}", outer_face_ccw_nbr);

    prev_nbr = v1.clone();
    for v in outer_face[1..outer_face.len()].iter().rev() {
        outer_face_cw_nbr.insert(prev_nbr, *v);
        prev_nbr = *v;
    }
    println!("outer_face_cw_nbr {:?}", outer_face_cw_nbr);

    fn is_outer_face_nbr(
        x: NodeIndex,
        y: NodeIndex,
        outer_face_cw_nbr: &HashMap<NodeIndex, NodeIndex>,
        outer_face_ccw_nbr: &HashMap<NodeIndex, NodeIndex>,
    ) -> bool {
        if !outer_face_ccw_nbr.contains_key(&x) {
            println!("1 is outer x {:?} y {:?} outer_ccw {:?} outer cw {:?}", x, y, outer_face_ccw_nbr, outer_face_cw_nbr);
            return outer_face_cw_nbr[&x] == y;
        }
        if !outer_face_cw_nbr.contains_key(&x) {
            println!("2 is outer x {:?} y {:?} outer_ccw {:?} outer cw {:?}", x, y, outer_face_ccw_nbr, outer_face_cw_nbr);
            return outer_face_ccw_nbr[&x] == y;
        }
        outer_face_cw_nbr[&x] == y || outer_face_ccw_nbr[&x] == y
    }

    fn is_on_outer_face(
        x: NodeIndex,
        v1: NodeIndex,
        marked_nodes: &HashSet<NodeIndex>,
        outer_face_ccw_nbr: &HashMap<NodeIndex, NodeIndex>,
    ) -> bool {
        !marked_nodes.contains(&x) && (outer_face_ccw_nbr.contains_key(&x) || x == v1)
    }

    for v in outer_face {
        let mut x = vec![];
        for nbr in planar_emb.neighbors_cw_order(v) {
            x.push(nbr);
        }
        println!("LIST NEIGBORS CW ORDER {:?}", x);
        println!("v in outer_face {:?}", v);
        for nbr in planar_emb.neighbors_cw_order(v) {
            println!("nbr in nbr cw order {:?}", nbr);

            if is_on_outer_face(nbr, v1, &marked_nodes, &outer_face_ccw_nbr)
                && !is_outer_face_nbr(v, nbr, &outer_face_cw_nbr, &outer_face_ccw_nbr)
            {
                let mut chords_plus = 0;
                if chords.contains_key(&v) {
                    chords_plus = chords[&v].clone();
                }
                chords.insert(v, chords_plus + 1);
                ready_to_pick.remove(&v);
                println!("in chords {:?} ready {:?}", chords, ready_to_pick);
            }
        }
        println!("READY 1st {:?} {:?}", v, ready_to_pick);
        println!("Chords 1st {:?}", chords);
    }

    let mut canon_order: Vec<(Option<NodeIndex>, Vec<Option<NodeIndex>>)> =
        vec![(None, vec![]); planar_emb.embedding.node_count()];

    canon_order[0] = (Some(v1), vec![]);
    canon_order[1] = (Some(v2), vec![]);
    println!("READY before {:?}", ready_to_pick);
    println!("Marked before {:?}", marked_nodes);

    ready_to_pick.remove(&v1);
    ready_to_pick.remove(&v2);

    println!("READY after {:?}", ready_to_pick);
    //println!("\n RANGE {:?}", (1..(planar_emb.embedding.node_count()-1)).rev());
    for k in (1..(planar_emb.embedding.node_count())).rev() {
        let v_try = ready_to_pick.iter().next();
        println!("\nv_try {:?}", v_try);
        if v_try.is_none() {
            continue;
        }
        let v = v_try.unwrap().clone();
        println!("in canon v {:?} k {:?}", v, k);
        ready_to_pick.remove(&v);
        marked_nodes.insert(v);
        println!("ready_to_pick {:?} marked {:?}", ready_to_pick, marked_nodes);

        let mut wp: Option<NodeIndex> = None;
        let mut wq: Option<NodeIndex> = None;

        println!("\n nbrs_cw_order of v {:?} {:?}", v, planar_emb.neighbors_cw_order(v));
        for nbr in planar_emb.neighbors_cw_order(v).iter() {
            println!("nbr {:?}", *nbr);
            if marked_nodes.contains(nbr) {
                continue;
            }
            println!("check outer face {:?}", *nbr);

            if is_on_outer_face(*nbr, v1, &marked_nodes, &outer_face_ccw_nbr) {
                println!("is outer {:?} {:?} {:?}", *nbr, v1, v2);//, outer_face_cw_nbr[nbr]);

                if *nbr == v1 {
                    wp = Some(v1);
                } else if *nbr == v2 {
                    wq = Some(v2);
                } else {
                    if outer_face_cw_nbr[nbr] == v {
                        wp = Some(*nbr);
                    } else {
                        wq = Some(*nbr);
                    }
                }
            }
            println!("wp {:?} wq {:?}", wp, wq);

            if (wp.is_some() && wq.is_some()) {
                println!("\nBREAKING {:?} {:?}", wp, wq);
                break;
            }
        }

        let mut wp_wq = vec![];
        if wp.is_some() {
            wp_wq = vec![wp];
            let mut nbr = wp.unwrap();
            while Some(nbr) != wq {
                let next_nbr = planar_emb.get_edge_weight(v, nbr, false).unwrap();
                println!("nbr {:?} next_nbr {:?} wp {:?} wq {:?} ", nbr, next_nbr, wp, wq);
                wp_wq.push(Some(next_nbr));
                outer_face_cw_nbr.insert(nbr, next_nbr);
                outer_face_ccw_nbr.insert(next_nbr, nbr);
                nbr = next_nbr;
                println!("of_cw {:?} of_ccw {:?} next_nbr {:?} wp_wq {:?} ", outer_face_cw_nbr, outer_face_ccw_nbr, next_nbr, wp_wq);
            }
            println!("after while wp_wq{:?}", wp_wq);
            if wp_wq.len() == 2 {
                let wp_un = wp.unwrap();
                if chords.contains_key(&wp_un) {
                    let chords_wp = chords[&wp_un].clone() - 1;
                    chords.insert(wp_un, chords_wp);
                    if chords[&wp_un] == 0 {
                        ready_to_pick.insert(wp_un);
                    }
                    let wq_un = wq.unwrap();
                    let chords_wq = chords[&wq_un].clone() - 1;
                    chords.insert(wq_un, chords_wq);
                    if chords[&wq_un] == 0 {
                        ready_to_pick.insert(wq_un);
                    }
                }
            } else {
                let mut new_face_nodes: HashSet<NodeIndex> = HashSet::new();
                if wp_wq.len() > 1 {
                    for w in &wp_wq[1..(wp_wq.len() - 1)] {
                        let w_un = w.unwrap();
                        new_face_nodes.insert(w_un);
                        for nbr in planar_emb.neighbors_cw_order(w_un) {
                            if is_on_outer_face(nbr, v1, &marked_nodes, &outer_face_ccw_nbr)
                                && !is_outer_face_nbr(
                                    w_un,
                                    nbr,
                                    &outer_face_cw_nbr,
                                    &outer_face_ccw_nbr,
                                )
                            {
                                if chords.contains_key(&w_un) {
                                    let chords_w = chords[&w_un].clone() + 1;
                                    chords.insert(w_un, chords_w);
                                    ready_to_pick.remove(&w_un);
                                    if !new_face_nodes.contains(&nbr) {
                                        let chords_nbr = chords[&nbr].clone() + 1;
                                        chords.insert(nbr, chords_nbr);
                                        ready_to_pick.remove(&nbr);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let x = wp_wq.clone();
        canon_order[k] = (Some(v), wp_wq);
        println!("\ncanon order k {:?} v {:?} wp_wq {:?}", k, v, x);
    }
    // let mut order: Vec<(Option<NodeIndex>, Vec<Option<NodeIndex>>)> = vec![];
    // //    vec![(None, vec![]); planar_emb.embedding.node_count()];
    // order.push((Some(NodeIndex::new(0)), vec![]));
    // order.push((Some(NodeIndex::new(1)), vec![]));
    // order.push((Some(NodeIndex::new(2)), vec![Some(NodeIndex::new(0)), Some(NodeIndex::new(1))]));
    // order.push((Some(NodeIndex::new(3)), vec![Some(NodeIndex::new(0)), Some(NodeIndex::new(2))]));
    // order.push((Some(NodeIndex::new(4)), vec![Some(NodeIndex::new(3)), Some(NodeIndex::new(2))]));
    println!("\nCANON ORDER {:?}", canon_order);
    canon_order
}
