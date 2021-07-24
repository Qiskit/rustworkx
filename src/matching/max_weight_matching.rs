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

// Needed to pass shared state between functions
// closures don't work because of recurssion
#![allow(clippy::too_many_arguments)]
// Allow single character names to match naming convention from
// paper
#![allow(clippy::many_single_char_names)]

use std::cmp::max;
use std::mem;
use std::panic;

use hashbrown::{HashMap, HashSet};

use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::graph::PyGraph;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

fn weight_callable(
    py: Python,
    weight: PyObject,
    weight_fn: &Option<PyObject>,
    default: i128,
) -> PyResult<i128> {
    match weight_fn {
        Some(weight_fn) => {
            let res = weight_fn.call1(py, (weight,))?;
            res.extract(py)
        }
        None => Ok(default),
    }
}

/// Return 2 * slack of edge k (does not work inside blossoms).
fn slack(
    edge_index: usize,
    dual_var: &[i128],
    edges: &[(usize, usize, i128)],
) -> i128 {
    let (source_index, target_index, weight) = edges[edge_index];
    dual_var[source_index] + dual_var[target_index] - 2 * weight
}

/// Generate the leaf vertices of a blossom.
fn blossom_leaves(
    blossom: usize,
    num_nodes: usize,
    blossom_children: &[Vec<usize>],
) -> PyResult<Vec<usize>> {
    let mut out_vec: Vec<usize> = Vec::new();
    if blossom < num_nodes {
        out_vec.push(blossom);
    } else {
        let child_blossom = &blossom_children[blossom];
        for c in child_blossom {
            let child = *c;
            if child < num_nodes {
                out_vec.push(child);
            } else {
                for v in blossom_leaves(child, num_nodes, blossom_children)? {
                    out_vec.push(v);
                }
            }
        }
    }
    Ok(out_vec)
}

/// Assign label t to the top-level blossom containing vertex w
/// and record the fact that w was reached through the edge with
/// remote endpoint p.
fn assign_label(
    w: usize,
    t: usize,
    p: Option<usize>,
    num_nodes: usize,
    in_blossoms: &[usize],
    labels: &mut Vec<Option<usize>>,
    label_ends: &mut Vec<Option<usize>>,
    best_edge: &mut Vec<Option<usize>>,
    queue: &mut Vec<usize>,
    blossom_children: &[Vec<usize>],
    blossom_base: &[Option<usize>],
    endpoints: &[usize],
    mate: &HashMap<usize, usize>,
) -> PyResult<()> {
    let b = in_blossoms[w];
    assert!(labels[w] == Some(0) && labels[b] == Some(0));
    labels[w] = Some(t);
    labels[b] = Some(t);
    label_ends[b] = p;
    label_ends[w] = p;
    best_edge[w] = None;
    best_edge[b] = None;
    if t == 1 {
        // b became an S-vertex/blossom; add it(s verticies) to the queue
        queue.append(&mut blossom_leaves(b, num_nodes, blossom_children)?);
    } else if t == 2 {
        // b became a T-vertex/blossom; assign label S to its mate.
        // (If b is a non-trivial blossom, its base is the only vertex
        // with an external mate.)
        let blossom_index: usize = b;
        let base: usize = blossom_base[blossom_index].unwrap();
        assert!(mate.get(&base).is_some());
        assign_label(
            endpoints[mate[&base]],
            1,
            mate.get(&base).map(|p| (p ^ 1)),
            num_nodes,
            in_blossoms,
            labels,
            label_ends,
            best_edge,
            queue,
            blossom_children,
            blossom_base,
            endpoints,
            mate,
        )?;
    }
    Ok(())
}

/// Trace back from vertices v and w to discover either a new blossom
/// or an augmenting path. Return the base vertex of the new blossom or None.
fn scan_blossom(
    node_a: usize,
    node_b: usize,
    in_blossoms: &[usize],
    blossom_base: &[Option<usize>],
    endpoints: &[usize],
    labels: &mut Vec<Option<usize>>,
    label_ends: &[Option<usize>],
    mate: &HashMap<usize, usize>,
) -> Option<usize> {
    let mut v: Option<usize> = Some(node_a);
    let mut w: Option<usize> = Some(node_b);
    // Trace back from v and w, placing breadcrumbs as we go
    let mut path: Vec<usize> = Vec::new();
    let mut base: Option<usize> = None;
    while v.is_some() || w.is_some() {
        // Look for a breadcrumb in v's blossom or put a new breadcrump
        let mut blossom = in_blossoms[v.unwrap()];
        if labels[blossom].is_none() || labels[blossom].unwrap() & 4 > 0 {
            base = blossom_base[blossom];
            break;
        }
        assert!(labels[blossom] == Some(1));
        path.push(blossom);
        labels[blossom] = Some(5);
        // Trace one step bacl.
        assert!(
            label_ends[blossom]
                == mate.get(&blossom_base[blossom].unwrap()).copied()
        );
        if label_ends[blossom].is_none() {
            // The base of blossom is single; stop tracing this path
            v = None;
        } else {
            let tmp = endpoints[label_ends[blossom].unwrap()];
            blossom = in_blossoms[tmp];
            assert!(labels[blossom] == Some(2));
            // blossom is a T-blossom; trace one more step back.
            assert!(label_ends[blossom].is_some());
            v = Some(endpoints[label_ends[blossom].unwrap()]);
        }
        // Swap v and w so that we alternate between both paths.
        if w.is_some() {
            mem::swap(&mut v, &mut w);
        }
    }
    // Remvoe breadcrumbs.
    for blossom in path {
        labels[blossom] = Some(1);
    }
    // Return base vertex, if we found one.
    base
}

/// Construct a new blossom with given base, containing edge k which
/// connects a pair of S vertices. Label the new blossom as S; set its dual
/// variable to zero; relabel its T-vertices to S and add them to the queue.
fn add_blossom(
    base: usize,
    edge: usize,
    blossom_children: &mut Vec<Vec<usize>>,
    num_nodes: usize,
    edges: &[(usize, usize, i128)],
    in_blossoms: &mut Vec<usize>,
    dual_var: &mut Vec<i128>,
    labels: &mut Vec<Option<usize>>,
    label_ends: &mut Vec<Option<usize>>,
    best_edge: &mut Vec<Option<usize>>,
    queue: &mut Vec<usize>,
    blossom_base: &mut Vec<Option<usize>>,
    endpoints: &[usize],
    blossom_endpoints: &mut Vec<Vec<usize>>,
    unused_blossoms: &mut Vec<usize>,
    blossom_best_edges: &mut Vec<Vec<usize>>,
    blossom_parents: &mut Vec<Option<usize>>,
    neighbor_endpoints: &[Vec<usize>],
    mate: &HashMap<usize, usize>,
) -> PyResult<()> {
    let (mut v, mut w, _weight) = edges[edge];
    let blossom_b = in_blossoms[base];
    let mut blossom_v = in_blossoms[v];
    let mut blossom_w = in_blossoms[w];
    // Create blossom
    let blossom = unused_blossoms.pop().unwrap();
    blossom_base[blossom] = Some(base);
    blossom_parents[blossom] = None;
    blossom_parents[blossom_b] = Some(blossom);
    // Make list of sub-blossoms and their interconnecting edge endpoints.
    blossom_children[blossom].clear();
    blossom_endpoints[blossom].clear();
    // Trace back from blossom_v to base.
    while blossom_v != blossom_b {
        // Add blossom_v to the new blossom
        blossom_parents[blossom_v] = Some(blossom);
        blossom_children[blossom].push(blossom_v);
        let blossom_v_endpoint_label = label_ends[blossom_v].unwrap();
        blossom_endpoints[blossom].push(blossom_v_endpoint_label);
        assert!(
            labels[blossom_v] == Some(2)
                || (labels[blossom_v] == Some(1)
                    && label_ends[blossom_v]
                        == mate
                            .get(&blossom_base[blossom_v].unwrap())
                            .copied())
        );
        // Trace one step back.
        assert!(label_ends[blossom_v].is_some());
        v = endpoints[blossom_v_endpoint_label];
        blossom_v = in_blossoms[v];
    }
    // Reverse lists, add endpoint that connects the pair of S vertices.
    blossom_children[blossom].push(blossom_b);
    blossom_children[blossom].reverse();
    blossom_endpoints[blossom].reverse();
    blossom_endpoints[blossom].push(2 * edge);
    // Trace back from w to base.
    while blossom_w != blossom_b {
        // Add blossom_w to the new blossom
        blossom_parents[blossom_w] = Some(blossom);
        blossom_children[blossom].push(blossom_w);
        let blossom_w_endpoint_label = label_ends[blossom_w].unwrap();
        blossom_endpoints[blossom].push(blossom_w_endpoint_label ^ 1);
        assert!(
            labels[blossom_w] == Some(2)
                || (labels[blossom_w] == Some(1)
                    && label_ends[blossom_w]
                        == mate
                            .get(&blossom_base[blossom_w].unwrap())
                            .copied())
        );
        // Trace one step back
        assert!(label_ends[blossom_w].is_some());
        w = endpoints[blossom_w_endpoint_label];
        blossom_w = in_blossoms[w];
    }
    // Set label to S.
    assert!(labels[blossom_b] == Some(1));
    labels[blossom] = Some(1);
    label_ends[blossom] = label_ends[blossom_b];
    // Set dual variable to 0
    dual_var[blossom] = 0;
    // Relabel vertices
    for node in blossom_leaves(blossom, num_nodes, blossom_children)? {
        if labels[in_blossoms[node]] == Some(2) {
            // This T-vertex now turns into an S-vertex because it becomes
            // part of an S-blossom; add it to the queue
            queue.push(node);
        }
        in_blossoms[node] = blossom;
    }
    // Compute blossom_best_edges[blossom]
    let mut best_edge_to: HashMap<usize, usize> =
        HashMap::with_capacity(2 * num_nodes);
    for bv_ref in &blossom_children[blossom] {
        let bv = *bv_ref;
        // This sub-blossom does not have a list of least-slack edges;
        // get the information from the vertices.
        let nblists: Vec<Vec<usize>> = if blossom_best_edges[bv].is_empty() {
            let mut tmp: Vec<Vec<usize>> = Vec::new();
            for node in blossom_leaves(bv, num_nodes, blossom_children)? {
                tmp.push(
                    neighbor_endpoints[node].iter().map(|p| p / 2).collect(),
                );
            }
            tmp
        } else {
            // Walk this sub-blossom's least-slack edges.
            vec![blossom_best_edges[bv].clone()]
        };
        for nblist in nblists {
            for edge_index in nblist {
                let (mut i, mut j, _edge_weight) = edges[edge_index];
                if in_blossoms[j] == blossom {
                    mem::swap(&mut i, &mut j);
                }
                let blossom_j = in_blossoms[j];
                if blossom_j != blossom
                    && labels[blossom_j] == Some(1)
                    && (best_edge_to.get(&blossom_j).is_none()
                        || slack(edge_index, dual_var, edges)
                            < slack(best_edge_to[&blossom_j], dual_var, edges))
                {
                    best_edge_to.insert(blossom_j, edge_index);
                }
            }
        }
        // Forget about least-slack edges of the sub-blossom.
        blossom_best_edges[bv].clear();
        best_edge[bv] = None;
    }
    blossom_best_edges[blossom] = best_edge_to.values().copied().collect();
    //select best_edge[blossom]
    best_edge[blossom] = None;
    for edge_index in &blossom_best_edges[blossom] {
        if best_edge[blossom].is_none()
            || slack(*edge_index, dual_var, edges)
                < slack(best_edge[blossom].unwrap(), dual_var, edges)
        {
            best_edge[blossom] = Some(*edge_index);
        }
    }
    Ok(())
}

/// Expand the given top level blossom
fn expand_blossom(
    blossom: usize,
    end_stage: bool,
    num_nodes: usize,
    blossom_children: &mut Vec<Vec<usize>>,
    blossom_parents: &mut Vec<Option<usize>>,
    in_blossoms: &mut Vec<usize>,
    dual_var: &[i128],
    labels: &mut Vec<Option<usize>>,
    label_ends: &mut Vec<Option<usize>>,
    best_edge: &mut Vec<Option<usize>>,
    queue: &mut Vec<usize>,
    blossom_base: &mut Vec<Option<usize>>,
    endpoints: &[usize],
    mate: &HashMap<usize, usize>,
    blossom_endpoints: &mut Vec<Vec<usize>>,
    allowed_edge: &mut Vec<bool>,
    unused_blossoms: &mut Vec<usize>,
) -> PyResult<()> {
    // Convert sub-blossoms into top-level blossoms.
    for s in blossom_children[blossom].clone() {
        blossom_parents[s] = None;
        if s < num_nodes {
            in_blossoms[s] = s
        } else if end_stage && dual_var[s] == 0 {
            // Recursively expand this sub-blossom
            expand_blossom(
                s,
                end_stage,
                num_nodes,
                blossom_children,
                blossom_parents,
                in_blossoms,
                dual_var,
                labels,
                label_ends,
                best_edge,
                queue,
                blossom_base,
                endpoints,
                mate,
                blossom_endpoints,
                allowed_edge,
                unused_blossoms,
            )?;
        } else {
            for v in blossom_leaves(s, num_nodes, blossom_children)? {
                in_blossoms[v] = s;
            }
        }
    }
    // if we expand a T-blossom during a stage, its a sub-blossoms must be
    // relabeled
    if !end_stage && labels[blossom] == Some(2) {
        // start at the sub-blossom through which the expanding blossom
        // obtained its label, and relabel sub-blossoms until we reach the
        // base.
        assert!(label_ends[blossom].is_some());
        let entry_child =
            in_blossoms[endpoints[label_ends[blossom].unwrap() ^ 1]];
        // Decied in which direction we will go around the blossom.
        let i = blossom_children[blossom]
            .iter()
            .position(|x| *x == entry_child)
            .unwrap();
        let mut j = i as i128;
        let j_step: i128;
        let endpoint_trick: usize = if i & 1 != 0 {
            // Start index is odd; go forward and wrap
            j -= blossom_children[blossom].len() as i128;
            j_step = 1;
            0
        } else {
            // start index is even; go backward
            j_step = -1;
            1
        };
        // Move along the blossom until we get to the base
        let mut p = label_ends[blossom].unwrap();
        while j != 0 {
            // Relabel the T-sub-blossom.
            labels[endpoints[p ^ 1]] = Some(0);
            if j < 0 {
                let length = blossom_endpoints[blossom].len();
                let index = length - j.abs() as usize;
                labels[endpoints[blossom_endpoints[blossom]
                    [index - endpoint_trick]
                    ^ endpoint_trick
                    ^ 1]] = Some(0);
            } else {
                labels[endpoints[blossom_endpoints[blossom]
                    [j as usize - endpoint_trick]
                    ^ endpoint_trick
                    ^ 1]] = Some(0);
            }
            assign_label(
                endpoints[p ^ 1],
                2,
                Some(p),
                num_nodes,
                in_blossoms,
                labels,
                label_ends,
                best_edge,
                queue,
                blossom_children,
                blossom_base,
                endpoints,
                mate,
            )?;
            // Step to the next S-sub-blossom and note it's forwward endpoint.
            let endpoint_index = if j < 0 {
                let tmp = j - endpoint_trick as i128;
                let length = blossom_endpoints[blossom].len();
                let index = length - tmp.abs() as usize;
                blossom_endpoints[blossom][index]
            } else {
                blossom_endpoints[blossom][j as usize - endpoint_trick]
            };
            allowed_edge[endpoint_index / 2] = true;
            j += j_step;
            p = if j < 0 {
                let tmp = j - endpoint_trick as i128;
                let length = blossom_endpoints[blossom].len();
                let index = length - tmp.abs() as usize;
                blossom_endpoints[blossom][index] ^ endpoint_trick
            } else {
                blossom_endpoints[blossom][j as usize - endpoint_trick]
                    ^ endpoint_trick
            };
            // Step to the next T-sub-blossom.
            allowed_edge[p / 2] = true;
            j += j_step;
        }
        // Relabel the base T-sub-blossom WITHOUT stepping through to
        // its mate (so don't call assign_label())
        let blossom_v = if j < 0 {
            let length = blossom_children[blossom].len();
            let index = length - j.abs() as usize;
            blossom_children[blossom][index]
        } else {
            blossom_children[blossom][j as usize]
        };
        labels[endpoints[p ^ 1]] = Some(2);
        labels[blossom_v] = Some(2);
        label_ends[endpoints[p ^ 1]] = Some(p);
        label_ends[blossom_v] = Some(p);
        best_edge[blossom_v] = None;
        // Continue along the blossom until we get back to entry_child
        j += j_step;
        let mut j_index = if j < 0 {
            let length = blossom_children[blossom].len();
            length - j.abs() as usize
        } else {
            j as usize
        };
        while blossom_children[blossom][j_index] != entry_child {
            // Examine the vertices of the sub-blossom to see whether
            // it is reachable from a neighboring S-vertex outside the
            // expanding blososm.
            let bv = blossom_children[blossom][j_index];
            if labels[bv] == Some(1) {
                // This sub-blossom just got label S through one of its
                // neighbors; leave it
                j += j_step;
                if j < 0 {
                    let length = blossom_children[blossom].len();
                    j_index = length - j.abs() as usize;
                } else {
                    j_index = j as usize;
                }
                continue;
            }
            let mut v: usize = 0;
            for tmp in blossom_leaves(bv, num_nodes, blossom_children)? {
                v = tmp;
                if labels[v].unwrap() != 0 {
                    break;
                }
            }
            // If the sub-blossom contains a reachable vertex, assign label T
            // to the sub-blossom
            if labels[v] != Some(0) {
                assert!(labels[v] == Some(2));
                assert!(in_blossoms[v] == bv);
                labels[v] = Some(0);
                labels[endpoints[mate[&blossom_base[bv].unwrap()]]] = Some(0);
                assign_label(
                    v,
                    2,
                    label_ends[v],
                    num_nodes,
                    in_blossoms,
                    labels,
                    label_ends,
                    best_edge,
                    queue,
                    blossom_children,
                    blossom_base,
                    endpoints,
                    mate,
                )?;
            }
            j += j_step;
            if j < 0 {
                let length = blossom_children[blossom].len();
                j_index = length - j.abs() as usize;
            } else {
                j_index = j as usize;
            }
        }
    }
    // Recycle the blossom number.
    labels[blossom] = None;
    label_ends[blossom] = None;
    blossom_children[blossom].clear();
    blossom_endpoints[blossom].clear();
    blossom_base[blossom] = None;
    best_edge[blossom] = None;
    unused_blossoms.push(blossom);
    Ok(())
}

/// Swap matched/unmatched edges over an alternating path through blossom b
/// between vertex v and the base vertex. Keep blossom bookkeeping consistent.
fn augment_blossom(
    blossom: usize,
    node: usize,
    num_nodes: usize,
    blossom_parents: &[Option<usize>],
    endpoints: &[usize],
    blossom_children: &mut Vec<Vec<usize>>,
    blossom_endpoints: &mut Vec<Vec<usize>>,
    blossom_base: &mut Vec<Option<usize>>,
    mate: &mut HashMap<usize, usize>,
) {
    // Bubble up through the blossom tree from vertex v to an immediate
    // sub-blossom of b.
    let mut tmp = node;
    while blossom_parents[tmp] != Some(blossom) {
        tmp = blossom_parents[tmp].unwrap();
    }
    // Recursively deal with the first sub-blossom.
    if tmp >= num_nodes {
        augment_blossom(
            tmp,
            node,
            num_nodes,
            blossom_parents,
            endpoints,
            blossom_children,
            blossom_endpoints,
            blossom_base,
            mate,
        );
    }
    // Decide in which direction we will go around the blossom.
    let i = blossom_children[blossom]
        .iter()
        .position(|x| *x == tmp)
        .unwrap();
    let mut j: i128 = i as i128;
    let j_step: i128;
    let endpoint_trick: usize = if i & 1 != 0 {
        // start index is odd; go forward and wrap.
        j -= blossom_children[blossom].len() as i128;
        j_step = 1;
        0
    } else {
        // Start index is even; go backward.
        j_step = -1;
        1
    };
    // Move along the blossom until we get to the base.
    while j != 0 {
        // Step to the next sub-blossom and augment it recursively.
        j += j_step;

        tmp = if j < 0 {
            let length = blossom_children[blossom].len();
            let index = length - j.abs() as usize;
            blossom_children[blossom][index]
        } else {
            blossom_children[blossom][j as usize]
        };
        let p = if j < 0 {
            let length = blossom_endpoints[blossom].len();
            let index = length - j.abs() as usize - endpoint_trick;
            blossom_endpoints[blossom][index] ^ endpoint_trick
        } else {
            blossom_endpoints[blossom][j as usize - endpoint_trick]
                ^ endpoint_trick
        };
        if tmp > num_nodes {
            augment_blossom(
                tmp,
                endpoints[p],
                num_nodes,
                blossom_parents,
                endpoints,
                blossom_children,
                blossom_endpoints,
                blossom_base,
                mate,
            );
        }
        j += j_step;
        if j < 0 {
            let length = blossom_children[blossom].len();
            let index = length - j.abs() as usize;
            tmp = blossom_children[blossom][index];
        } else {
            tmp = blossom_children[blossom][j as usize];
        }
        if tmp > num_nodes {
            augment_blossom(
                tmp,
                endpoints[p ^ 1],
                num_nodes,
                blossom_parents,
                endpoints,
                blossom_children,
                blossom_endpoints,
                blossom_base,
                mate,
            );
        }
        // Match the edge connecting those sub-blossoms.
        mate.insert(endpoints[p], p ^ 1);
        mate.insert(endpoints[p ^ 1], p);
    }
    // Rotate the list of sub-blossoms to put the new base at the front.
    let mut children: Vec<usize> = blossom_children[blossom][i..].to_vec();
    children.extend_from_slice(&blossom_children[blossom][..i]);
    blossom_children[blossom] = children;
    let mut endpoint: Vec<usize> = blossom_endpoints[blossom][i..].to_vec();
    endpoint.extend_from_slice(&blossom_endpoints[blossom][..i]);
    blossom_endpoints[blossom] = endpoint;
    blossom_base[blossom] = blossom_base[blossom_children[blossom][0]];
    assert!(blossom_base[blossom] == Some(node));
}

/// Swap matched/unmatched edges over an alternating path between two
/// single vertices. The augmenting path runs through edge k, which
/// connects a pair of S vertices.
fn augment_matching(
    edge: usize,
    num_nodes: usize,
    edges: &[(usize, usize, i128)],
    in_blossoms: &[usize],
    labels: &[Option<usize>],
    label_ends: &[Option<usize>],
    blossom_parents: &[Option<usize>],
    endpoints: &[usize],
    blossom_children: &mut Vec<Vec<usize>>,
    blossom_endpoints: &mut Vec<Vec<usize>>,
    blossom_base: &mut Vec<Option<usize>>,
    mate: &mut HashMap<usize, usize>,
) {
    let (v, w, _weight) = edges[edge];
    for (s_ref, p_ref) in [(v, 2 * edge + 1), (w, 2 * edge)].iter() {
        // Match vertex s to remote endpoint p. Then trace back from s
        // until we find a single vertex, swapping matched and unmatched
        // edges as we go.
        let mut s: usize = *s_ref;
        let mut p: usize = *p_ref;
        loop {
            let blossom_s = in_blossoms[s];
            assert!(labels[blossom_s] == Some(1));
            assert!(
                label_ends[blossom_s]
                    == mate.get(&blossom_base[blossom_s].unwrap()).copied()
            );
            // Augment through the S-blossom from s to base.
            if blossom_s >= num_nodes {
                augment_blossom(
                    blossom_s,
                    s,
                    num_nodes,
                    blossom_parents,
                    endpoints,
                    blossom_children,
                    blossom_endpoints,
                    blossom_base,
                    mate,
                );
            }
            // Update mate[s]
            mate.insert(s, p);
            // Trace one step back.
            if label_ends[blossom_s].is_none() {
                // Reached a single vertex; stop
                break;
            }
            let t = endpoints[label_ends[blossom_s].unwrap()];
            let blossom_t = in_blossoms[t];
            assert!(labels[blossom_t] == Some(2));
            // Trace one step back
            assert!(label_ends[blossom_t].is_some());
            s = endpoints[label_ends[blossom_t].unwrap()];
            let j = endpoints[label_ends[blossom_t].unwrap() ^ 1];
            // Augment through the T-blossom from j to base.
            assert!(blossom_base[blossom_t] == Some(t));
            if blossom_t >= num_nodes {
                augment_blossom(
                    blossom_t,
                    j,
                    num_nodes,
                    blossom_parents,
                    endpoints,
                    blossom_children,
                    blossom_endpoints,
                    blossom_base,
                    mate,
                );
            }
            // Update mate[j]
            mate.insert(j, label_ends[blossom_t].unwrap());
            // Keep the opposite endpoint;
            // it will be assigned to mate[s] in the next step.
            p = label_ends[blossom_t].unwrap() ^ 1;
        }
    }
}

/// Swap matched/unmatched edges over an alternating path between two
/// single vertices. The augmenting path runs through the edge, which
/// connects a pair of S vertices.
fn verify_optimum(
    max_cardinality: bool,
    num_nodes: usize,
    num_edges: usize,
    edges: &[(usize, usize, i128)],
    endpoints: &[usize],
    dual_var: &[i128],
    blossom_parents: &[Option<usize>],
    blossom_endpoints: &[Vec<usize>],
    blossom_base: &[Option<usize>],
    mate: &HashMap<usize, usize>,
) {
    let dual_var_node_min: i128 = *dual_var[..num_nodes].iter().min().unwrap();
    let node_dual_offset: i128 = if max_cardinality {
        // Vertices may have negative dual;
        // find a constant non-negative number to add to all vertex duals.
        max(0, -dual_var_node_min)
    } else {
        0
    };
    assert!(dual_var_node_min + node_dual_offset >= 0);
    assert!(*dual_var[num_nodes..].iter().min().unwrap() >= 0);
    // 0. all edges have non-negative slack and
    // 1. all matched edges have zero slack;
    for (edge, (i, j, weight)) in edges.iter().enumerate().take(num_edges) {
        let mut s = dual_var[*i] + dual_var[*j] - 2 * weight;
        let mut i_blossoms: Vec<usize> = vec![*i];
        let mut j_blossoms: Vec<usize> = vec![*j];
        while blossom_parents[*i_blossoms.last().unwrap()].is_some() {
            i_blossoms
                .push(blossom_parents[*i_blossoms.last().unwrap()].unwrap());
        }
        while blossom_parents[*j_blossoms.last().unwrap()].is_some() {
            j_blossoms
                .push(blossom_parents[*j_blossoms.last().unwrap()].unwrap());
        }
        i_blossoms.reverse();
        j_blossoms.reverse();
        for (blossom_i, blossom_j) in i_blossoms.iter().zip(j_blossoms.iter()) {
            if blossom_i != blossom_j {
                break;
            }
            s += 2 * dual_var[*blossom_i];
        }
        assert!(s >= 0);

        if (mate.get(i).is_some() && mate.get(i).unwrap() / 2 == edge)
            || (mate.get(j).is_some() && mate.get(j).unwrap() / 2 == edge)
        {
            assert!(mate[i] / 2 == edge && mate[j] / 2 == edge);
            assert!(s == 0);
        }
    }
    // 2. all single vertices have zero dual value;
    for (node, dual_var_node) in dual_var.iter().enumerate().take(num_nodes) {
        assert!(
            mate.get(&node).is_some() || dual_var_node + node_dual_offset == 0
        );
    }
    // 3. all blossoms with positive dual value are full.
    for blossom in num_nodes..2 * num_nodes {
        if blossom_base[blossom].is_some() && dual_var[blossom] > 0 {
            assert!(blossom_endpoints[blossom].len() % 2 == 1);
            for p in blossom_endpoints[blossom].iter().skip(1).step_by(2) {
                assert!(mate.get(&endpoints[*p]).copied() == Some(p ^ 1));
                assert!(mate.get(&endpoints[*p ^ 1]) == Some(p));
            }
        }
    }
}

/// Compute a maximum-weighted matching in the general undirected weighted
/// graph given by "edges". If `max_cardinality` is true, only
/// maximum-cardinality matchings are considered as solutions.
///
/// The algorithm is based on "Efficient Algorithms for Finding Maximum
/// Matching in Graphs" by Zvi Galil, ACM Computing Surveys, 1986.
///
/// Based on networkx implementation
/// https://github.com/networkx/networkx/blob/3351206a3ce5b3a39bb2fc451e93ef545b96c95b/networkx/algorithms/matching.py
///
/// With reference to the standalone protoype implementation from:
/// http://jorisvr.nl/article/maximum-matching
///
/// http://jorisvr.nl/files/graphmatching/20130407/mwmatching.py
///
/// The function takes time O(n**3)
pub fn max_weight_matching(
    py: Python,
    graph: &PyGraph,
    max_cardinality: bool,
    weight_fn: Option<PyObject>,
    default_weight: i128,
    verify_optimum_flag: bool,
) -> PyResult<HashSet<(usize, usize)>> {
    let num_edges = graph.graph.edge_count();
    let num_nodes = graph.graph.node_count();
    let mut out_set: HashSet<(usize, usize)> =
        HashSet::with_capacity(num_nodes);
    // Exit fast for graph without edges
    if num_edges == 0 {
        return Ok(HashSet::new());
    }
    // Node indicies in the PyGraph may not be contiguous however the
    // algorithm operates on contiguous indices 0..num_nodes. node_map maps
    // the PyGraph's NodeIndex to the contingous usize used inside the
    // algorithm
    let node_map: HashMap<NodeIndex, usize> = graph
        .graph
        .node_indices()
        .enumerate()
        .map(|(index, node_index)| (node_index, index))
        .collect();
    let mut edges: Vec<(usize, usize, i128)> = Vec::with_capacity(num_edges);
    let mut max_weight: i128 = 0;
    for edge in graph.graph.edge_references() {
        let edge_weight: i128 = weight_callable(
            py,
            edge.weight().clone_ref(py),
            &weight_fn,
            default_weight,
        )?;
        if edge_weight > max_weight {
            max_weight = edge_weight;
        };
        edges.push((
            node_map[&edge.source()],
            node_map[&edge.target()],
            edge_weight,
        ));
    }
    // If p is an edge endpoint
    // endpoints[p] is the node index to which endpoint p is attached
    let endpoints: Vec<usize> = (0..2 * num_edges)
        .map(|endpoint| {
            let edge_tuple = edges[endpoint / 2];
            let out_value: usize = if endpoint % 2 == 0 {
                edge_tuple.0
            } else {
                edge_tuple.1
            };
            out_value
        })
        .collect();
    // If v is a node/vertex
    // neighbor_endpoints[v] is the list of remote endpoints of the edges
    // attached to v.
    // Not modified by algorithm (only mut to initially construct contents).
    let mut neighbor_endpoints: Vec<Vec<usize>> =
        (0..num_nodes).map(|_| Vec::new()).collect();
    for edge in 0..num_edges {
        neighbor_endpoints[edges[edge].0].push(2 * edge + 1);
        neighbor_endpoints[edges[edge].1].push(2 * edge);
    }
    // If v is a vertex,
    // mate[v] is the remote endpoint of its matched edge, or None if it is
    // single (i.e. endpoint[mate[v]] is v's partner vertex).
    // Initially all vertices are single; updated during augmentation.
    let mut mate: HashMap<usize, usize> = HashMap::with_capacity(num_nodes);
    // If b is a top-level blossom
    // label[b] is 0 if b is unlabeled (free);
    //             1 if b is a S-vertex/blossom;
    //             2 if b is a T-vertex/blossom;
    // The label of a vertex/node is found by looking at the label of its
    // top-level containing blossom
    // If v is a node/vertex inside a T-Blossom,
    // label[v] is 2 if and only if v is reachable from an S_Vertex outside
    // the blossom.
    let mut labels: Vec<Option<usize>>;
    // If b is a labeled top-level blossom,
    // label_ends[b] is the remote endpoint of the edge through which b
    // obtained its label, or None if b's base vertex is single.
    // If v is a vertex inside a T-blossom and label[v] == 2,
    // label_ends[v] is the remote endpoint of the edge through which v is
    // reachable from outside the blossom
    let mut label_ends: Vec<Option<usize>>;
    // If v is a vertex/node
    // in_blossoms[v] is the top-level blossom to which v belongs.
    // If v is a top-level vertex, v is itself a blossom (a trivial blossom)
    // and in_blossoms[v] == v.
    // Initially all nodes are top-level trivial blossoms.
    let mut in_blossoms: Vec<usize> = (0..num_nodes).collect();
    // if b is a sub-blossom
    // blossom_parents[b] is its immediate parent
    // If b is a top-level blossom, blossom_parents[b] is None
    let mut blossom_parents: Vec<Option<usize>> = vec![None; 2 * num_nodes];
    // If b is a non-trivial (sub-)blossom,
    // blossom_children[b] is an ordered list of its sub-blossoms, starting with
    // the base and going round the blossom.
    let mut blossom_children: Vec<Vec<usize>> =
        (0..2 * num_nodes).map(|_| Vec::new()).collect();
    // If b is a (sub-)blossom,
    // blossombase[b] is its base VERTEX (i.e. recursive sub-blossom).
    let mut blossom_base: Vec<Option<usize>> =
        (0..num_nodes).map(Some).collect();
    blossom_base.append(&mut vec![None; num_nodes]);
    // If b is a non-trivial (sub-)blossom,
    // blossom_endpoints[b] is a list of endpoints on its connecting edges,
    // such that blossom_endpoints[b][i] is the local endpoint of
    // blossom_children[b][i] on the edge that connects it to
    // blossom_children[b][wrap(i+1)].
    let mut blossom_endpoints: Vec<Vec<usize>> =
        (0..2 * num_nodes).map(|_| Vec::new()).collect();
    // If v is a free vertex (or an unreached vertex inside a T-blossom),
    // best_edge[v] is the edge to an S-vertex with least slack,
    // or None if there is no such edge. If b is a (possibly trivial)
    // top-level S-blossom,
    // best_edge[b] is the least-slack edge to a different S-blossom,
    // or None if there is no such edge.
    let mut best_edge: Vec<Option<usize>>;
    // If b is a non-trivial top-level S-blossom,
    // blossom_best_edges[b] is a list of least-slack edges to neighboring
    // S-blossoms, or None if no such list has been computed yet.
    // This is used for efficient computation of delta3.
    let mut blossom_best_edges: Vec<Vec<usize>> =
        (0..2 * num_nodes).map(|_| Vec::new()).collect();
    let mut unused_blossoms: Vec<usize> = (num_nodes..2 * num_nodes).collect();
    // If v is a vertex,
    // dual_var[v] = 2 * u(v) where u(v) is the v's variable in the dual
    // optimization problem (multiplication by two ensures integer values
    // throughout the algorithm if all edge weights are integers).
    // If b is a non-trivial blossom,
    // dual_var[b] = z(b) where z(b) is b's variable in the dual optimization
    // problem.
    // dual_var is for vertex v in 0..num_nodes and blossom b in
    // num_nodes..2* num nodes
    let mut dual_var: Vec<i128> = vec![max_weight; num_nodes];
    dual_var.append(&mut vec![0; num_nodes]);
    // If allowed_edge[k] is true, edge k has zero slack in the optimization
    // problem; if allowed_edge[k] is false, the edge's slack may or may not
    // be zero.
    let mut allowed_edge: Vec<bool>;
    // Queue of newly discovered S-vertices
    let mut queue: Vec<usize> = Vec::with_capacity(num_nodes);

    // Main loop: continue until no further improvement is possible
    for _ in 0..num_nodes {
        // Each iteration of this loop is a "stage".
        // A stage finds an augmenting path and uses that to improve
        // the matching.

        // Removal labels from top-level blossoms/vertices
        labels = vec![Some(0); 2 * num_nodes];
        label_ends = vec![None; 2 * num_nodes];
        // Forget all about least-slack edges.
        best_edge = vec![None; 2 * num_nodes];
        blossom_best_edges
            .splice(num_nodes.., (0..num_nodes).map(|_| Vec::new()));
        // Loss of labeling means that we can not be sure that currently
        // allowable edges remain allowable througout this stage.
        allowed_edge = vec![false; num_edges];
        // Make queue empty
        queue.clear();
        // Label single blossom/vertices with S and put them in queue.
        for v in 0..num_nodes {
            if mate.get(&v).is_none() && labels[in_blossoms[v]] == Some(0) {
                assign_label(
                    v,
                    1,
                    None,
                    num_nodes,
                    &in_blossoms,
                    &mut labels,
                    &mut label_ends,
                    &mut best_edge,
                    &mut queue,
                    &blossom_children,
                    &blossom_base,
                    &endpoints,
                    &mate,
                )?;
            }
        }
        // Loop until we succeed in augmenting the matching.
        let mut augmented = false;
        loop {
            // Each iteration of this loop is a "substage".
            // A substage tries to find an augmenting path;
            // if found, the path is used to improve the matching and
            // the stage ends. If there is no augmenting path, the
            // primal-dual method is used to find some slack out of
            // the dual variables.

            // Continue labeling until all vertices which are reachable
            // through an alternating path have got a label.
            while !queue.is_empty() && !augmented {
                // Take an S vertex from the queue
                let v = queue.pop().unwrap();
                assert!(labels[in_blossoms[v]] == Some(1));

                // Scan its neighbors
                for p in &neighbor_endpoints[v] {
                    let k = *p / 2;
                    let mut kslack = 0;
                    let w = endpoints[*p];
                    // w is a neighbor of v
                    if in_blossoms[v] == in_blossoms[w] {
                        // this edge is internal to a blossom; ignore it
                        continue;
                    }
                    if !allowed_edge[k] {
                        kslack = slack(k, &dual_var, &edges);
                        if kslack <= 0 {
                            // edge k has zero slack -> it is allowable
                            allowed_edge[k] = true;
                        }
                    }
                    if allowed_edge[k] {
                        if labels[in_blossoms[w]] == Some(0) {
                            // (C1) w is a free vertex;
                            // label w with T and label its mate with S (R12).
                            assign_label(
                                w,
                                2,
                                Some(*p ^ 1),
                                num_nodes,
                                &in_blossoms,
                                &mut labels,
                                &mut label_ends,
                                &mut best_edge,
                                &mut queue,
                                &blossom_children,
                                &blossom_base,
                                &endpoints,
                                &mate,
                            )?;
                        } else if labels[in_blossoms[w]] == Some(1) {
                            // (C2) w is an S-vertex (not in the same blossom);
                            // follow back-links to discover either an
                            // augmenting path or a new blossom.
                            let base = scan_blossom(
                                v,
                                w,
                                &in_blossoms,
                                &blossom_base,
                                &endpoints,
                                &mut labels,
                                &label_ends,
                                &mate,
                            );
                            match base {
                                // Found a new blossom; add it to the blossom
                                // bookkeeping and turn it into an S-blossom.
                                Some(base) => add_blossom(
                                    base,
                                    k,
                                    &mut blossom_children,
                                    num_nodes,
                                    &edges,
                                    &mut in_blossoms,
                                    &mut dual_var,
                                    &mut labels,
                                    &mut label_ends,
                                    &mut best_edge,
                                    &mut queue,
                                    &mut blossom_base,
                                    &endpoints,
                                    &mut blossom_endpoints,
                                    &mut unused_blossoms,
                                    &mut blossom_best_edges,
                                    &mut blossom_parents,
                                    &neighbor_endpoints,
                                    &mate,
                                )?,
                                // Found an augmenting path; augment the
                                // matching and end this stage.
                                None => {
                                    augment_matching(
                                        k,
                                        num_nodes,
                                        &edges,
                                        &in_blossoms,
                                        &labels,
                                        &label_ends,
                                        &blossom_parents,
                                        &endpoints,
                                        &mut blossom_children,
                                        &mut blossom_endpoints,
                                        &mut blossom_base,
                                        &mut mate,
                                    );
                                    augmented = true;
                                    break;
                                }
                            };
                        } else if labels[w] == Some(0) {
                            // w is inside a T-blossom, but w itself has not
                            // yet been reached from outside the blossom;
                            // mark it as reached (we need this to relabel
                            // during T-blossom expansion).
                            assert!(labels[in_blossoms[w]] == Some(2));
                            labels[w] = Some(2);
                            label_ends[w] = Some(*p ^ 1);
                        }
                    } else if labels[in_blossoms[w]] == Some(1) {
                        // Keep track of the least-slack non-allowable edge to
                        // a different S-blossom
                        let blossom = in_blossoms[v];
                        if best_edge[blossom].is_none()
                            || kslack
                                < slack(
                                    best_edge[blossom].unwrap(),
                                    &dual_var,
                                    &edges,
                                )
                        {
                            best_edge[blossom] = Some(k);
                        }
                    } else if labels[w] == Some(0) {
                        // w is a free vertex (or an unreached vertex inside
                        // a T-blossom) but we can not reach it yet;
                        // keep track of the least-slack edge that reaches w.
                        if best_edge[w].is_none()
                            || kslack
                                < slack(
                                    best_edge[w].unwrap(),
                                    &dual_var,
                                    &edges,
                                )
                        {
                            best_edge[w] = Some(k)
                        }
                    }
                }
            }
            if augmented {
                break;
            }
            // There is no augmenting path under these constraints;
            // compute delta and reduce slack in the optimization problem.
            // (Note that our vertex dual variables, edge slacks and delta's
            // are pre-multiplied by two.)
            let mut delta_type = -1;
            let mut delta: Option<i128> = None;
            let mut delta_edge: Option<usize> = None;
            let mut delta_blossom: Option<usize> = None;

            // Compute delta1: the minimum value of any vertex dual.
            if !max_cardinality {
                delta_type = 1;
                delta = Some(*dual_var[..num_nodes].iter().min().unwrap());
            }

            // Compute delta2: the minimum slack on any edge between
            // an S-vertex and a free vertex.
            for v in 0..num_nodes {
                if labels[in_blossoms[v]] == Some(0) && best_edge[v].is_some() {
                    let d = slack(best_edge[v].unwrap(), &dual_var, &edges);
                    if delta_type == -1 || Some(d) < delta {
                        delta = Some(d);
                        delta_type = 2;
                        delta_edge = best_edge[v];
                    }
                }
            }

            // Compute delta3: half the minimum slack on any edge between a
            // pair of S-blossoms
            for blossom in 0..2 * num_nodes {
                if blossom_parents[blossom].is_none()
                    && labels[blossom] == Some(1)
                    && best_edge[blossom].is_some()
                {
                    let kslack =
                        slack(best_edge[blossom].unwrap(), &dual_var, &edges);
                    assert!(kslack % 2 == 0);
                    let d = Some(kslack / 2);
                    if delta_type == -1 || d < delta {
                        delta = d;
                        delta_type = 3;
                        delta_edge = best_edge[blossom];
                    }
                }
            }

            // Compute delta4: minimum z variable of any T-blossom
            for blossom in num_nodes..2 * num_nodes {
                if blossom_base[blossom].is_some()
                    && blossom_parents[blossom].is_none()
                    && labels[blossom] == Some(2)
                    && (delta_type == -1 || dual_var[blossom] < delta.unwrap())
                {
                    delta = Some(dual_var[blossom]);
                    delta_type = 4;
                    delta_blossom = Some(blossom);
                }
            }
            if delta_type == -1 {
                // No further improvement possible; max-cardinality optimum
                // reached. Do a final delta update to make the optimum
                // verifyable
                assert!(max_cardinality);
                delta_type = 1;
                delta =
                    Some(max(0, *dual_var[..num_nodes].iter().min().unwrap()));
            }

            // Update dual variables according to delta.
            for v in 0..num_nodes {
                if labels[in_blossoms[v]] == Some(1) {
                    // S-vertex: 2*u = 2*u - 2*delta
                    dual_var[v] -= delta.unwrap();
                } else if labels[in_blossoms[v]] == Some(2) {
                    // T-vertex: 2*u = 2*u + 2*delta
                    dual_var[v] += delta.unwrap();
                }
            }
            for b in num_nodes..2 * num_nodes {
                if blossom_base[b].is_some() && blossom_parents[b].is_none() {
                    if labels[b] == Some(1) {
                        // top-level S-blossom: z = z + 2*delta
                        dual_var[b] += delta.unwrap();
                    } else if labels[b] == Some(2) {
                        // top-level T-blossom: z = z - 2*delta
                        dual_var[b] -= delta.unwrap();
                    }
                }
            }
            // Take action at the point where minimum delta occured.
            if delta_type == 1 {
                // No further improvement possible; optimum reached
                break;
            } else if delta_type == 2 {
                // Use the least-slack edge to continue the search.
                allowed_edge[delta_edge.unwrap()] = true;
                let (mut i, mut j, _weight) = edges[delta_edge.unwrap()];
                if labels[in_blossoms[i]] == Some(0) {
                    mem::swap(&mut i, &mut j);
                }
                assert!(labels[in_blossoms[i]] == Some(1));
                queue.push(i);
            } else if delta_type == 3 {
                // Use the least-slack edge to continue the search.
                allowed_edge[delta_edge.unwrap()] = true;
                let (i, _j, _weight) = edges[delta_edge.unwrap()];
                assert!(labels[in_blossoms[i]] == Some(1));
                queue.push(i);
            } else if delta_type == 4 {
                // Expand the least-z blossom
                expand_blossom(
                    delta_blossom.unwrap(),
                    false,
                    num_nodes,
                    &mut blossom_children,
                    &mut blossom_parents,
                    &mut in_blossoms,
                    &dual_var,
                    &mut labels,
                    &mut label_ends,
                    &mut best_edge,
                    &mut queue,
                    &mut blossom_base,
                    &endpoints,
                    &mate,
                    &mut blossom_endpoints,
                    &mut allowed_edge,
                    &mut unused_blossoms,
                )?;
            }
            // end of this substage
        }
        // Stop when no more augment paths can be found
        if !augmented {
            break;
        }

        // End of a stage; expand all S-blossoms which have a dual_var == 0
        for blossom in num_nodes..2 * num_nodes {
            if blossom_parents[blossom].is_none()
                && blossom_base[blossom].is_some()
                && labels[blossom] == Some(1)
                && dual_var[blossom] == 0
            {
                expand_blossom(
                    blossom,
                    true,
                    num_nodes,
                    &mut blossom_children,
                    &mut blossom_parents,
                    &mut in_blossoms,
                    &dual_var,
                    &mut labels,
                    &mut label_ends,
                    &mut best_edge,
                    &mut queue,
                    &mut blossom_base,
                    &endpoints,
                    &mate,
                    &mut blossom_endpoints,
                    &mut allowed_edge,
                    &mut unused_blossoms,
                )?;
            }
        }
    }
    if verify_optimum_flag {
        match panic::catch_unwind(|| {
            verify_optimum(
                max_cardinality,
                num_nodes,
                num_edges,
                &edges,
                &endpoints,
                &dual_var,
                &blossom_parents,
                &blossom_endpoints,
                &blossom_base,
                &mate,
            );
        }) {
            Ok(_) => (),
            Err(_) => {
                return Err(PyException::new_err(
                    "The found solution was not optimal",
                ))
            }
        }
    }

    // Transform mate[] such that mate[v] is the vertex to which v is paired
    // Also handle holes in node indices from PyGraph node removals by mapping
    // linear index to node index.
    let mut seen: HashSet<(usize, usize)> =
        HashSet::with_capacity(2 * num_nodes);
    let node_list: Vec<NodeIndex> = graph.graph.node_indices().collect();
    for (index, node) in mate.iter() {
        let tmp = (
            node_list[*index].index(),
            node_list[endpoints[*node]].index(),
        );
        let rev_tmp = (
            node_list[endpoints[*node]].index(),
            node_list[*index].index(),
        );
        if !seen.contains(&tmp) && !seen.contains(&rev_tmp) {
            out_set.insert(tmp);
            seen.insert(tmp);
            seen.insert(rev_tmp);
        }
    }
    Ok(out_set)
}
