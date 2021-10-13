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

//! `DisjointSet<K>` is a disjoint-set data structure.

use petgraph::graph::IndexType;
use std::cmp::Ordering;

/// `DisjointSet<K>` is a disjoint-set data structure. It tracks set membership of *n* elements
/// indexed from *0* to *n - 1*. The scalar type is `K` which must be an unsigned integer type.
///
/// <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>
///
/// Too awesome not to quote:
///
/// “The amortized time per operation is **O(α(n))** where **α(n)** is the
/// inverse of **f(x) = A(x, x)** with **A** being the extremely fast-growing Ackermann function.”
///
/// The implementation is taken from petgraph [`UnionFind`](https://docs.rs/petgraph/0.6.0/petgraph/unionfind/struct.UnionFind.html)
/// with an extra field that allows us to efficiently retrieve all members of a set.
#[derive(Debug, Clone)]
pub struct DisjointSet<K> {
    /// For element at index *i*, store the index of its parent; the representative itself
    /// stores its own index. This forms equivalence classes which are the disjoint sets, each
    /// with a unique representative.
    parent: Vec<K>,
    /// It is a balancing tree structure,
    /// so the ranks are logarithmic in the size of the container -- a byte is more than enough.
    ///
    /// Rank is separated out both to save space and to save cache in when searching in the parent
    /// vector.
    rank: Vec<u8>,
    // For element at index *i*, store the next element in the same set. The items in a set form a
    // circular linked list, so we can efficiently traverse all of them.
    next: Vec<K>,
}

#[inline]
unsafe fn get_unchecked<K>(xs: &[K], index: usize) -> &K {
    debug_assert!(index < xs.len());
    xs.get_unchecked(index)
}

#[inline]
unsafe fn get_unchecked_mut<K>(xs: &mut [K], index: usize) -> &mut K {
    debug_assert!(index < xs.len());
    xs.get_unchecked_mut(index)
}

impl<K> DisjointSet<K>
where
    K: IndexType,
{
    /// Create a new `DisjointSet` of `n` disjoint sets.
    pub fn new(n: usize) -> Self {
        let rank = vec![0; n];
        let parent = (0..n).map(K::new).collect::<Vec<K>>();
        let next = (0..n).map(K::new).collect::<Vec<K>>();

        DisjointSet { parent, rank, next }
    }

    /// Return the representative for `x`.
    ///
    /// **Panics** if `x` is out of bounds.
    pub fn find(&self, x: K) -> K {
        assert!(x.index() < self.parent.len());
        unsafe {
            let mut x = x;
            loop {
                // Use unchecked indexing because we can trust the internal set ids.
                let xparent = *get_unchecked(&self.parent, x.index());
                if xparent == x {
                    break;
                }
                x = xparent;
            }
            x
        }
    }

    /// Return the representative for `x`.
    ///
    /// Write back the found representative, flattening the internal
    /// datastructure in the process and quicken future lookups.
    ///
    /// **Panics** if `x` is out of bounds.
    pub fn find_mut(&mut self, x: K) -> K {
        assert!(x.index() < self.parent.len());
        unsafe { self.find_mut_recursive(x) }
    }

    unsafe fn find_mut_recursive(&mut self, mut x: K) -> K {
        let mut parent = *get_unchecked(&self.parent, x.index());
        while parent != x {
            let grandparent = *get_unchecked(&self.parent, parent.index());
            *get_unchecked_mut(&mut self.parent, x.index()) = grandparent;
            x = parent;
            parent = grandparent;
        }
        x
    }

    /// Unify the two sets containing `x` and `y`.
    ///
    /// Return `false` if the sets were already the same, `true` if they were unified.
    ///
    /// **Panics** if `x` or `y` is out of bounds.
    pub fn union(&mut self, x: K, y: K) -> bool {
        if x == y {
            return false;
        }
        let xrep = self.find_mut(x);
        let yrep = self.find_mut(y);

        if xrep == yrep {
            return false;
        }

        let xrepu = xrep.index();
        let yrepu = yrep.index();
        let xrank = self.rank[xrepu];
        let yrank = self.rank[yrepu];

        // The rank corresponds roughly to the depth of the treeset, so put the
        // smaller set below the larger
        match xrank.cmp(&yrank) {
            Ordering::Less => self.parent[xrepu] = yrep,
            Ordering::Greater => self.parent[yrepu] = xrep,
            Ordering::Equal => {
                self.parent[yrepu] = xrep;
                self.rank[xrepu] += 1;
            }
        }

        self.next.swap(xrepu, yrepu);
        true
    }

    /// Reports all elements in the same set as `x`.
    ///
    /// **Panics** if `x` is out of bounds.
    pub fn set(&self, x: K) -> Vec<K> {
        assert!(x.index() < self.next.len());
        unsafe {
            let mut cur = x;
            let mut members = Vec::new();
            loop {
                members.push(cur);
                // Use unchecked indexing because we can trust the internal set ids.
                cur = *self.next.get_unchecked(cur.index());
                if cur == x {
                    break;
                }
            }
            members
        }
    }

    /// Return `true` if `x` is its own parent.
    pub fn is_root(&self, x: K) -> bool {
        assert!(x.index() < self.parent.len());
        self.parent[x.index()] == x
    }
}
