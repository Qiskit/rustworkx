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

extern crate rustworkx_core;

use rand::{thread_rng, Rng};
use rustworkx_core::disjoint_set::DisjointSet;

use std::collections::HashSet;
use std::iter::FromIterator;

#[test]
fn dsu_test() {
    let n = 8;
    let mut u = DisjointSet::new(n);
    for i in 0..n {
        assert_eq!(u.find(i), i);
        assert_eq!(u.find_mut(i), i);
        assert!(!u.union(i, i));
    }

    u.union(0, 1);
    assert_eq!(u.find(0), u.find(1));
    u.union(1, 3);
    u.union(1, 4);
    u.union(4, 7);
    assert_eq!(u.find(0), u.find(3));
    assert_eq!(u.find(1), u.find(3));
    assert!(u.find(0) != u.find(2));
    assert_eq!(u.find(7), u.find(0));
    u.union(5, 6);
    assert_eq!(u.find(6), u.find(5));
    assert!(u.find(6) != u.find(7));

    // check that there are now 3 disjoint sets
    let set = (0..n).map(|i| u.find(i)).collect::<HashSet<_>>();
    assert_eq!(set.len(), 3);
}

#[test]
fn dsu_set_members_test() {
    let n = 8;
    let mut u = DisjointSet::new(n);
    u.union(0, 1);
    u.union(1, 3);
    u.union(1, 4);
    u.union(4, 7);
    u.union(5, 6);

    // check the members of each set
    let sets = (0..n)
        .map(|i| HashSet::<usize>::from_iter(u.set(i)))
        .collect::<Vec<_>>();
    assert_eq!(sets[0], HashSet::from_iter([0, 1, 3, 4, 7]));
    assert_eq!(sets[1], HashSet::from_iter([0, 1, 3, 4, 7]));
    assert_eq!(sets[2], HashSet::from_iter([2]));
    assert_eq!(sets[3], HashSet::from_iter([0, 1, 3, 4, 7]));
    assert_eq!(sets[4], HashSet::from_iter([0, 1, 3, 4, 7]));
    assert_eq!(sets[5], HashSet::from_iter([5, 6]));
    assert_eq!(sets[6], HashSet::from_iter([5, 6]));
    assert_eq!(sets[7], HashSet::from_iter([0, 1, 3, 4, 7]));
}

#[test]
fn dsu_rand_test() {
    let n = 1 << 14;
    let mut rng = thread_rng();
    let mut u = DisjointSet::new(n);
    for _ in 0..100 {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        let ar = u.find(a);
        let br = u.find(b);
        assert_eq!(ar != br, u.union(a, b));
    }
}

#[test]
fn dsu_rand_u8_test() {
    let n = 256;
    let mut rng = thread_rng();
    let mut u = DisjointSet::<u8>::new(n);
    for _ in 0..(n * 8) {
        let a = rng.gen();
        let b = rng.gen();
        let ar = u.find(a);
        let br = u.find(b);
        assert_eq!(ar != br, u.union(a, b));
    }
}
