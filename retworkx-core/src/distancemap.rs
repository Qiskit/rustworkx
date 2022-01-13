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

//! This module contains the [`DistanceMap`] trait which is used in
//! [`shortest_path`].
//!
//! The trait allows the shortest path functions to support multiple
//! return types.

use std::hash::Hash;

use petgraph::graph::IndexType;

use crate::dictmap::*;
use hashbrown::HashMap;

pub trait DistanceMap<K, V> {
    fn build(num_elements: usize) -> Self;
    fn get_item(&self, pos: K) -> Option<&V>;
    fn put_item(&mut self, pos: K, val: V);
}

impl<K: IndexType, V: Clone> DistanceMap<K, V> for Vec<Option<V>> {
    #[inline]
    fn build(num_elements: usize) -> Self {
        vec![None; num_elements]
    }

    #[inline]
    fn get_item(&self, pos: K) -> Option<&V> {
        self[pos.index()].as_ref()
    }

    #[inline]
    fn put_item(&mut self, pos: K, val: V) {
        self[pos.index()] = Some(val);
    }
}

impl<K: Eq + Hash, V: Clone> DistanceMap<K, V> for DictMap<K, V> {
    #[inline]
    fn build(_num_elements: usize) -> Self {
        DictMap::<K, V>::default()
    }

    #[inline]
    fn get_item(&self, pos: K) -> Option<&V> {
        self.get(&pos)
    }

    #[inline]
    fn put_item(&mut self, pos: K, val: V) {
        self.insert(pos, val);
    }
}

impl<K: Eq + Hash, V: Clone> DistanceMap<K, V> for HashMap<K, V> {
    #[inline]
    fn build(_num_elements: usize) -> Self {
        HashMap::<K, V>::default()
    }

    #[inline]
    fn get_item(&self, pos: K) -> Option<&V> {
        self.get(&pos)
    }

    #[inline]
    fn put_item(&mut self, pos: K, val: V) {
        self.insert(pos, val);
    }
}
