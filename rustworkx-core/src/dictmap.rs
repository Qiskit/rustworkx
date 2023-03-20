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

//! This module contains the [`DictMap`] type alias which is a combination of
//! [`IndexMap`] and [`AHash`].
//!
//! It is used as a return type for rustworkx for compatibility
//! with Python's dict which preserves insertion order.
//!
//! [`AHash`]: https://crates.io/crates/ahash

use indexmap::IndexMap;

/// Convenient alias to build an [`IndexMap`] using a custom hasher.
/// For the moment, we use ahash which is the default hasher
/// for [`HashMap`], another hashmap we use.
///
/// [`HashMap`]: https://docs.rs/hashbrown/0.11.2/hashbrown/hash_map/struct.HashMap.html
pub type DictMap<K, V> = IndexMap<K, V, ahash::RandomState>;

pub trait InitWithHasher {
    fn new() -> Self
    where
        Self: Sized;

    fn with_capacity(n: usize) -> Self
    where
        Self: Sized;
}

impl<K, V> InitWithHasher for DictMap<K, V> {
    #[inline]
    fn new() -> Self {
        indexmap::IndexMap::with_capacity_and_hasher(0, ahash::RandomState::default())
    }

    #[inline]
    fn with_capacity(n: usize) -> Self {
        indexmap::IndexMap::with_capacity_and_hasher(n, ahash::RandomState::default())
    }
}
