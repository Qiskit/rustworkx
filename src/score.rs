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
#![allow(clippy::derive_partial_eq_without_eq)]

use std::cmp::Ordering;
use std::ops::{Add, AddAssign};

use num_traits::Zero;

/// `Score<K>` holds a score `K` for use with a `PriorityHeap`.
///
/// **Note:** `Score` implements a total order (`Ord`), so that it is
/// possible to use float types as scores.
#[derive(Clone, Copy, PartialEq)]
pub struct Score<K>(pub K);

impl<K: Add<Output = K>> Add for Score<K> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Score(self.0 + rhs.0)
    }
}

impl<K: AddAssign> AddAssign for Score<K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl<K: Zero> Zero for Score<K> {
    fn zero() -> Self {
        Score(K::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<K: PartialOrd> PartialOrd for Score<K> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: PartialOrd> Eq for Score<K> {}
impl<K: PartialOrd> Ord for Score<K> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Order NaN less, so that it is last in the Score.
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}
