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

use std::iter;

pub fn pairwise<I>(right: I) -> impl Iterator<Item = (Option<I::Item>, I::Item)>
where
    I: IntoIterator + Clone,
{
    let left = iter::once(None).chain(right.clone().into_iter().map(Some));
    left.zip(right)
}
