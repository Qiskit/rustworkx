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

//! Module for graphs with vertices embedded in a metric space.

mod distances;
mod greedy_routing;

pub use distances::{
    IncompatiblePointsError, angular_distance, euclidean_distance, hyperboloid_hyperbolic_distance,
    lp_distance, maximum_distance, polar_hyperbolic_distance,
};
pub use greedy_routing::{NodeNotReachedError, greedy_routing, greedy_routing_success_rate};
