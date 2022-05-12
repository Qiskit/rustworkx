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

use petgraph::EdgeType;

use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;

pub fn random_layout<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let random_tuple: [f64; 2] = rng.gen();
                match center {
                    Some(center) => (
                        n.index(),
                        [random_tuple[0] + center[0], random_tuple[1] + center[1]],
                    ),
                    None => (n.index(), random_tuple),
                }
            })
            .collect(),
    }
}
