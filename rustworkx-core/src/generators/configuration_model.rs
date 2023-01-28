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

use crate::generators::InvalidInputError;
use num_traits::{Num, ToPrimitive};
use petgraph::data::{Build, Create};
use petgraph::visit::{Data, GraphProp, NodeIndexable};
use petgraph::Undirected;
use rand::distributions::Distribution;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::iter::repeat;

/// A degree sequence is a vector of integers that sum up to an even number.
///
/// A graph with a given degree sequence can be generated with
/// [configuration_model](fn.configuration_model.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegreeSequence {
    values: Vec<usize>,
}

impl DegreeSequence {
    /// Returns a new degree sequence.
    ///
    /// Arguments:
    /// * `vec`: A vector of integers that sum up to an even number.
    pub fn new(vec: Vec<usize>) -> Result<Self, DegreeSequenceError> {
        let sum: usize = vec.iter().sum();
        if sum % 2 != 0 {
            return Err(DegreeSequenceError::OddSum);
        }
        Ok(Self { values: vec })
    }

    /// Generates a random degree sequence with the given length and values
    /// sampled from the given distribution.
    ///
    /// Arguments:
    /// * `rng`: A random number generator.
    /// * `length`: The length of the degree sequence.
    /// * `distribution`: A distribution that samples values that are convertible to `usize`.
    ///
    /// NOTE: If the sampled values are floating point numbers, they will be truncated.
    ///
    /// # Example
    /// ```rust
    /// use rand::SeedableRng;
    /// use rustworkx_core::petgraph::graph::UnGraph;
    /// use rustworkx_core::generators::DegreeSequence;
    /// use rand_distr::Poisson;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// let poisson = Poisson::new(2.0).unwrap();
    /// let degree_sequence = DegreeSequence::from_distribution(&mut rng, 10, &poisson).unwrap();
    /// let expected_sequence = vec![3, 0, 4, 0, 3, 1, 0, 1, 4, 2];
    /// assert_eq!(*degree_sequence, expected_sequence);
    /// ```
    pub fn from_distribution<R: Rng + ?Sized, I: Num + PartialOrd + ToPrimitive>(
        rng: &mut R,
        length: usize,
        distribution: &impl Distribution<I>,
    ) -> Result<Self, DegreeSequenceError> {
        let mut degree_sequence: Vec<usize> = Vec::with_capacity(length);
        let mut sum: usize = 0;
        for value in distribution.sample_iter(&mut *rng).take(length) {
            if value < I::zero() {
                return Err(DegreeSequenceError::NegativeDegree);
            }
            let value = match value.to_usize() {
                Some(value) => value,
                None => return Err(DegreeSequenceError::Conversion),
            };
            degree_sequence.push(value);
            sum += value;
        }
        if sum % 2 == 1 {
            let index = rng.gen_range(0..length);
            degree_sequence[index] += 1;
        }
        Ok(Self {
            values: degree_sequence,
        })
    }
}

impl std::ops::Deref for DegreeSequence {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// An error that can occur when generating a degree sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeSequenceError {
    NegativeDegree,
    OddSum,
    Conversion,
}

impl std::fmt::Display for DegreeSequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            DegreeSequenceError::NegativeDegree => "Sampled value was negative",
            DegreeSequenceError::OddSum => "Degree sequence has an odd sum",
            DegreeSequenceError::Conversion => "Sampled value could not be converted to usize",
        })
    }
}

impl std::error::Error for DegreeSequenceError {}

/// Generates a random undirected graph with the given degree sequence.
///
/// Arguments:
///
/// * `rng`: A random number generator.
/// * `degree_sequence`: A degree sequence.
/// * `weights` - A `Vec` of node weight objects. If specified, the length must be
///    equal to the length of the `degree_sequence`.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified,
///     as the weights from that argument will be used instead.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// The algorithm is based on "M.E.J. Newman, "The structure and function of complex networks",
///  SIAM REVIEW 45-2, pp 167-256, 2003.".
///
/// The graph construction process might attempt to insert parallel edges and self-loops.
/// If the graph type does not support either of them, the resulting graph might not have the
/// exact degree sequence. However, the probability of parallel edges and self-loops tends to
/// converge towards zero as the number of nodes increases.
///
/// Time complexity: **O(n + m)**
///
/// # Example
///
/// Approximating Erdős-Rényi G(n, p) model by using a Poisson degree distribution:
/// ```rust
/// use rustworkx_core::generators::{undirected_configuration_model, DegreeSequence};
/// use petgraph::graph::UnGraph;
/// use rand_distr::Poisson;
///
/// let mut rng = rand::thread_rng();
/// let n = 200;
/// let p = 0.01;
/// let lambda = n as f64 * p;
/// let poisson = Poisson::new(lambda).unwrap();
/// let degree_sequence = DegreeSequence::from_distribution(&mut rng, n, &poisson).unwrap();
/// let graph: UnGraph<(), ()> = undirected_configuration_model(&mut rng, &degree_sequence, None, || (), || ()).unwrap();
/// assert_eq!(graph.node_count(), 200);
/// ```
pub fn undirected_configuration_model<G, T, F, H, M, R>(
    rng: &mut R,
    degree_sequence: &DegreeSequence,
    weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: GraphProp<EdgeType = Undirected>
        + Build
        + Create
        + Data<NodeWeight = T, EdgeWeight = M>
        + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    R: Rng + ?Sized,
{
    let num_nodes = degree_sequence.len();

    let mut graph = G::with_capacity(num_nodes, num_nodes);
    let mut nodes: Vec<G::NodeId> = Vec::with_capacity(num_nodes);

    match weights {
        Some(weights) => {
            if weights.len() != num_nodes {
                return Err(InvalidInputError {});
            }
            for weight in weights {
                nodes.push(graph.add_node(weight));
            }
        }
        None => {
            for _ in 0..num_nodes {
                nodes.push(graph.add_node(default_node_weight()));
            }
        }
    };

    let mut stubs: Vec<G::NodeId> = nodes
        .into_iter()
        .zip(degree_sequence.iter())
        .flat_map(|(node, degree)| repeat(node).take(*degree))
        .collect();
    stubs.shuffle(rng);

    let (sources, targets) = stubs.split_at(stubs.len() / 2);
    for (&source, &target) in sources.iter().zip(targets) {
        graph.add_edge(source, target, default_edge_weight());
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph::visit::EdgeRef;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_degree_sequence() {
        let degree_sequence = DegreeSequence::new(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(*degree_sequence, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_degree_sequence_empty() {
        let degree_sequence = DegreeSequence::new(vec![]).unwrap();
        assert_eq!(*degree_sequence, vec![]);
    }

    #[test]
    fn test_degree_sequence_odd_sum() {
        let result = DegreeSequence::new(vec![1, 2, 3, 5]);
        assert_eq!(result, Err(DegreeSequenceError::OddSum));
    }

    #[test]
    fn test_degree_sequence_from_discrete_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let poisson = rand_distr::Poisson::new(2.0).unwrap();
        let degree_sequence = DegreeSequence::from_distribution(&mut rng, 10, &poisson).unwrap();
        assert_eq!(*degree_sequence, vec![3, 0, 4, 0, 3, 1, 0, 1, 4, 2]);
    }

    #[test]
    fn test_degree_sequence_from_continuous_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let exp = rand_distr::Exp::new(0.5).unwrap();
        let degree_sequence = DegreeSequence::from_distribution(&mut rng, 10, &exp).unwrap();
        assert_eq!(*degree_sequence, vec![1, 1, 0, 0, 0, 1, 0, 4, 0, 1]);
    }

    #[test]
    fn test_degree_sequence_from_distribution_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = rand_distr::Uniform::new(0, 10);
        let degree_sequence = DegreeSequence::from_distribution(&mut rng, 0, &uniform).unwrap();
        assert_eq!(*degree_sequence, vec![]);
    }

    #[test]
    fn test_degree_sequence_from_distribution_conversion_error() {
        let mut rng = StdRng::seed_from_u64(42);
        let exp = rand_distr::Exp::new(0.0).unwrap(); // samples always yield infinity
        let result = DegreeSequence::from_distribution(&mut rng, 1, &exp);
        assert_eq!(result, Err(DegreeSequenceError::Conversion));
    }

    #[test]
    fn test_degree_sequence_from_distribution_negative_degree() {
        let mut rng = StdRng::seed_from_u64(42);
        let poisson = rand_distr::Uniform::new(-1, 0);
        let result = DegreeSequence::from_distribution(&mut rng, 10, &poisson);
        assert_eq!(result, Err(DegreeSequenceError::NegativeDegree));
    }

    #[test]
    fn test_configuration_model() {
        let mut rng = StdRng::seed_from_u64(42);
        let degree_sequence = DegreeSequence::new(vec![1, 2, 3, 4]).unwrap();
        let graph: UnGraph<(), ()> =
            undirected_configuration_model(&mut rng, &degree_sequence, None, || (), || ()).unwrap();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 5);
        assert_eq!(
            vec![(3, 3), (2, 3), (3, 2), (2, 1), (0, 1)],
            graph
                .edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_configuration_model_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let degree_sequence = DegreeSequence::new(vec![]).unwrap();
        let graph: UnGraph<(), ()> =
            undirected_configuration_model(&mut rng, &degree_sequence, None, || (), || ()).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_configuration_model_with_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let degree_sequence = DegreeSequence::new(vec![1, 2, 3, 4]).unwrap();
        let graph: UnGraph<usize, ()> = undirected_configuration_model(
            &mut rng,
            &degree_sequence,
            Some(vec![0, 1, 2, 3]),
            || 1,
            || (),
        )
        .unwrap();
        assert_eq!(
            vec![0, 1, 2, 3],
            graph.node_weights().copied().collect::<Vec<usize>>(),
        );
    }

    #[test]
    fn test_configuration_model_degree_sequence_and_weights_have_different_lengths() {
        let mut rng = StdRng::seed_from_u64(42);
        let degree_sequence = DegreeSequence::new(vec![1, 2, 3, 4]).unwrap();
        let result: Result<UnGraph<usize, ()>, _> = undirected_configuration_model(
            &mut rng,
            &degree_sequence,
            Some(vec![0, 1, 2]),
            || 1,
            || (),
        );
        match result {
            Ok(_) => panic!("Expected error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        }
    }
}
