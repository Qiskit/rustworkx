mod metrics;
pub use metrics::{modularity, Modularity};

mod louvain;
pub use louvain::louvain_communities;

mod utils;

use core::fmt;
use std::error::Error;

#[derive(Debug, PartialEq, Eq)]
pub struct NotAPartitionError;
impl Error for NotAPartitionError {}
impl fmt::Display for NotAPartitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The input subsets do not form a partition of the input graph."
        )
    }
}
