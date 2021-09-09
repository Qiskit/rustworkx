// Convenient alias to build IndexMap using a custom hasher.
// For the moment, we use ahash which is the default hasher
// for hashbrown::HashMap, another hashmap we use

pub type DictMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

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
        indexmap::IndexMap::with_capacity_and_hasher(
            0,
            ahash::RandomState::default(),
        )
    }

    #[inline]
    fn with_capacity(n: usize) -> Self {
        indexmap::IndexMap::with_capacity_and_hasher(
            n,
            ahash::RandomState::default(),
        )
    }
}
