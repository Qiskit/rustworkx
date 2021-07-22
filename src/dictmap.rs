// Convenient alias to build IndexMap using a custom hasher.
// For the moment, we use ahash which is the default hasher
// for hashbrown::HashMap, another hashmap we use

pub type DictMap<K, V> =
    indexmap::IndexMap<K, V, hashbrown::hash_map::DefaultHashBuilder>;

#[macro_export]
macro_rules! _dictmap_new {
    () => {
        indexmap::IndexMap::with_capacity_and_hasher(
            0,
            hashbrown::hash_map::DefaultHashBuilder::default(),
        )
    };
}

// Re-export macro not at the top of crate
pub(crate) use _dictmap_new as dictmap_new;

#[macro_export]
macro_rules! _dictmap_with_capacity {
    ($n:expr) => {
        indexmap::IndexMap::with_capacity_and_hasher(
            $n,
            hashbrown::hash_map::DefaultHashBuilder::default(),
        )
    };
}

// Re-export macro not at the top of crate
pub(crate) use _dictmap_with_capacity as dictmap_with_capacity;
