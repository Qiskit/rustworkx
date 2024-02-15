use std::cell::RefCell;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashMap;

use crate::iterators::{NodeIndices, RelationalCoarsestPartition};
use crate::{digraph, Directed, StablePyGraph};
use petgraph::graph;
use petgraph::Direction::{Incoming, Outgoing};

type Block = Vec<graph::NodeIndex>;
type Counterimage = HashMap<graph::NodeIndex, Vec<graph::NodeIndex>>;
type NodeToBlockVec = Vec<Rc<RefCell<FineBlock>>>;
type CoarsePartition = Vec<Rc<CoarseBlock>>;
type FineBlockPointer = Rc<RefCell<FineBlock>>;
type CoarseBlockPointer = Rc<CoarseBlock>;
type CounterimageGrouped = HashMap<Block, CounterImageGroup>;

struct FineBlock {
    values: Block,
    coarse_block_that_supersets_self: Rc<CoarseBlock>,
}

#[derive(Clone)]
struct CoarseBlock {
    values: Block,
    fine_blocks_that_are_subsets_of_self: RefCell<Vec<Rc<RefCell<FineBlock>>>>,
}
impl CoarseBlock {
    fn add_fine_block(&self, fine_block: Rc<RefCell<FineBlock>>) {
        self.fine_blocks_that_are_subsets_of_self
            .borrow_mut()
            .push(fine_block);
    }
    fn remove_fine_block(&self, fine_block: &Rc<RefCell<FineBlock>>) {
        self.fine_blocks_that_are_subsets_of_self
            .borrow_mut()
            .retain(|x| !Rc::ptr_eq(x, fine_block));
    }
    fn fine_block_count(&self) -> usize {
        self.fine_blocks_that_are_subsets_of_self.borrow().len()
    }
}

struct CounterImageGroup {
    block: Rc<RefCell<FineBlock>>,
    subblock: Block,
}

trait HasValues {
    fn values(&self) -> Block;
}

impl HasValues for FineBlockPointer {
    fn values(&self) -> Block {
        (**self).borrow().values.clone()
    }
}
impl HasValues for CoarseBlock {
    fn values(&self) -> Block {
        self.values.clone()
    }
}

fn initialization(
    graph: &StablePyGraph<Directed>,
) -> (
    (FineBlockPointer, FineBlockPointer),
    CoarsePartition,
    NodeToBlockVec,
) {
    let graph_node_indices: Block = graph.node_indices().clone().collect();

    let coarse_initial_block_pointer: Rc<CoarseBlock> = {
        let coarse_initial_block = CoarseBlock {
            values: graph_node_indices.clone(),
            fine_blocks_that_are_subsets_of_self: RefCell::new(vec![]),
        };

        Rc::new(coarse_initial_block)
    };

    let (leaf_node_block_pointer, non_leaf_node_block_pointer) = {
        let (leaf_node_indices, non_leaf_node_indices): (Block, Block) = graph_node_indices
            .clone()
            .into_iter()
            .partition(|x| graph.neighbors_directed(*x, Outgoing).count() == 0);

        let leaf_node_block = FineBlock {
            values: leaf_node_indices,
            coarse_block_that_supersets_self: Rc::clone(&coarse_initial_block_pointer),
        };
        let non_leaf_node_block = FineBlock {
            values: non_leaf_node_indices,
            coarse_block_that_supersets_self: Rc::clone(&coarse_initial_block_pointer),
        };

        (
            Rc::new(RefCell::new(leaf_node_block)),
            Rc::new(RefCell::new(non_leaf_node_block)),
        )
    };

    coarse_initial_block_pointer
        .fine_blocks_that_are_subsets_of_self
        .borrow_mut()
        .extend([
            Rc::clone(&leaf_node_block_pointer),
            Rc::clone(&non_leaf_node_block_pointer),
        ]);

    let node_to_block_vec = (*leaf_node_block_pointer)
        .borrow()
        .values
        .clone()
        .iter()
        .fold(
            vec![
                Rc::clone(&non_leaf_node_block_pointer);
                (*graph_node_indices.last().unwrap()).index() as usize + 1
            ],
            |mut accumulator, value| {
                accumulator[value.index()] = Rc::clone(&leaf_node_block_pointer);
                accumulator
            },
        );

    (
        (leaf_node_block_pointer, non_leaf_node_block_pointer),
        vec![coarse_initial_block_pointer],
        node_to_block_vec,
    )
}

fn build_counterimage<IndexHolder: HasValues>(
    graph: &StablePyGraph<Directed>,
    fine_block: IndexHolder,
) -> Counterimage {
    let mut counterimage = HashMap::new();
    fine_block.values().iter().for_each(|node_index_pointer| {
        counterimage.insert(
            *node_index_pointer,
            graph
                .neighbors_directed(*node_index_pointer, Incoming)
                .collect::<Vec<graph::NodeIndex>>(),
        );
    });

    counterimage
}

fn group_by_counterimage(
    counterimage: Counterimage,
    node_to_block: &NodeToBlockVec,
) -> CounterimageGrouped {
    let mut counterimage_grouped: CounterimageGrouped = HashMap::new();

    for incoming_neighbor_group in counterimage.values() {
        for node in incoming_neighbor_group {
            let block = Rc::clone(&node_to_block[node.index()]);

            let key = (*block).borrow().values.clone();
            match counterimage_grouped.contains_key(&key) {
                true => {
                    counterimage_grouped
                        .get_mut(&key)
                        .unwrap()
                        .subblock
                        .push(*node);
                }
                false => {
                    counterimage_grouped.insert(
                        key,
                        CounterImageGroup {
                            block: Rc::clone(&block),
                            subblock: Vec::from([*node]),
                        },
                    );
                }
            }
        }
    }

    counterimage_grouped
}

fn split_blocks_with_grouped_counterimage(
    mut counterimage_grouped: CounterimageGrouped,
    node_to_block_vec: &mut NodeToBlockVec,
) -> (
    (Vec<FineBlockPointer>, Vec<FineBlockPointer>),
    Vec<CoarseBlockPointer>,
) {
    let mut all_new_fine_blocks: Vec<Rc<RefCell<FineBlock>>> = vec![];
    let mut all_removed_fine_blocks: Vec<Rc<RefCell<FineBlock>>> = vec![];
    let mut new_compound_coarse_blocks: Vec<Rc<CoarseBlock>> = vec![];

    for (block, counter_image_group) in counterimage_grouped.iter_mut() {
        let borrowed_coarse_block = Rc::clone(
            &(*counter_image_group.block)
                .borrow()
                .coarse_block_that_supersets_self,
        );

        let proper_subblock = {
            let fine_block = FineBlock {
                values: counter_image_group.subblock.clone(),
                coarse_block_that_supersets_self: Rc::clone(&borrowed_coarse_block),
            };

            Rc::new(RefCell::new(fine_block))
        };
        let prior_count = borrowed_coarse_block.fine_block_count();
        borrowed_coarse_block.add_fine_block(Rc::clone(&proper_subblock));

        if prior_count == 1 {
            new_compound_coarse_blocks.push(Rc::clone(&borrowed_coarse_block));
        }

        for node_index in counter_image_group.subblock.iter() {
            node_to_block_vec[node_index.index()] = Rc::clone(&proper_subblock);
        }

        // subtract subblock from block
        (*counter_image_group.block).borrow_mut().values = block
            .iter()
            .filter(|x| !(*proper_subblock).borrow().values.contains(x))
            .copied()
            .collect();

        if (*counter_image_group.block).borrow().values.is_empty() {
            borrowed_coarse_block.remove_fine_block(&counter_image_group.block);
            all_removed_fine_blocks.push(Rc::clone(&counter_image_group.block));
        }
        all_new_fine_blocks.push(Rc::clone(&proper_subblock));
    }
    (
        (all_new_fine_blocks, all_removed_fine_blocks),
        new_compound_coarse_blocks,
    )
}

fn maximum_bisimulation(graph: &StablePyGraph<Directed>) -> Vec<Block> {
    let (fine_block_tuple, initial_coarse_partition, mut node_to_block_vec) = initialization(graph);

    let mut queue: CoarsePartition = initial_coarse_partition;
    let mut all_fine_blocks = vec![fine_block_tuple.0, fine_block_tuple.1];

    while !queue.is_empty() {
        let (smaller_component, simple_splitter_block) = {
            let splitter_block = queue.pop().unwrap();
            let mut fine_blocks_in_splitter_block = splitter_block
                .fine_blocks_that_are_subsets_of_self
                .borrow()
                .clone();

            let smaller_component_index = fine_blocks_in_splitter_block
                .iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| {
                    (***x)
                        .borrow()
                        .values
                        .len()
                        .cmp(&(***y).borrow().values.len())
                })
                .map(|(index, _)| index)
                .unwrap();

            let smaller_component = fine_blocks_in_splitter_block.remove(smaller_component_index);

            let simple_splitter_block_values: Block = splitter_block
                .values
                .clone()
                .iter()
                .filter(|x| !(*smaller_component).borrow().values.contains(x))
                .copied()
                .collect();

            let simple_splitter_block = CoarseBlock {
                values: simple_splitter_block_values,
                fine_blocks_that_are_subsets_of_self: RefCell::new(fine_blocks_in_splitter_block),
            };
            let simple_splitter_block_pointer = Rc::new(simple_splitter_block);

            if simple_splitter_block_pointer
                .fine_blocks_that_are_subsets_of_self
                .borrow()
                .len()
                > 1
            {
                queue.push(Rc::clone(&simple_splitter_block_pointer));
            }

            (smaller_component, simple_splitter_block_pointer)
        };
        simple_splitter_block
            .fine_blocks_that_are_subsets_of_self
            .borrow()
            .iter()
            .for_each(|x| {
                (*x).borrow_mut().coarse_block_that_supersets_self =
                    Rc::clone(&simple_splitter_block);
            });

        let mut counterimage = build_counterimage(graph, smaller_component);

        let counterimage_group = group_by_counterimage(counterimage.clone(), &node_to_block_vec);
        let ((new_fine_blocks, removeable_fine_blocks), coarse_block_that_are_now_compound) =
            split_blocks_with_grouped_counterimage(counterimage_group, &mut node_to_block_vec);

        all_fine_blocks.extend(new_fine_blocks);
        all_fine_blocks.retain(|x| !removeable_fine_blocks.iter().any(|y| Rc::ptr_eq(x, y)));
        queue.extend(coarse_block_that_are_now_compound);

        {
            let counterimage_splitter_complement =
                build_counterimage(graph, (*simple_splitter_block).clone());

            counterimage_splitter_complement.keys().for_each(|node| {
                counterimage.remove(node);
            });
        }

        let counterimage_group = group_by_counterimage(counterimage, &node_to_block_vec);
        let ((new_fine_blocks, removeable_fine_blocks), coarse_block_that_are_now_compound) =
            split_blocks_with_grouped_counterimage(counterimage_group, &mut node_to_block_vec);

        all_fine_blocks.extend(new_fine_blocks);
        all_fine_blocks.retain(|x| !removeable_fine_blocks.iter().any(|y| Rc::ptr_eq(x, y)));
        queue.extend(coarse_block_that_are_now_compound);
    }

    all_fine_blocks
        .iter()
        .map(|x| (**x).borrow().values.clone())
        .filter(|x| !x.is_empty()) // remove leaf block when there are no leaves
        .collect()
}

/// Calculates the relational coarsest partition otherwise known as the maximum bisimulation relation
/// using the Paige-Tarjan algorithm described in "Three partition refinement algorithms".
///
/// :param PyDiGraph graph: The graph to find the maximum bisimulation relation for
///
/// :returns: The relational coarsest partition which is an iterator of :class:`~rustworkx.NodeIndices`.
///
/// :rtype: :class:`~rustworkx.RelationalCoarsestPartition`
#[pyfunction()]
#[pyo3(text_signature = "(graph)")]
pub fn digraph_maximum_bisimulation(
    _py: Python,
    graph: &digraph::PyDiGraph,
) -> RelationalCoarsestPartition {
    if graph.graph.node_count() == 0 {
        return RelationalCoarsestPartition { partition: vec![] };
    }
    let result = maximum_bisimulation(&graph.graph)
        .into_iter()
        .map(|block| NodeIndices {
            nodes: block
                .into_iter()
                .map(|node| node.index())
                .collect::<Vec<usize>>(),
        });
    RelationalCoarsestPartition {
        partition: result.collect(),
    }
}
