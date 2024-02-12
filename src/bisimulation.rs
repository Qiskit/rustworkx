use std::borrow::Borrow;
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

type Image = HashMap<graph::NodeIndex, Vec<graph::NodeIndex>>;
type Counterimage = Image;
type CounterimageGrouped = HashMap<Block, CounterImageGroup>;

struct FineBlock {
    values: Block,
    coarse_block_that_supersets_self: Rc<CoarseBlock>,
}

struct CoarseBlock {
    values: Block,
    fine_blocks_that_are_subsets_of_self: RefCell<Vec<Rc<RefCell<FineBlock>>>>,
}

struct CounterImageGroup {
    block: Rc<RefCell<FineBlock>>,
    subblock: Block,
}
impl CounterImageGroup {
    fn subblock_to_fine_block(&self) -> Rc<RefCell<FineBlock>> {}
}

type NodeToBlockVec = Vec<Rc<RefCell<FineBlock>>>;

type CoarsePartition = Vec<Rc<CoarseBlock>>;

fn initialization(
    graph: &StablePyGraph<Directed>,
) -> (
    (Rc<RefCell<FineBlock>>, Rc<RefCell<FineBlock>>),
    CoarsePartition,
    NodeToBlockVec,
) {
    let graph_node_indices: Block = graph.node_indices().clone().into_iter().collect();

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
                graph_node_indices.last().unwrap().clone().index() as usize
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

fn build_counterimage(
    graph: &StablePyGraph<Directed>,
    fine_block: Rc<RefCell<FineBlock>>,
) -> Counterimage {
    let mut counterimage = HashMap::new();
    (*fine_block)
        .borrow()
        .values
        .iter()
        .for_each(|node_index_pointer| {
            counterimage.insert(
                *node_index_pointer,
                graph
                    .neighbors_directed(*node_index_pointer, Incoming)
                    .into_iter()
                    .collect::<Vec<graph::NodeIndex>>(),
            );
        });

    counterimage
}

fn group_by_counterimage(
    counterimage: Counterimage,
    node_to_block: NodeToBlockVec,
) -> CounterimageGrouped {
    let mut counterimage_grouped: CounterimageGrouped = HashMap::new();

    for incoming_neighbor_group in counterimage.values() {
        for node in incoming_neighbor_group {
            let block = node_to_block[node.index()];

            counterimage_grouped
                .entry((*block).borrow().values.clone())
                .or_insert(CounterImageGroup {
                    block: Rc::clone(&block),
                    subblock: Vec::from([*node]),
                })
                .subblock
                .push(*node);
        }
    }

    counterimage_grouped
}

fn split_blocks_with_grouped_counterimage(
    mut counterimage_grouped: CounterimageGrouped,
    node_to_block_vec: &mut NodeToBlockVec,
) -> Vec<(Rc<RefCell<FineBlock>>, Rc<RefCell<FineBlock>>)> {
    let mut changed_blocks = Vec::new();

    for (block, counter_image_group) in counterimage_grouped.iter_mut() {
        let proper_subblock = {
            let fine_block = FineBlock {
                values: counter_image_group.subblock,
                coarse_block_that_supersets_self: Rc::clone(
                    &(*counter_image_group.block)
                        .borrow()
                        .coarse_block_that_supersets_self,
                ),
            };

            Rc::new(RefCell::new(fine_block))
        };

        for node_index in counter_image_group.subblock.into_iter() {
            node_to_block_vec[node_index.index()] = Rc::clone(&proper_subblock);
        }

        // subtract subblock from block
        (*counter_image_group.block).borrow_mut().values = block
            .iter()
            .filter(|x| !(*proper_subblock).borrow().values.contains(x))
            .map(|x| *x)
            .collect();

        changed_blocks.push((
            Rc::clone(&counter_image_group.block),
            Rc::clone(&proper_subblock),
        ));
    }
    changed_blocks
}

fn repartition(
    changed_blocks: Vec<(Rc<Block>, Rc<Block>)>,
    partition: &mut Partition,
    fine_partition_to_coarse: &mut MapFineBlockCoarseBlock,
    coarse_block_makeup: &mut CoarseBlockMakeup,
    queue: &mut CoarsePartition,
) {
    for (block, subblock) in changed_blocks {
        let coarse_block = fine_partition_to_coarse.get(&block).unwrap();
        coarse_block_makeup
            .get_mut(coarse_block)
            .unwrap()
            .push(Rc::clone(&subblock));

        if block.is_empty() {
            coarse_block_makeup.remove(coarse_block);
            partition.remove(&*block);

            fine_partition_to_coarse.remove(&block);
        } else if coarse_block.len() == 2 {
            queue.push(Rc::clone(coarse_block));
        }
        (*partition).insert(*subblock);
    }
}

fn find_smallest_component_block_and_copy(coarse_block_makeup: Vec<Rc<Block>>) -> Block {
    let smallest_component_block_pointer = coarse_block_makeup
        .iter()
        .min_by(|x, y| x.len().cmp(&y.len()))
        .unwrap();

    (**smallest_component_block_pointer).clone()
}

fn maximum_bisimulation(graph: &StablePyGraph<Directed>) -> Partition {
    let (fine_block_tuple, initial_coarse_partition, mut node_to_block_vec) = initialization(graph);

    let mut queue: CoarsePartition = initial_coarse_partition;

    while !queue.is_empty() {
        let (smaller_component, simple_splitter_block) = {
            let splitter_block = queue.pop().unwrap();
            let fine_blocks_in_splitter_block = splitter_block
                .fine_blocks_that_are_subsets_of_self
                .borrow()
                .clone();

            if !(fine_blocks_in_splitter_block.len() == 2) {
                queue.push(splitter_block);
            }

            let smaller_component = fine_blocks_in_splitter_block
                .iter()
                .min_by(|x, y| {
                    (***x)
                        .borrow()
                        .values
                        .len()
                        .cmp(&(***x).borrow().values.len())
                })
                .unwrap()
                .clone();

            let simple_splitter_block_subsets = {
                let t = fine_blocks_in_splitter_block
                    .iter()
                    .filter(|x| (***x).borrow().values != (*smaller_component).borrow().values)
                    .map(|x| *x)
                    .collect::<Vec<Rc<RefCell<FineBlock>>>>();
                RefCell::new(t)
            };

            let simple_splitter_block_values: Block = splitter_block
                .values
                .clone()
                .iter()
                .filter(|x| !(*smaller_component).borrow().values.contains(x))
                .map(|x| *x)
                .collect();

            let simple_splitter_block = CoarseBlock {
                values: simple_splitter_block_values,
                fine_blocks_that_are_subsets_of_self: simple_splitter_block_subsets,
            };

            (smaller_component, simple_splitter_block)
        };

        let mut counterimage = build_counterimage(graph, smaller_component);

        let changed_blocks = {
            let counterimage_group = group_by_counterimage(counterimage, node_to_block_vec);
            split_blocks_with_grouped_counterimage(counterimage_group, &mut node_to_block_vec)
        };

        repartition(
            changed_blocks,
            &mut partition,
            &mut fine_partition_to_coarse,
            &mut coarse_block_makeup,
            &mut queue,
        );

        {
            let counterimage_splitter_complement = build_counterimage(graph, simple_splitter_block);

            counterimage_splitter_complement
                .keys()
                .into_iter()
                .for_each(|node| {
                    counterimage.remove(node);
                });
        }

        let changed_blocks = {
            let counterimage_group = group_by_counterimage(counterimage, node_to_block_vec);
            split_blocks_with_grouped_counterimage(counterimage_group, &mut node_to_block_vec)
        };

        repartition(
            changed_blocks,
            &mut partition,
            &mut fine_partition_to_coarse,
            &mut coarse_block_makeup,
            &mut queue,
        );
    }

    return partition;
}

#[pyfunction()]
#[pyo3(text_signature = "(graph)")]
pub fn digraph_maximum_bisimulation(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> RelationalCoarsestPartition {
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
