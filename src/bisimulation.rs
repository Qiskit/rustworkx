use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashMap;

use crate::iterators::{NodeIndices, RelationalCoarsestPartition};
use crate::{digraph, Directed, StablePyGraph};
use petgraph::graph;
use petgraph::Direction::{Incoming, Outgoing};

type Block = Vec<graph::NodeIndex>;
type Partition = HashSet<Block>;

type CoarseBlock = Vec<graph::NodeIndex>;
type CoarsePartition = Vec<Rc<CoarseBlock>>;

type SubBlock = Block;

type Image = HashMap<graph::NodeIndex, Vec<graph::NodeIndex>>;
type Counterimage = Image;
type CounterimageGrouped = HashMap<Rc<Block>, Rc<SubBlock>>;

type MapFineBlockCoarseBlock = HashMap<Rc<Block>, Rc<Block>>;
type CoarseBlockMakeup = HashMap<Rc<CoarseBlock>, Vec<Rc<Block>>>;
type NodeToBlockVec = Vec<Rc<Block>>;

fn initialization(
    graph: &StablePyGraph<Directed>,
) -> (
    Partition,
    CoarsePartition,
    NodeToBlockVec,
    CoarseBlockMakeup,
    MapFineBlockCoarseBlock,
) {
    let mut graph_node_indices: Block = graph.node_indices().clone().into_iter().collect();
    graph_node_indices.sort();

    let mut leaf_node_indices_pointer: RefCell<Block> = RefCell::new(Vec::new());
    let mut non_leaf_node_indices_pointer: RefCell<Block> = RefCell::new(Vec::new());

    let mut node_to_block_vec = vec![
        Rc::clone(&non_leaf_node_indices_pointer);
        graph_node_indices.last().unwrap().clone().index() as usize
    ];

    let (coarse_block_pointer, coarse_partition_pointer): (Rc<CoarseBlock>, Rc<CoarsePartition>) = {
        let coarse_block = Rc::new(graph_node_indices);

        (
            Rc::clone(&coarse_block),
            Rc::new(vec![Rc::clone(&coarse_block)]),
        )
    };

    let fine_block_to_coarse: MapFineBlockCoarseBlock = HashMap::from([
        (
            Rc::clone(&non_leaf_node_indices_pointer),
            Rc::clone(&coarse_block_pointer),
        ),
        (
            Rc::clone(&leaf_node_indices_pointer),
            Rc::clone(&coarse_block_pointer),
        ),
    ]);

    let coarse_block_makeup: CoarseBlockMakeup = HashMap::from([(
        Rc::clone(&coarse_block_pointer),
        vec![
            Rc::clone(&non_leaf_node_indices_pointer),
            Rc::clone(&leaf_node_indices_pointer),
        ],
    )]);

    graph.node_indices().into_iter().for_each(|x| {
        match graph.neighbors_directed(x, Outgoing).count() == 0 {
            true => {
                node_to_block_vec[x.index()] = Rc::clone(&leaf_node_indices_pointer);
                leaf_node_indices_pointer.push(x);
            }
            false => {
                non_leaf_node_indices_pointer.push(x);
            }
        }
    });

    return (
        HashSet::from([*leaf_node_indices_pointer, *non_leaf_node_indices_pointer]),
        *coarse_partition_pointer,
        node_to_block_vec,
        coarse_block_makeup,
        fine_block_to_coarse,
    );
}

fn build_counterimage(graph: &StablePyGraph<Directed>, block: Block) -> Counterimage {
    let mut counterimage = HashMap::new();
    block.iter().for_each(|node_index_pointer| {
        counterimage.insert(
            (*node_index_pointer),
            graph
                .neighbors_directed(*node_index_pointer, Incoming)
                .into_iter()
                .collect::<Vec<graph::NodeIndex>>(),
        );
    });

    return counterimage;
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
                .entry(block)
                .or_insert(Rc::new(vec![*node]))
                .push(*node);
        }
    }

    return counterimage_grouped;
}

fn split_blocks_with_grouped_counterimage(
    mut counterimage_grouped: CounterimageGrouped,
    node_to_block_vec: &mut NodeToBlockVec,
) -> Vec<(Rc<Block>, Rc<SubBlock>)> {
    let mut changed_blocks = Vec::new();
    for (block, subblock) in counterimage_grouped.iter_mut() {
        for node_index in subblock.into_iter() {
            node_to_block_vec[node_index.index()] = Rc::clone(&subblock);
        }
        // subtract subblock from block
        (**block) = block
            .iter()
            .filter(|x| !subblock.contains(x))
            .map(|x| *x)
            .collect();

        changed_blocks.push((Rc::clone(block), Rc::clone(subblock)));
    }
    return changed_blocks;
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

fn find_smallest_component_block_and_copy(
    coarse_block: CoarseBlock,
    coarse_block_makeup: Vec<Rc<Block>>,
) -> Block {
    let smallest_component_block_pointer = coarse_block_makeup
        .iter()
        .min_by(|x, y| x.len().cmp(&y.len()))
        .unwrap();
    return (**smallest_component_block_pointer).clone();
}

fn maximum_bisimulation(graph: &StablePyGraph<Directed>) -> Partition {
    let (
        mut partition,
        initial_coarse_partition,
        mut node_to_block_vec,
        mut coarse_block_makeup,
        mut fine_partition_to_coarse,
    ) = initialization(graph);

    let mut queue: CoarsePartition = initial_coarse_partition;

    while !queue.is_empty() {
        let (smaller_component, simple_splitter_block) = {
            let splitter_block = queue.pop().unwrap();
            let coarse_blocks_in_splitter_block =
                *coarse_block_makeup.get(&splitter_block).unwrap();
            if !(coarse_blocks_in_splitter_block.len() == 2) {
                queue.push(splitter_block);
            }

            let smaller_component = find_smallest_component_block_and_copy(
                *splitter_block,
                coarse_blocks_in_splitter_block,
            );

            let simple_splitter_block: CoarseBlock = splitter_block
                .into_iter()
                .filter(|x| !smaller_component.contains(x))
                .collect();

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
