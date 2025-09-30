use petgraph::graph::{DiGraph, UnGraph};
use petgraph::visit::{EdgeRef, IntoNodeReferences, NodeRef};
use quickcheck::{TestResult, quickcheck};
use rustworkx_core::generators::{
    InvalidInputError, hexagonal_lattice_graph, hexagonal_lattice_graph_weighted,
};
use std::collections::HashSet;

#[test]
fn prop_hexagonal_lattice_node_and_edge_count() {
    fn prop(rows: u8, cols: u8, bidirectional: bool) -> TestResult {
        let (rows, cols) = (rows as usize % 5, cols as usize % 5);
        if rows == 0 || cols == 0 {
            return TestResult::discard();
        }

        let expected = {
            let edge_factor = if bidirectional { 2 } else { 1 };
            let rowlen = 2 * rows + 2;
            let collen = cols + 1;
            let num_nodes = rowlen * collen - 2;
            let num_edges = edge_factor * (3 * rows * cols + 2 * (rows + cols) - 1);
            (num_nodes, num_edges)
        };

        let g: DiGraph<(), ()> =
            match hexagonal_lattice_graph(rows, cols, || (), || (), bidirectional, false) {
                Ok(g) => g,
                Err(_) => return TestResult::failed(),
            };

        if g.node_count() != expected.0 || g.edge_count() != expected.1 {
            return TestResult::failed();
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8, u8, bool) -> TestResult);
}

#[test]
fn prop_hexagonal_lattice_bidirectionality() {
    fn prop(rows: u8, cols: u8) -> TestResult {
        let (rows, cols) = (rows as usize % 4 + 1, cols as usize % 4 + 1);
        let g: DiGraph<(), ()> =
            match hexagonal_lattice_graph(rows, cols, || (), || (), true, false) {
                Ok(g) => g,
                Err(_) => return TestResult::failed(),
            };

        let mut seen = HashSet::new();
        for edge in g.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            seen.insert((u, v));
        }

        for &(u, v) in &seen {
            if !seen.contains(&(v, u)) {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8, u8) -> TestResult);
}

#[test]
fn prop_hexagonal_lattice_position_weights() {
    fn prop(rows: u8, cols: u8) -> TestResult {
        let (rows, cols) = (rows as usize % 4 + 1, cols as usize % 4 + 1);
        let g: UnGraph<(usize, usize), ()> = match hexagonal_lattice_graph_weighted(
            rows,
            cols,
            |u, v| (u, v),
            || (),
            false,
            false,
        ) {
            Ok(g) => g,
            Err(_) => return TestResult::failed(),
        };

        for node in g.node_references() {
            let (u, v) = *node.weight();
            if !g.node_weights().any(|&(x, y)| x == u && y == v) {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    quickcheck(prop as fn(u8, u8) -> TestResult);
}

#[test]
fn prop_hexagonal_lattice_invalid_periodic_rejected() {
    fn prop(rows: u8, cols: u8) -> TestResult {
        let rows = rows as usize % 4 + 1;
        let cols = (2 * (cols as usize % 2)) + 1; // make sure cols is odd

        let g = hexagonal_lattice_graph::<UnGraph<(), ()>, _, _, _, ()>(
            rows,
            cols,
            || (),
            || (),
            false,
            true,
        );

        match g {
            Err(InvalidInputError) => TestResult::passed(),
            _ => TestResult::failed(),
        }
    }

    quickcheck(prop as fn(u8, u8) -> TestResult);
}
