use petgraph::visit::Reversed;
use petgraph::Graph;
use retworkx_core::centrality::closeness_centrality;
use retworkx_core::petgraph::graph::{DiGraph, UnGraph};

#[test]
fn test_simple() {
    let g = UnGraph::<i32, ()>::from_edges(&[(1, 2), (2, 3), (3, 4), (1, 4)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![
            Some(0.0),
            Some(0.5625),
            Some(0.5625),
            Some(0.5625),
            Some(0.5625)
        ],
        c
    );
}

#[test]
fn test_wf_improved() {
    let g = UnGraph::<i32, ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![
            Some(1. / 4.),
            Some(3. / 8.),
            Some(3. / 8.),
            Some(1. / 4.),
            Some(2. / 9.),
            Some(1. / 3.),
            Some(2. / 9.)
        ],
        c
    );
    let cwf = closeness_centrality(&g, false);
    assert_eq!(
        vec![
            Some(1. / 2.),
            Some(3. / 4.),
            Some(3. / 4.),
            Some(1. / 2.),
            Some(2. / 3.),
            Some(1.),
            Some(2. / 3.)
        ],
        cwf
    );
}

#[test]
fn test_digraph() {
    let g = DiGraph::<i32, ()>::from_edges(&[(0, 1), (1, 2)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(vec![Some(0.), Some(1. / 2.), Some(2. / 3.)], c);

    let cr = closeness_centrality(Reversed(&g), true);
    assert_eq!(vec![Some(2. / 3.), Some(1. / 2.), Some(0.)], cr);
}

#[test]
fn test_k5() {
    let g = UnGraph::<i32, ()>::from_edges(&[
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]);
    let c = closeness_centrality(&g, true);
    assert_eq!(
        vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
        c
    );
}

#[test]
fn test_path() {
    let g = UnGraph::<i32, ()>::from_edges(&[(0, 1), (1, 2)]);
    let c = closeness_centrality(&g, true);
    assert_eq!(vec![Some(2. / 3.), Some(1.), Some(2. / 3.)], c);
}

#[test]
fn test_weighted_closeness() {
    let mut g = Graph::new();
    let s = g.add_node(0);
    let u = g.add_node(0);
    let x = g.add_node(0);
    let v = g.add_node(0);
    let y = g.add_node(0);
    g.add_edge(s, u, 10.);
    g.add_edge(s, x, 5.);
    g.add_edge(u, v, 1.);
    g.add_edge(u, x, 2.);
    g.add_edge(v, y, 1.);
    g.add_edge(x, u, 3.);
    g.add_edge(x, v, 5.);
    g.add_edge(x, y, 2.);
    g.add_edge(y, s, 7.);
    g.add_edge(y, v, 6.);
    let c = closeness_centrality(&g, true);
    println!("{:?}", c);
    assert_eq!(0, 0)
}
