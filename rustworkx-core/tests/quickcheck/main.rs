macro_rules! common_test {
    ($func:ident, $graph_ty:ty) => {
        #[test]
        fn $func() {
            rustworkx_tests::lollipop::$func::<$graph_ty>();
        }
    };
}

mod special_graph;
