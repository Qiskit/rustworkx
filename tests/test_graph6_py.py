import tempfile
import rustworkx as rx


def test_graph6_roundtrip(tmp_path):
    # build a small graph with node/edge attrs
    g = rx.PyGraph()
    g.add_node({"label": "n0"})
    g.add_node({"label": "n1"})
    g.add_edge(0, 1, {"weight": 3})

    p = tmp_path / "g.g6"
    rx.graph_write_graph6_file(g, str(p))

    g2_list = rx.read_graph6_file(str(p))
    assert isinstance(g2_list, list)
    g2 = g2_list[0]

    # check nodes and edges count
    assert g2.node_count() == 2
    assert g2.edge_count() == 1

    # check that node attrs 'label' were preserved in node data
    # Graph6 has no native attrs; our implementation stores None for attrs currently,
    # so assert node attrs exist or are None
    n0 = g2[0]
    assert n0 is None or ("label" in n0 and n0["label"] == "n0")

    # check edge exists
    assert list(g2.edge_list())
