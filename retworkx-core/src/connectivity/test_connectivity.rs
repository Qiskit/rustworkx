// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

#[cfg(test)]
mod test_conn_components {
    use std::iter::FromIterator;
    use hashbrown::HashSet;
    use petgraph::graph::Graph;
    use petgraph::graph::NodeIndex;
    use petgraph::Undirected;
    use petgraph::graph::node_index as ndx;

    use crate::connectivity::conn_components::{connected_components, number_connected_components};

    #[test]
    fn test_number_connected() {
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (3, 4)]);
        assert_eq!(number_connected_components(&graph), 2);
    }

    #[test]
    fn test_number_node_holes() {
        let mut graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2)]);
        graph.remove_node(NodeIndex::new(1));
        assert_eq!(number_connected_components(&graph), 2);
    }

    #[test]
    fn test_connected_components() {
        let graph = Graph::<(), (), Undirected>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ]);
        let components = connected_components(&graph);
        let exp1 = HashSet::from_iter([ndx(0), ndx(1), ndx(3), ndx(2)]);
        let exp2 = HashSet::from_iter([ndx(7), ndx(5), ndx(4), ndx(6)]);
        let expected = vec![exp1, exp2];
        assert_eq!(expected, components);
    }
}
