# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from retworkx import PyGraph

import pytest

@pytest.mark.mypy_testing
def test_pygraph_simple() -> None:
    graph: PyGraph[str, int] = PyGraph()
    node_a = graph.add_node("A")
    node_b = graph.add_node("B")
    edge_ab = graph.add_edge(node_a, node_b, 3)
    reveal_type(node_a) # note: Revealed type is "builtins.int"
    reveal_type(edge_ab) # note: Revealed type is "builtins.int"
