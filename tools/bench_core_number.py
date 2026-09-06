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

"""Benchmark the installed Python extension, including dict conversion.

Run outside the checkout root against each built wheel:
RAYON_NUM_THREADS=1 python /path/to/tools/bench_core_number.py --nodes 10000 --samples 20
Cases match rustworkx-core/benches/core_number.rs. Graph setup and exact answer
validation are untimed. Each case has one warmup and emits individual CSV samples.
"""

import argparse
import heapq
import time

import rustworkx


# The integer generator and insertion order also appear in the Rust benchmark.
def irregular_edges(n):
    connected = n - 1
    dense = max(2, connected * 3 // 4)
    state = 0xA0761D6478BD642F
    edges, seen = [], set()

    def add(source, target):
        if (source, target) not in seen:
            seen.add((source, target))
            edges.append((source, target))

    for source in range(dense):
        attempts = 64 if source % 31 == 0 else 1 + (source * 17 + 11) % 16
        for attempt in range(min(attempts, dense - 1)):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            target = (source + 1 + state % (dense - 1)) % dense
            add(source, target)
            if attempt % 3 == 0:
                add(target, source)
    for source in range(dense, connected):
        for offset in range(1 + source % 3):
            add(source, (source * 17 + offset * 13) % dense)
    return edges


def reference_core_numbers(n, edges):
    # Independent min-degree elimination with a heap, no degree buckets or dense markers.
    adjacency = [set() for _ in range(n)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    degree = [len(neighbors) for neighbors in adjacency]
    pending = [(value, node) for node, value in enumerate(degree)]
    heapq.heapify(pending)
    removed, expected, level = [False] * n, [0] * n, 0
    while pending:
        value, node = heapq.heappop(pending)
        if removed[node] or value != degree[node]:
            continue
        level = max(level, value)
        expected[node], removed[node] = level, True
        for neighbor in adjacency[node]:
            if not removed[neighbor]:
                degree[neighbor] -= 1
                heapq.heappush(pending, (degree[neighbor], neighbor))
    return expected


def fixture(shape, n, stride):
    graph = rustworkx.PyDiGraph()
    graph.add_nodes_from(range(n * stride))
    graph.remove_nodes_from(i for i in range(n * stride) if i % stride)
    expected = dict.fromkeys(range(0, n * stride, stride), 0)
    connected = n - 1
    edges = []
    if shape == "ring":
        steps = min(4, (connected - 1) // 2)
        for a in range(connected):
            expected[a * stride] = 2 * steps
            for step in range(1, steps + 1):
                b = (a + step) % connected
                edges.extend([(a * stride, b * stride), (b * stride, a * stride)])
    elif shape == "hub":
        expected.update((a * stride, 1) for a in range(connected))
        edges = [(0, b * stride) for b in range(1, connected)]
    elif shape == "irregular":
        dense_edges = irregular_edges(n)
        expected = {
            node * stride: core for node, core in enumerate(reference_core_numbers(n, dense_edges))
        }
        edges = [(source * stride, target * stride) for source, target in dense_edges]
    else:
        width = connected if shape == "clique" else 16
        for start in range(0, connected, width):
            end = min(start + width, connected)
            expected.update((a * stride, end - start - 1) for a in range(start, end))
            edges.extend(
                (a * stride, b * stride) for a in range(start, end) for b in range(a + 1, end)
            )
    graph.add_edges_from_no_data(edges)
    return graph, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.nodes < 8 or args.samples < 1:
        parser.error("nodes must be at least 8 and samples must be positive")
    print("shape,nodes,edges,stride,sample,nanoseconds")
    for shape in ("ring", "hub", "cliques", "clique", "irregular"):
        # Bound the quadratic fixture independently of the sparse graph size.
        n = min(args.nodes, 256) if shape == "clique" else args.nodes
        for stride in (1, 2):
            graph, expected = fixture(shape, n, stride)
            for sample in range(args.samples + 1):
                start = time.perf_counter_ns()
                result = rustworkx.core_number(graph)
                elapsed = time.perf_counter_ns() - start
                assert result == expected and list(result) == list(expected)
                if sample:
                    print(
                        f"{shape},{n},{graph.num_edges()},{stride},{sample},{elapsed}",
                        flush=True,
                    )
                del result


if __name__ == "__main__":
    main()
