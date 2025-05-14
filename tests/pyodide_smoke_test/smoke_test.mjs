import { loadPyodide } from "pyodide";
import * as fs from 'fs';
import * as path from 'path';

async function get_pyodide_with_rustworkx() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");

  // Load rustworkx wheel into memory
  const filePath = process.argv[2];
  const filename = path.basename(filePath);

  const wheelPath = path.resolve(filePath);
  const wheelData = fs.readFileSync(wheelPath);

  pyodide.FS.writeFile(`/tmp/${filename}`, new Uint8Array(wheelData));

  await micropip.install(`emfs:/tmp/${filename}`);
  return pyodide;
}

async function run_smoke_test() {
  let pyodide = await get_pyodide_with_rustworkx();
  return pyodide.runPythonAsync(`
import rustworkx
import unittest

class TestPyodide(unittest.TestCase):
  def test_isomorphism(self):
    # Adapted from tests/graph/test_isomorphic.py
    import rustworkx

    n = 15
    upper_bound_k = (n - 1) // 2
    for k in range(1, upper_bound_k + 1):
        for t in range(k, upper_bound_k + 1):
            result = rustworkx.is_isomorphic(
                rustworkx.generators.generalized_petersen_graph(n, k),
                rustworkx.generators.generalized_petersen_graph(n, t),
            )
            expected = (k == t) or (k == n - t) or (k * t % n == 1) or (k * t % n == n - 1)
            self.assertEqual(result, expected)

  def test_rayon_works(self):
    # This essentially tests that multi-threading is set to one core and does not panic
    import rustworkx
    graph = rustworkx.generators.cycle_graph(10)
    path_lenghts_floyd = rustworkx.floyd_warshall(graph)
    path_lenghts_no_self = rustworkx.all_pairs_dijkstra_path_lengths(graph, lambda _: 1.0)
    path_lenghts_dijkstra = {k: {**v, k: 0.0} for k, v in path_lenghts_no_self.items()}
    self.assertEqual(path_lenghts_floyd, path_lenghts_dijkstra)

suite = unittest.TestLoader().loadTestsFromTestCase(TestPyodide)
runner = unittest.TextTestRunner()
runner.run(suite)
`);
}

const result = await run_smoke_test();
console.log("Smoke test completed successfully");
