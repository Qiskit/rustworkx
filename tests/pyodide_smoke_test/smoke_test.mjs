import { loadPyodide } from "pyodide";
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

async function getPyodideWithRustworkx() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");

  // Load rustworkx from local file system to Pyodide's
  const filePath = process.argv[2];
  const filename = path.basename(filePath);

  const wheelPath = path.resolve(filePath);
  const wheelData = fs.readFileSync(wheelPath);

  pyodide.FS.writeFile(`/tmp/${filename}`, new Uint8Array(wheelData));

  // Install with micropip
  await micropip.install(`emfs:/tmp/${filename}`);
  return pyodide;
}

async function getUnitTest() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  let unitTestFile = `${__dirname}/test_smoke.py`;
  let unitTestData = fs.readFileSync(unitTestFile);
  return unitTestData.toString();
}

async function runSmokeTest() {
  let pyodide = await getPyodideWithRustworkx();
  let unitTest = await getUnitTest();
  return pyodide.runPythonAsync(unitTest);
}

await runSmokeTest();
console.log("Smoke test completed successfully");
