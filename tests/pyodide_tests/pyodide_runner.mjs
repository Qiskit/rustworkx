/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * Running this file is the equivalent of running:
 * python -m unittest discover .
 * at the root of the tests folder. It works both with Pyodide and
 * in a normal Python environment.
 */

/*
  Usage: node pyodide_runner.mjs <path_to_wheel>
  This loads Pyodide, rustworkx, networkx, numpy, and the test files
  and runs the tests in memory with unittest.
*/

import { loadPyodide } from "pyodide";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

async function getPyodideWithDeps() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");

  // Load rustworkx from local file system to Pyodide's
  const filePath = process.argv[2];
  if (process.argv[2] === "") {
    throw new Error("Wheel path is empty, check the logs to see if there are multiple wheels in dist/.");
  }
  const filename = path.basename(filePath);

  const wheelPath = path.resolve(filePath);
  const wheelData = fs.readFileSync(wheelPath);

  pyodide.FS.writeFile(`/tmp/${filename}`, new Uint8Array(wheelData));

  // Install rustworkx + networkx for testing. We ignore the optional dependencies.
  await micropip.install.callKwargs({
    requirements: [`emfs:/tmp/${filename}`, "numpy", "networkx"],
    deps: false,
  })

  return pyodide;
}

async function getUnitTests(pyodide) {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  let unitTestFile = `${__dirname}/runner.py`;
  let unitTestDir = path.dirname(__dirname);
  // Mount test directory to Pyodide's file system
  let mountDir = "/tmp/tests";
  pyodide.FS.mkdirTree(mountDir);
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: unitTestDir }, mountDir);
  // Read the test runner file
  let unitTestRunner = fs.readFileSync(unitTestFile);
  return unitTestRunner.toString();
}

async function main() {
  let pyodide = await getPyodideWithDeps();
  let unitTests = await getUnitTests(pyodide);
  try {
    await pyodide.runPythonAsync(unitTests);
    console.log("Smoke test completed successfully");
  } catch (error) {
    console.error("Error during smoke test:", error);
    process.exit(1);
  }
}

if (process.argv[1] === import.meta.filename){
  await main();
}

