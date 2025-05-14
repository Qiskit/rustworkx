import { loadPyodide } from "pyodide";
import * as fs from 'fs';
import * as path from 'path';

async function rustworkx_python() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");

  // Load rustworkx wheel into memory
  const filename = 'rustworkx-0.17.0-cp39-abi3-pyodide_2024_0_wasm32.whl';

  const wheelPath = path.resolve(filename);
  const wheelData = fs.readFileSync(wheelPath);
  // console.log("Read data");

  pyodide.FS.mkdir('/tmp', 0o777);
  pyodide.FS.writeFile(`/tmp/${filename}`, new Uint8Array(wheelData));

  // console.log(`Successfully loaded ${filename} into Pyodide's virtual file system`);

  await micropip.install(`emfs://tmp/${filename}`);
  return pyodide.runPythonAsync(`
    import rustworkx
    print(rustworkx.__version__)
  `);
}

const result = await rustworkx_python();
console.log("Smoke test completed successfully");
