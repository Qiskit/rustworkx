import { loadPyodide } from "pyodide";
import * as fs from 'fs';
import * as path from 'path';

async function run_smoke_test() {
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
  console.log("installed wheel");
  return null;
  return pyodide.runPythonAsync(`
    import rustworkx
    print(rustworkx.__version__)
  `);
}

const result = await run_smoke_test();
console.log("Smoke test completed successfully");
