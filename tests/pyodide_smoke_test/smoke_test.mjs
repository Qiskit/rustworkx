import { loadPyodide } from "pyodide";

async function rustworkx_python() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(`http://localhost:8000/rustworkx-0.17.0-cp39-abi3-pyodide_2024_0_wasm32.whl`);
  return pyodide.runPythonAsync(`
	  import rustworkx
	  print(rustworkx.__version__)
  `);
}

const result = await rustworkx_python();
console.log("Smoke test completed successfully");
