import qiskit
import logging

qc = qiskit.circuit.library.QuantumVolume(100, 20)
qc.measure_all()

cmap = qiskit.transpiler.CouplingMap.from_grid(10, 10)
qiskit.transpile(qc, basis_gates=['cx', 'p', 'sx', 'u', 'id'],
                 coupling_map=cmap)
