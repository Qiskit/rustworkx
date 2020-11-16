import numpy as np
import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb

cmap = qiskit.transpiler.CouplingMap.from_ring(15)
length_vector = np.arange(1, 200, 4)
rb_pattern = [[0, 2], [1]]
length_multiplier = 1
seed_offset=0
align_cliffs = False

rb_circs, _ = rb.randomized_benchmarking_seq(
    nseeds=1, length_vector=length_vector, rb_pattern=rb_pattern,
    length_multiplier=length_multiplier, seed_offset=seed_offset,
    align_cliffs=align_cliffs)

all_circuits = []
for seq in rb_circs:
    all_circuits += seq

qiskit.transpile(all_circuits, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                 coupling_map=cmap)
