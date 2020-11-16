import os

from qiskit.aqua.operators import PauliOp, ListOp
from qiskit.aqua.operators.converters import AbelianGrouper
from qiskit.quantum_info import Pauli


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'abelian_data')
files = os.listdir(data_dir)

for input_file in files:
    with open(os.path.join(data_dir, input_file)) as fd:
        lines = fd.readlines()
        lst = []
        for i in range(0, len(lines), 2):
            lst.append(PauliOp(Pauli.from_label(
                lines[i].strip()), float(lines[i + 1])))
        op = ListOp(lst)
    AbelianGrouper.group_subops(op)
