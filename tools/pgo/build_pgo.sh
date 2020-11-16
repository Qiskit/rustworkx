#!/bin/bash

rm -rf /tmp/pgo-data
python -m venv build_pgo
source build_pgo/bin/activate
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" build_pgo/bin/pip install .
pushd tools/pgo
git clone --depth 1 https://github.com/Qiskit/qiskit-terra
pushd qiskit-terra
pip install .
python setup.py build_ext --inplace
pip install -r requirements-dev.txt
stestr run
popd
pip install git+https://github.com/Qiskit/qiskit-ignis
pip install git+https://github.com/Qiskit/qiskit-aqua
python qv.py
python rb.py
python abelian.py
popd
deactivate
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
