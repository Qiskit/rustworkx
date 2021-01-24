#!/bin/bash

if [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    WINDOWS=1
else
    WINDOWS=0

fi

rm -rf /tmp/pgo-data
python -m venv build_pgo
if [[ $WINDOWS -eq 1 ]] ; then
    source build_pgo/Scripts/activate
else
    source build_pgo/bin/activate
fi
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" pip install .
pushd tools/pgo
git clone --depth 1 https://github.com/Qiskit/qiskit-terra
pushd qiskit-terra
pip install .
python setup.py build_ext --inplace
sed '/jax.*/d' requirements-dev.txt > requirements-dev_nojax.txt
pip install -r requirements-dev_nojax.txt
stestr run
popd
python qv.py
pip install git+https://github.com/Qiskit/qiskit-ignis
python rb.py
if [[ !($WINDOWS -eq 1 && `uname -m` != "x86_64") ]] ; then
    pip install git+https://github.com/Qiskit/qiskit-aqua
    python abelian.py
fi
popd
deactivate
if [[ "$OSTYPE" == "darwin"* ]]; then
    xcrun llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
else
    llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
fi
