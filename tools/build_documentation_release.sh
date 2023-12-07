#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Script for building docs
# It builds the latest docs and pulls the historical stable docs
# from the git repository
set -e

# Build the documentation.
tox -edocs

# Extract the release version from Cargo.toml
VERSION=$(grep -Po -m1 'version = "\K[0-9]+\.[0-9]+' Cargo.toml)

TMP_DIR=$(mktemp -d)
git clone --depth 1 --branch gh-pages https://github.com/Qiskit/rustworkx.git $TMP_DIR
rm -rf $TMP_DIR/stable/$VERSION # Remove the old version in case of a revision release
mkdir -p $TMP_DIR/stable/$VERSION
cp -r docs/build/html/* $TMP_DIR/stable/$VERSION/
mkdir -p release_docs
cp -r $TMP_DIR/* release_docs/
touch release_docs/.nojekyll # Prevent GitHub from ignoring the _static directory

# Delete the temporary directory
rm -rf $TMP_DIR