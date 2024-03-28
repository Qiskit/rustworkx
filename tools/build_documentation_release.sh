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
nox -edocs

# Extract the release version from Cargo.toml
VERSION=$(grep -Po -m1 'version = "\K[0-9]+\.[0-9]+' Cargo.toml)

TMP_DIR=$(mktemp -d)
TMP_BAK_DIR=$(mktemp -d)
git clone --depth 1 --branch gh-pages https://github.com/Qiskit/rustworkx.git $TMP_DIR
rm -rf $TMP_DIR/stable/$VERSION # Remove the old version in case of a revision release

# Copying dev, older stable versions and aux files
mkdir $TMP_BAK_DIR/stable
cp -r $TMP_DIR/stable/* $TMP_BAK_DIR/stable
mkdir $TMP_BAK_DIR/dev
cp -r $TMP_DIR/dev/* $TMP_BAK_DIR/dev
if [ -e $TMP_DIR/CNAME ]; then
    cp $TMP_DIR/CNAME $TMP_BAK_DIR/CNAME
fi

# Copy the new version to the stable directory and to the root
mkdir -p release_docs
cp -r $TMP_BAK_DIR/* release_docs/
mkdir -p release_docs/stable/$VERSION
cp -r docs/build/html/* release_docs/stable/$VERSION/
cp -r docs/build/html/* release_docs/
touch release_docs/.nojekyll # Prevent GitHub from ignoring the _static directory

# Delete the temporary directory
rm -rf $TMP_DIR
rm -rf $TMP_BAK_DIR