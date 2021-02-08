#!/bin/bash
wget https://static.rust-lang.org/dist/rust-1.49.0-s390x-unknown-linux-gnu.tar.gz
tar xzvf rust-1.49.0-s390x-unknown-linux-gnu.tar.gz
rm -rf rust-1.49.0-s390x-unknown-linux-gnu.tar.gz
pushd rust-1.49.0-s390x-unknown-linux-gnu
./install.sh
popd
