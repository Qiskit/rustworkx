#!/bin/bash
wget https://static.rust-lang.org/dist/rust-nightly-s390x-unknown-linux-gnu.tar.gz
yum clean packages
yum clean headers
yum clean metadata
yum clean all
rm -rf /var/cache/yum
rm -rf /var/tmp/yum-*
package-cleanup --oldkernels --count=1
tar xzvf rust-nightly-s390x-unknown-linux-gnu.tar.gz
rm -rf rust-nightly-s390x-unknown-linux-gnu.tar.gz
pushd rust-nightly-s390x-unknown-linux-gnu
./install.sh
popd
