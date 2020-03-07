if [ ! -d rust-installer ]; then
    mkdir rust-installer
    uname -m
    if [[ `uname -m` == "s390x" ]] ; then
        export PATH=$PATH:/usr/local/lib
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    fi
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh --default-toolchain nightly -y
fi
