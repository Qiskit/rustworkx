if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh --default-toolchain nightly -y
fi
