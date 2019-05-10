if [ ! -d rust-installer ]; then
    mkdir rust-installer
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust-installer/rustup.sh
    sh rust-installer/rustup.sh --default-toolchain nightly -y
fi
