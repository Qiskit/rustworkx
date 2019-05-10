if [ ! -d rust-installer ]; then
    mkdir rust-installer
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust-installer/rustup.sh
    sh rust-installer/rustup.sh --prefix=~/rust --spec=nightly --disable-sudo -y
fi
