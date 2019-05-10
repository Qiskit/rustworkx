if [ ! -d ~/rust-installer ]; then
    mkdir ~/rust-installer
    curl -sL https://static.rust-lang.org/rustup.sh -o ~/rust-installer/rustup.sh
    sh ~/rust-installer/rustup.sh --prefix=~/rust --spec=nightly --disable-sudo -y
fi
