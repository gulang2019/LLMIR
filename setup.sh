#!/bin/bash
ROOT_DIR="$(pwd)/compiler"
BIN_DIR="$ROOT_DIR/bin"
mkdir -p "$BIN_DIR"

export PATH="$BIN_DIR:$PATH"

# Helper: Link binaries from subtool dir to compiler/bin/
link_binaries() {
  local tool_bin="$1"
  for f in "$tool_bin"/*; do
    [ -f "$f" ] && ln -sf "$f" "$BIN_DIR/$(basename "$f")"
  done
}

# Helper: Install only if binary not found
check_installed() {
  local binary="$1"
  if command -v "$binary" &>/dev/null && [[ "$(command -v "$binary")" == "$BIN_DIR/"* ]]; then
    echo "✅ $binary already installed."
    return 1
  fi
  return 0
}

install_node_ts() {
  check_installed node || return 0
  NODE_VERSION="20.11.1"
  curl -LO "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz"
  tar -xf "node-v$NODE_VERSION-linux-x64.tar.xz" -C "$ROOT_DIR"
  link_binaries "$ROOT_DIR/node-v$NODE_VERSION-linux-x64/bin"
  npm install -g typescript
}

install_julia() {
  check_installed julia || return 0
  JULIA_VERSION="1.8.2"
  curl -LO "https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
  tar -xzf julia-$JULIA_VERSION-linux-x86_64.tar.gz -C "$ROOT_DIR"
  link_binaries "$ROOT_DIR/julia-$JULIA_VERSION/bin"
}

install_swift() {
  check_installed swiftc || return 0
  SWIFT_VERSION="5.7"
  curl -LO "https://download.swift.org/swift-$SWIFT_VERSION-release/ubuntu2204/swift-$SWIFT_VERSION-RELEASE/swift-$SWIFT_VERSION-RELEASE-ubuntu22.04.tar.gz"
  tar -xzf swift-$SWIFT_VERSION-RELEASE-ubuntu22.04.tar.gz -C "$ROOT_DIR"
  link_binaries "$ROOT_DIR/swift-$SWIFT_VERSION-RELEASE-ubuntu22.04/usr/bin"
}

install_elixir_erlang() {
  check_installed elixir || return 0
  mkdir -p "$ROOT_DIR/erlang" && cd "$ROOT_DIR/erlang"
  curl -LO https://builds.hex.pm/builds/otp/ubuntu-22.04/OTP-27.0.tar.gz
  tar xzf OTP-27.0.tar.gz --strip-components=1
  ./Install -minimal "$ROOT_DIR/erlang"
  link_binaries "$ROOT_DIR/erlang/bin"

  mkdir -p "$ROOT_DIR/elixir" && cd "$ROOT_DIR/elixir"
  curl -LO https://builds.hex.pm/builds/elixir/v1.17.2-otp-27.zip
  unzip -q v1.17.2-otp-27.zip
  link_binaries "$ROOT_DIR/elixir/bin"
}

install_clojure() {
  check_installed clojure || return 0
  cd "$ROOT_DIR"
  curl -LO https://github.com/clojure/brew-install/releases/latest/download/linux-install.sh
  chmod +x linux-install.sh
  ./linux-install.sh --prefix "$ROOT_DIR/clojure"
  link_binaries "$ROOT_DIR/clojure/bin"
}

install_luau() {
  check_installed luau || return 0
  curl -LO https://github.com/luau-lang/luau/releases/download/0.667/luau-ubuntu.zip
  unzip -q luau-ubuntu.zip -d "$ROOT_DIR/luau"
  ln -sf "$ROOT_DIR/luau/luau" "$BIN_DIR/luau"
}

install_dart() {
  check_installed dart || return 0
  DART_SDK_VERSION="3.2.3"
  curl -LO "https://storage.googleapis.com/dart-archive/channels/stable/release/$DART_SDK_VERSION/sdk/dartsdk-linux-x64-release.zip"
  unzip -q dartsdk-linux-x64-release.zip -d "$ROOT_DIR/dart"
  link_binaries "$ROOT_DIR/dart/dart-sdk/bin"
}

install_racket() {
  check_installed racket || return 0
  curl -LO https://download.racket-lang.org/installers/8.12/racket-8.12-x86_64-linux.sh
  bash racket-8.12-x86_64-linux.sh --in-place --dest "$ROOT_DIR/racket"
  link_binaries "$ROOT_DIR/racket/bin"
}

install_go() {
  check_installed go || return 0
  GO_VERSION="1.22.0"
  curl -LO "https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz"
  tar -C "$ROOT_DIR" -xzf "go$GO_VERSION.linux-amd64.tar.gz"
  link_binaries "$ROOT_DIR/go/bin"
}

install_php() {
  check_installed php || return 0
  PHP_VERSION="8.2.17"
  curl -LO "https://www.php.net/distributions/php-$PHP_VERSION.tar.gz"
  tar -xzf "php-$PHP_VERSION.tar.gz"
  cd "php-$PHP_VERSION"
  ./configure --prefix="$ROOT_DIR/php" --disable-all
  make -j$(nproc) && make install
  link_binaries "$ROOT_DIR/php/bin"
  cd ..
}

install_python_packages() {
  python3 -m pip show numpy tqdm &>/dev/null && return 0
  python3 -m pip install --user numpy tqdm
}

install_scala() {
  check_installed scala || return 0
  curl -LO https://downloads.lightbend.com/scala/2.13.13/scala-2.13.13.tgz
  tar -xzf scala-2.13.13.tgz -C "$ROOT_DIR"
  link_binaries "$ROOT_DIR/scala-2.13.13/bin"
}

install_rust() {
  check_installed rustc || return 0
  curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path
  export PATH="$HOME/.cargo/bin:$PATH"
  ln -sf "$(which rustc)" "$BIN_DIR/rustc"
}

install_ocaml() {
  check_installed ocaml || return 0
  # Download and patch OCaml install script
  curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh -o install_ocaml.sh

  # Patch DEFAULT_BINDIR to compiler/bin
  sed -i "s|^DEFAULT_BINDIR=.*|DEFAULT_BINDIR=\"$BIN_DIR\"|" install_ocaml.sh

  # Run patched installer
  bash install_ocaml.sh

  # Init opam (user local)
  export OPAMROOT="$ROOT_DIR/opam"
  "$BIN_DIR/opam" init --disable-sandboxing -y
  "$BIN_DIR/opam" switch create 4.14.0
  eval "$("$BIN_DIR/opam" env)"

  # Symlink main binary
  ln -sf "$(which ocaml)" "$BIN_DIR/ocaml"
}

install_haskell() {
  check_installed runghc || return 0
  curl -sSL https://get-ghcup.haskell.org | BOOTSTRAP_HASKELL_NONINTERACTIVE=1 sh
  export PATH="$HOME/.ghcup/bin:$PATH"
  ghcup install ghc
  ln -sf "$(which runghc)" "$BIN_DIR/runghc"
}

# === Run installs ===
# install_ocaml
install_luau
# install_node_ts
# install_julia
# install_swift
# install_elixir_erlang
# install_clojure
# install_dart
# install_racket
# install_go
# install_php
# install_python_packages
# install_scala
# install_rust
# install_haskell

echo "✅ All compilers installed in ./compiler/, binaries in ./compiler/bin/"
echo "➕ Add to shell config: export PATH=\"$(pwd)/compiler/bin:\$PATH\""
