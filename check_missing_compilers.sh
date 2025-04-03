#!/bin/bash
set -e

BIN_DIR="$(pwd)/compiler/bin"

# List of expected binaries to check
BINARIES=(
  elixir
  clojure
  php
  swiftc
  racket
  runghc
  node
  julia
  scala
  ocaml
  rustc
  go
  dart
  luau
  typescript
)

echo "🔍 Checking for missing compilers/interpreters in $BIN_DIR"

MISSING=()
for binary in "${BINARIES[@]}"; do
  if [[ ! -x "$BIN_DIR/$binary" ]]; then
    MISSING+=("$binary")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "✅ All expected compilers/interpreters are installed in compiler/bin."
else
  echo "❌ Missing the following compilers/interpreters:"
  for bin in "${MISSING[@]}"; do
    echo " - $bin"
  done
  echo "💡 Run install_user_compilers.sh to install the missing ones."
fi
