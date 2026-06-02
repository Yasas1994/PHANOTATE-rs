#!/bin/bash
set -ex

# Based on bioconda recipes for Rust packages:
# https://github.com/bioconda/bioconda-recipes/tree/master/recipes/nanoq

RUST_BACKTRACE=full

if [ "$(uname)" == "Darwin" ]; then
  # Ensure HOME is set for cargo on macOS
  export HOME=$(pwd)
fi

cargo-bundle-licenses --format yaml --output THIRDPARTY.yml

cargo install -v --locked --root "$PREFIX" --path .
