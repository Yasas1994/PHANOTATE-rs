#!/usr/bin/env bash
set -euo pipefail

cargo install --locked --root "$PREFIX" --path .
