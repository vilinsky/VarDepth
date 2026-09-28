#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
tmp="build/panel_io.$$.tmp"
trap 'rm -f "$tmp"' EXIT
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra core/panel_io.cpp $(pkg-config --cflags --libs htslib) -o "$tmp"
mv "$tmp" build/panel_io
