#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
if [[ $(uname) == Darwin ]]; then
  c++ -O3 -std=c++20 -dynamiclib -o build/libcaller.dylib core/core.cpp
else
  c++ -O3 -std=c++20 -fPIC -shared -o build/libcaller.so core/core.cpp
fi
