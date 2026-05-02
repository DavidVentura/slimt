#!/bin/bash

set -eo pipefail

# Configure
ARGS=(
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DUSE_AVX2=ON
  -DUSE_BUILTIN_SENTENCEPIECE=ON

  -DCMAKE_BUILD_TYPE=Debug
  -DWITH_ASAN=ON
)

cmake -B build -S $PWD "${ARGS[@]}"

# Build
cmake --build build --target all
