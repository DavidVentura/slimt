#!/bin/bash

set -eo pipefail

ARGS=(
  -DUSE_AVX2=ON

  # TODO(jerinphilip) Adjust, later.
  -DCMAKE_BUILD_TYPE=Debug
  -DWITH_ASAN=ON

  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

set -x

# Configure
cmake -B build -S $PWD "${ARGS[@]}"

# Build
cmake --build build --target all
