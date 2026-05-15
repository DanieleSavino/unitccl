#!/usr/bin/env bash

FASTEST_HOME="$(realpath vendor/fastest)"

export FASTEST_EXTRA_LIBS="$(realpath nccl/build/lib/libnccl_static.a) -latomic -lcudart"
export FASTEST_EXTRA_COMPILE_ARGS="-std=c++17"

name=$(basename "$PWD")

pushd "$FASTEST_HOME/bindings" || exit 1

pip install -r requirements.txt

export FASTEST_HOME
export FASTEST_USER_LIB="../../../build/lib$name.a"
export FASTEST_MODULE_NAME="$name"

pip install -e .

popd
