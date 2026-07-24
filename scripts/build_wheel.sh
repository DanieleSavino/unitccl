#!/usr/bin/env bash
SCRIPT_DIR=$(dirname $(realpath $0))
ROOT=$(realpath "$SCRIPT_DIR/..")

FASTEST_HOME="$(realpath "$ROOT/vendor/fastest")"
export FASTEST_EXTRA_LIBS="$(realpath "$ROOT/nccl/build/lib/libnccl_static.a") -latomic -lcudart"
export FASTEST_EXTRA_COMPILE_ARGS="-std=c++17"

name=$(basename "$ROOT")

pushd "$FASTEST_HOME/bindings" || exit 1
pip install -r requirements.txt

export FASTEST_HOME
export FASTEST_USER_LIB="$ROOT/build/lib$name.a"
export FASTEST_MODULE_NAME="$name"

pip install -e .
popd
