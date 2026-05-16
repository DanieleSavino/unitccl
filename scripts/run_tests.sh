#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $(realpath $0))
export FASTEST_PROJECT=$(basename $(dirname $SCRIPT_DIR))
exec "$SCRIPT_DIR/../tests.py" "$@"
