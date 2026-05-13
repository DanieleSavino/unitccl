#!/usr/bin/env bash

export FASTEST_PROJECT=$(basename "$PWD")
exec ./tests.py "$@"
