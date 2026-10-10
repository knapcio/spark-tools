#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python3}" tests/smoke_examples.py
