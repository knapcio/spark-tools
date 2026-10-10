# SPDX-License-Identifier: Apache-2.0
"""Inspect EXL3 checkpoint metadata without loading weights or a GPU runtime."""
import argparse
from pathlib import Path

from common.config import dump
from lab.exl3.checkpoint import inspect_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="validate config and safetensors headers")
    inspect.add_argument("model_dir", type=Path)
    inspect.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_checkpoint(args.model_dir)
    except (OSError, ValueError) as exc:
        error = str(exc) if isinstance(exc, ValueError) else "checkpoint could not be read"
        report = {"schema": 1, "status": "FAIL", "format": "EXL3", "errors": [error]}
    dump(report, args.out)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
