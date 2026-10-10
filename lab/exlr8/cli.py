# SPDX-License-Identifier: Apache-2.0
"""Inspect a remote EXLR8 release without downloading its weight payloads."""
import argparse
import os
from pathlib import Path

from common.config import dump
from lab.exlr8.remote import inspect_hf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-hf", help="inspect Hugging Face metadata, manifests and bounded headers")
    inspect.add_argument("repo", help="Hugging Face owner/name")
    inspect.add_argument("--revision", default="main")
    inspect.add_argument("--token-env", default="HF_" + "TOKEN")
    inspect.add_argument("--timeout", type=float, default=30)
    inspect.add_argument("--header-samples", type=int, default=10)
    inspect.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_hf(args.repo, args.revision, os.environ.get(args.token_env),
                            args.timeout, args.header_samples)
    except (OSError, ValueError) as exc:
        report = {"schema": 1, "status": "FAIL", "format": "EXLR8", "errors": [str(exc)]}
    dump(report, args.out)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
