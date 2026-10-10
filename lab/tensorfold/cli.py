# SPDX-License-Identifier: Apache-2.0
"""Inspect TensorFold sources or probe its OpenAI-compatible endpoint."""
import argparse
import os
from pathlib import Path

from common.config import dump
from lab.tensorfold.probe import probe
from lab.tensorfold.source import source_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source-manifest", help="hash and classify runtime/kernel sources")
    source.add_argument("source", type=Path)
    source.add_argument("--require-exl3", action="store_true")
    source.add_argument("--out", type=Path)
    api = sub.add_parser("probe", help="validate API and optionally compare serial/drafted output")
    api.add_argument("--endpoint", required=True)
    api.add_argument("--model")
    api.add_argument("--token-env", default="OPENAI_API_KEY")
    api.add_argument("--timeout", type=float, default=90)
    api.add_argument("--compare-draft", action="store_true")
    api.add_argument("--prompt", default="Reply with exactly SPARK_TOOLS_OK and nothing else.")
    api.add_argument("--expect-text")
    api.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "source-manifest":
            report = source_manifest(args.source, args.require_exl3)
        else:
            report = probe(args.endpoint, args.model, os.environ.get(args.token_env), args.timeout,
                           args.compare_draft, args.prompt, args.expect_text)
    except (OSError, ValueError) as exc:
        report = {"schema": 1, "status": "FAIL", "runtime": "TensorFold", "errors": [str(exc)]}
    dump(report, args.out)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
