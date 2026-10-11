# SPDX-License-Identifier: Apache-2.0
"""Check and report explicit local experiment evidence; never execute workloads."""
import argparse
import json
from pathlib import Path

from common.atomic_publish import publish_bytes, validate_bytes
from .manifest import InvalidExperiment
from .report import build_report, markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("manifest", type=Path)
    report = sub.add_parser("report")
    report.add_argument("manifest", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_report(args.manifest)
        encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.command == "check":
            print(json.dumps({"schema": "spark-experiment-check", "version": 1,
                              "identity": result["identity"], "integrity": result["integrity"]}, sort_keys=True))
        else:
            if args.markdown:
                try:
                    same_destination = (args.out.resolve(strict=False)
                                        == args.markdown.resolve(strict=False))
                except (OSError, RuntimeError):
                    raise InvalidExperiment("report destination is unavailable or unsafe") from None
                if same_destination:
                    raise InvalidExperiment("JSON and Markdown report destinations must differ")
            primary = encoded.encode("utf-8")
            secondary = markdown(result).encode("utf-8") if args.markdown else None
            try:
                records = [(args.out, primary)]
                if secondary is not None:
                    records.append((args.markdown, secondary))
                for path, expected in records:
                    try:
                        validate_bytes(path, expected)
                    except FileNotFoundError:
                        pass

                # The optional human-readable report is secondary. Publish it
                # first and the JSON report last as the pair's commit point.
                if secondary is not None:
                    publish_bytes(args.markdown, secondary, mode=0o666)
                publish_bytes(args.out, primary, mode=0o666)
            except OSError:
                raise InvalidExperiment("report destination unavailable or already exists") from None
    except InvalidExperiment as exc:
        parser.error(str(exc))
    return 0
