# SPDX-License-Identifier: Apache-2.0
"""Check and report explicit local experiment evidence; never execute workloads."""
import argparse
import json
import os
from pathlib import Path

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
            # Create-only prevents clobbering any evidence/input, including aliases.
            # Open both destinations before writing so conflicts produce no report.
            streams = []
            try:
                streams.append((args.out, args.out.open("x", encoding="utf-8")))
                if args.markdown:
                    streams.append((args.markdown, args.markdown.open("x", encoding="utf-8")))
                streams[0][1].write(encoded)
                if len(streams) == 2:
                    streams[1][1].write(markdown(result))
            except OSError:
                for path, stream in streams:
                    stream.close()
                    os.unlink(path)
                raise InvalidExperiment("report destination unavailable or already exists") from None
            finally:
                for _, stream in streams:
                    stream.close()
    except InvalidExperiment as exc:
        parser.error(str(exc))
    return 0
