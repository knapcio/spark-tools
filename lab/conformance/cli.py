# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures and pinned external plans: dense decoder conformance only."""
import argparse
import json
import subprocess

from . import SCOPE
from .manifests import RUNNERS, UPSTREAMS, check_source, runner_manifest
from .suite import cpu_report, fixtures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cpu", "fixtures"):
        command = sub.add_parser(name)
        command.add_argument("--format", choices=("all", "exl3", "exlr8"), default="all")
    manifest = sub.add_parser("manifest", help="print a pinned external runner plan; execute nothing")
    manifest.add_argument("--runner", choices=RUNNERS, required=True)
    check = sub.add_parser("check-sources", help="read-only local checkout pin validation")
    check.add_argument("--upstream", choices=tuple(UPSTREAMS), required=True)
    check.add_argument("--source", required=True)
    sub.add_parser("cuda", help="local GPU adapter unavailable; use pinned external manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "cpu":
            report = cpu_report(args.format)
        elif args.command == "fixtures":
            report = fixtures(args.format)
        elif args.command == "manifest":
            report = runner_manifest(args.runner)
        elif args.command == "check-sources":
            report = check_source(args.upstream, args.source)
        else:
            report = {"schema": 1, "scope": SCOPE, "status": "UNAVAILABLE", "backend": "cuda",
                      "reason": "No local GPU API adapter is bundled. Run the pinned tensorfold-cuda external plan on a qualified CUDA host.",
                      "executed_tests": 0, "simulated_gpu": False}
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        report = {"schema": 1, "scope": SCOPE, "status": "FAIL", "errors": [str(exc)]}
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 3 if report["status"] == "UNAVAILABLE" else (1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
