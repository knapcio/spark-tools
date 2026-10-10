# SPDX-License-Identifier: Apache-2.0
"""One command for the existing scheduler, rank and file/counter utilities."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "conformance":
        from lab.conformance.cli import main as run
        return run(args[1:])
    if args and args[0] == "observations":
        from lab.observations.cli import main as run
        return run(args[1:])
    if args and args[0] == "experiment":
        from lab.experiment.cli import main as run
        return run(args[1:])
    if args and args[0] == "schedreplay":
        from lab.schedreplay.cli import main as run
        return run(args[1:])
    if args and args[0] == "rankharness":
        from lab.rankharness.cli import main as run
        return run(args[1:])
    if args and args[0] == "exl3":
        from lab.exl3.cli import main as run
        return run(args[1:])
    if args and args[0] == "exlr8":
        from lab.exlr8.cli import main as run
        return run(args[1:])
    if args and args[0] == "tensorfold":
        from lab.tensorfold.cli import main as run
        return run(args[1:])
    if args and args[0] in {"sha256", "acceptance"}:
        from lab.cli import main as run
        return run(args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["conformance", "observations", "experiment", "schedreplay", "rankharness", "exl3", "exlr8", "tensorfold", "sha256", "acceptance"],
                        help="choose a utility; add --help after it for its options")
    parser.parse_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
