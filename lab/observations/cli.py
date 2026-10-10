# SPDX-License-Identifier: Apache-2.0
"""Import pinned local runtime measurements and calibrate exact lookup cells."""
import argparse
import json
import os
import sys

from .core import InvalidObservation, PINS, read_local, strict_json
from .importers import import_observations
from .calibration import calibrate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="import a pinned local source")
    imp.add_argument("--adapter", required=True, choices=sorted(PINS))
    imp.add_argument("--input", required=True)
    imp.add_argument("--metadata", required=True)
    imp.add_argument("--before", help="before snapshot for the Prometheus interval adapter")
    imp.add_argument("--client-capture", help="versioned native streaming companion proving interval responses succeeded")
    imp.add_argument("--output", help="new JSON file; default stdout")
    cal = commands.add_parser("calibrate", help="fit exact cells with an explicit disjoint split")
    cal.add_argument("--input", required=True, action="append", help="normalized observation JSON; repeat for multiple sources")
    cal.add_argument("--split", required=True)
    cal.add_argument("--output", help="new JSON file; default stdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            result = import_observations(args.adapter, read_local(args.input), strict_json(read_local(args.metadata)),
                                         before=read_local(args.before) if args.before else None,
                                         client_capture=read_local(args.client_capture) if args.client_capture else None)
        else:
            result = calibrate([strict_json(read_local(path)) for path in args.input], strict_json(read_local(args.split)))
        output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            # Exclusive creation prevents overwriting a source, symlink or existing report.
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    except (InvalidObservation, OSError, KeyError, TypeError, OverflowError, RecursionError):
        # Exception messages can contain input paths or source strings; suppress them.
        print("observations: invalid local input or output; no observation exported", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
