# SPDX-License-Identifier: Apache-2.0
"""Existing mac-tools file/counter utilities and CPU lab dispatch."""
import argparse
from pathlib import Path
from common.config import dump
from lab import utilities


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    h = sub.add_parser('sha256'); h.add_argument('file', type=Path)
    a = sub.add_parser('acceptance')
    a.add_argument('--before', type=Path, required=True)
    a.add_argument('--after', type=Path, required=True)
    a.add_argument('--k', type=int, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == 'sha256':
            dump({'sha256': utilities.sha256(args.file)})
        else:
            if args.k < 1: raise ValueError('positive draft width required')
            dump(utilities.acceptance(args.before, args.after, args.k))
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
