# SPDX-License-Identifier: Apache-2.0
"""Existing offline A/B/A-prime decision and T0 gate, plus portable request transport."""
import argparse
import json
import os
from pathlib import Path
from common.config import dump, load
from common.endpoint import HTTP
from common.prereg import freeze, verify
from abharness.logic import compare_t0, decision

SOURCES = (Path(__file__), Path(__file__).with_name('logic.py'),
           Path(__file__).parents[1] / 'common/prereg.py',
           Path(__file__).parents[1] / 'common/config.py')

RULE = {'rule': 'paired-cycle-v1', 'panels': 3, 'prompts_per_kind': 10,
        'prose_saving_ms': 0.2, 'committed_ratio_min': 0.995,
        'code_cycle_regression_max': 0.005}


def validate_rule(config):
    if config != RULE:
        raise ValueError('preregistration must declare the existing fixed rule')
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('preregister'); f.add_argument('--config', type=Path, required=True)
    f.add_argument('--prereg', type=Path, required=True)
    request = sub.add_parser('request'); request.add_argument('--config', type=Path, required=True)
    request.add_argument('--out', type=Path)
    d = sub.add_parser('decide'); d.add_argument('--input', type=Path, required=True)
    d.add_argument('--prereg', type=Path, required=True); d.add_argument('--out', type=Path)
    t = sub.add_parser('near-tie'); t.add_argument('--input', type=Path, required=True); t.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == 'preregister':
            freeze(validate_rule(load(args.config)), args.prereg, SOURCES)
            return 0
        if args.command == 'request':
            cfg = load(args.config)
            transport = HTTP(cfg['endpoint'], os.environ.get(cfg.get('token_env', 'OPENAI_API_KEY')))
            try: result = json.loads(transport.ask(cfg['body'], cfg.get('timeout_s', 90)))
            finally: transport.cancel()
            if result.get('error') or len(result.get('choices', [])) != 1:
                raise ValueError('failed response')
        elif args.command == 'near-tie':
            data = load(args.input)
            result = compare_t0(data['A'], data['B'], data.get('variant', 'draft'), data.get('reference'))
        else:
            cfg, sha = verify(args.prereg, SOURCES)
            data = load(args.input)
            if data.get('prereg_sha256') != sha: raise ValueError('preregistration drift')
            validate_rule(cfg)
            result = decision(data['A'], data['Aprime'], data['B'])
            result['prereg_sha256'] = sha
        dump(result, args.out)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.error(str(exc))
    return int(args.command in ('decide', 'near-tie') and result['decision'] not in ('PASS','PASS_NEAR_TIE'))
