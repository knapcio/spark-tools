# SPDX-License-Identifier: Apache-2.0
"""Packaging adapter for the existing CPU scheduler replay and policy sweep."""
import argparse
import itertools
from pathlib import Path
from common.config import dump, load as load_config
from lab.schedreplay.policy import Policy
from lab.schedreplay.replay import Replay, pareto
from lab.schedreplay.subsets import CAPTURE_REQUESTS, supported_subset
from lab.schedreplay.timing import Timing
from lab.schedreplay.workloads import load, synthetic, write


def comma(value, convert=str):
    try:
        result = [convert(x.strip()) for x in value.split(',')]
        if not result:
            raise ValueError()
        return result
    except (TypeError, ValueError):
        raise ValueError('invalid comma-separated list')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    sweep = sub.add_parser('sweep')
    for command in (run, sweep):
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--workload', type=Path, required=True)
        command.add_argument('--out', type=Path)
    sweep.add_argument('--chunks', default='512,1024,2048,4096')
    sweep.add_argument('--decode-steps', default='0,10,20,40,80')
    sweep.add_argument('--variants', default='none')
    sweep.add_argument('--admissions', default='on,off')
    sweep.add_argument('--max-seqs', default='4')
    sweep.add_argument('--include-requests', action='store_true')
    build = sub.add_parser('generate')
    build.add_argument('--name', choices=['agents', 'bursty', 'long-during-decode', 'newcomer'], default='agents')
    build.add_argument('--sessions', type=int, default=4)
    build.add_argument('--rounds', type=int, default=3)
    build.add_argument('--seed', type=int, default=0)
    build.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == 'generate':
            if args.sessions < 1 or args.rounds < 1 or args.sessions * args.rounds > 50000:
                raise ValueError('invalid synthetic workload bounds')
            works, meta = synthetic(args.name, args.seed, args.sessions, args.rounds)
            write(args.out, works, meta)
            return 0
        cfg = load_config(args.config)
        works, meta = load(args.workload)
        calibration = load_config(args.config.parent / cfg['timing'])

        def timing():
            return Timing(calibration)

        def runner(policy, seqs=None, admission=None):
            return Replay(policy, timing=timing(),
                          admission=cfg.get('admission', True) if admission is None else admission,
                          max_seqs=cfg.get('max_seqs', 4) if seqs is None else seqs,
                          max_steps=cfg.get('max_steps', 2000000),
                          max_time_s=cfg.get('max_time_s', 86400)).run(works, meta)

        if args.command == 'run':
            policy = Policy.parse(cfg.get('policy', 'fair:4096:40'), variant=cfg.get('variant', 'none'))
            result = runner(policy)
        else:
            chunks, steps = comma(args.chunks, int), comma(args.decode_steps, int)
            variants, admissions, seqs = comma(args.variants), comma(args.admissions), comma(args.max_seqs, int)
            if not set(admissions) <= {'on', 'off'}:
                raise ValueError('invalid admissions')
            if not set(seqs) <= {4, 6}:
                raise ValueError('invalid max-seqs')
            policies = [Policy(off=True, variant=v) for v in variants]
            policies += [Policy.parse('fair:%d:%d' % (c, n), variant=v)
                         for c, n, v in itertools.product(chunks, steps, variants)]
            combinations = len(policies) * len(admissions) * len(seqs)
            if combinations > 1000:
                raise ValueError('sweep exceeds 1000-case bound')
            if args.include_requests and combinations * len(works) > 50000:
                raise ValueError('include-requests exceeds 50000-row output bound')
            baseline = runner(Policy(), seqs=4, admission=True)
            results = []
            for policy, admission, seq in itertools.product(policies, admissions, seqs):
                r = runner(policy, seqs=seq, admission=admission == 'on')
                if not args.include_requests:
                    r['status_counts'] = {status: sum(x['status'] == status for x in r['requests'])
                                          for status in {x['status'] for x in r['requests']}}
                    del r['requests']
                    r.pop('decode_during_prefills', None)
                results.append(r)
            result = {'supported_subset': supported_subset(works, meta, timing),
                      'schema': 'schedreplay-sweep v2', 'evidence': 'estimated offline screen',
                      'capture_requests': CAPTURE_REQUESTS,
                      'baseline': {k: baseline[k] for k in ('policy', 'admission', 'max_num_seqs', 'ttft_s',
                                   'ttft_by_prompt_bucket', 'max_decode_stall_s', 'decode_during_prefills_tok_s',
                                   'prefill_throughput_tok_s', 'queue_wait_s', 'completed')},
                      'pareto': pareto(results, baseline), 'results': results}
        dump(result, args.out)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
