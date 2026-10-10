# SPDX-License-Identifier: Apache-2.0
"""Package the existing item-pass rule and HTTP transport with explicit configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from common.config import dump, load
from common.endpoint import HTTP
from common.prereg import freeze, verify
from qpanel.logic import decide

SOURCES = (Path(__file__), Path(__file__).with_name('logic.py'),
           Path(__file__).parents[1] / 'common/endpoint.py',
           Path(__file__).parents[1] / 'common/prereg.py',
           Path(__file__).parents[1] / 'common/config.py')

DECISION_RULE = 'loss <= max(abs(null net loss), sqrt(n*p*(1-p)))'


def validate_config(config):
    if config.get('rule', 'null-pass-v1') != 'null-pass-v1' or config.get('decision_rule', DECISION_RULE) != DECISION_RULE:
        raise ValueError('preregistration must declare the existing null-pass rule')
    tasks = config.get('tasks')
    n = config.get('items', len(tasks) if isinstance(tasks, list) else 0)
    if type(n) is not int or n <= 0:
        raise ValueError('positive preregistered item count required')
    if tasks is not None:
        if not isinstance(tasks, list) or len(tasks) != n or len({t['key'] for t in tasks}) != n:
            raise ValueError('nonempty unique tasks matching the item count required')
        for task in tasks:
            if not isinstance(task['key'], str) or not task['key'] or not isinstance(task['body'], dict) or not isinstance(task['expected'], str):
                raise ValueError('task key, request body and exact expected text required')
    return config


def run(config, arm, transport):
    validate_config(config)
    rows = []
    tasks = config['tasks']
    if not tasks or len({t['key'] for t in tasks}) != len(tasks):
        raise ValueError('nonempty unique tasks required')
    for task in tasks:
        raw = transport.ask(task['body'], config.get('timeout_s', 90))
        response = json.loads(raw)
        if response.get('error') or len(response.get('choices', [])) != 1:
            raise ValueError('failed response')
        choice = response['choices'][0]; message = choice['message']; finish = choice.get('finish_reason')
        if finish not in ('stop', 'length'): raise ValueError('missing/invalid finish_reason')
        content = message.get('content') or ''
        rows.append(dict(key=task['key'], **{'pass': content == task['expected']},
                         truncated=finish == 'length', response_sha256=hashlib.sha256(raw).hexdigest(),
                         output=dict(content=content, reasoning=message.get('reasoning_content') or message.get('reasoning') or '',
                                     finish_reason=finish)))
    return dict(status='COMPLETE', arm=arm, results=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('preregister'); f.add_argument('--config', type=Path, required=True)
    f.add_argument('--prereg', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--prereg', type=Path, required=True)
    r.add_argument('--arm', choices=('A','A2','B'), required=True); r.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('decide'); d.add_argument('--prereg', type=Path, required=True)
    for name in ('a', 'a2', 'b'): d.add_argument('--'+name, type=Path, required=True)
    d.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == 'preregister':
            freeze(validate_config(load(args.config)), args.prereg, SOURCES)
        else:
            cfg, sha = verify(args.prereg, SOURCES)
            validate_config(cfg)
            if args.command == 'run':
                transport = HTTP(cfg['endpoints'][args.arm], os.environ.get(cfg.get('token_env', 'OPENAI_API_KEY')))
                try: result = run(cfg, args.arm, transport)
                finally: transport.cancel()
                result['prereg_sha256'] = sha
            else:
                reports = [load(getattr(args, n)) for n in ('a', 'a2', 'b')]
                if any(x.get('prereg_sha256') != sha or x.get('arm') != arm for x,arm in zip(reports, ('A','A2','B'))):
                    raise ValueError('arm/preregistration drift')
                n = cfg.get('items', len(cfg.get('tasks', [])))
                if any(len(r['results']) != n for r in reports):
                    raise ValueError('preregistered item count drift')
                if 'tasks' in cfg and any([x['key'] for x in r['results']] != [t['key'] for t in cfg['tasks']] for r in reports):
                    raise ValueError('preregistered item identity drift')
                result = decide(*reports); result['prereg_sha256'] = sha
            dump(result, args.out)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.error(str(exc))
    return int(args.command == 'decide' and result['decision'] != 'PASS')
