# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Offline MoE route conversion, verification and TP4/EP replay."""
import argparse
import json
from pathlib import Path
import sys
from moereplay.routes import Trace
from moereplay.converters import convert_pt, convert_schema1
from moereplay.replay import replay, parse_proxy_widths
from moereplay.model import MIN_BOOTSTRAP
from moereplay.regression import reproduce, compare


def run_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    pt = sub.add_parser('convert-pt')
    pt.add_argument('input'); pt.add_argument('--out',required=True)
    pt.add_argument('--layer',type=int,default=3); pt.add_argument('--prompt-rows',type=int,default=2048)
    pt.add_argument('--model',default='GLM-5.3')
    cap = sub.add_parser('convert-schema1')
    cap.add_argument('input'); cap.add_argument('--out',required=True); cap.add_argument('--model',default='GLM-5.3')
    cap.add_argument('--source-status',choices=('measured','assumed'),required=True)
    check = sub.add_parser('check')
    check.add_argument('trace'); check.add_argument('--all-75',action='store_true'); check.add_argument('--min-steps',type=int,default=0); check.add_argument('--decode-only',action='store_true')
    run = sub.add_parser('replay')
    run.add_argument('trace'); run.add_argument('--cost-fit'); run.add_argument('--train-trace')
    run.add_argument('--samples',type=int,default=1000); run.add_argument('--bootstrap',type=int,default=200)
    run.add_argument('--seed',type=int,default=1234); run.add_argument('--replicate-h',type=int,default=0)
    run.add_argument('--rows-per-request',type=parse_proxy_widths,
                     help='prefill proxy c=rows comma list; defaults 1=4,2=3,4=3; omitted c keeps default; decode uses recorded rows')
    run.add_argument('--details',help='create-only streamed JSONL per step/layer/layout')
    run.add_argument('--extrapolate-layers',type=int)
    reg = sub.add_parser('regression')
    reg.add_argument('--fixtures',required=True)
    reg.add_argument('--reference',help='imbalance.json reference within supplied fixture directory')
    reg.add_argument('--version',choices=('frozen','current'),default='frozen')
    for cmd in (check,run,reg):
        cmd.add_argument('--out',help='create-only JSON output; otherwise stdout')
    # --check trace is the short capture-validation spelling promised in spec.
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == '--check':
        args_in[0] = 'check'
    a = p.parse_args(args_in)
    if any(getattr(a,k,0) < 0 for k in ('bootstrap','replicate_h','min_steps')) or getattr(a,'replicate_h',0)>256:
        p.error('bootstrap/min-steps >=0 and replicate-h in [0,256] required')
    if getattr(a,'samples',1) < 1 or 0 < getattr(a,'bootstrap',0) < MIN_BOOTSTRAP:
        p.error('samples >=1; bootstrap 0 or >=200 required')
    try:
        if a.command == 'convert-pt':
            result = convert_pt(a.input,a.out,a.layer,a.prompt_rows,a.model)
            print(json.dumps(result,indent=2)); return result
        if a.command == 'convert-schema1':
            result = convert_schema1(a.input,a.out,a.model,a.source_status)
            print(json.dumps(result,indent=2)); return result
        if getattr(a,'out',None) and Path(a.out).exists():
            raise ValueError('output exists')
        if a.command == 'regression':
            reference = a.reference or str(Path(a.fixtures)/('imbalance-current.json' if a.version=='current' else 'imbalance.json'))
            actual = reproduce(a.fixtures,a.version)
            errors = compare(actual,json.loads(Path(reference).read_text()))
            result = dict(status='measured',check='PASS' if all(e['absolute_error']<1e-9 for e in errors) else 'FAIL',
                          fields=len(errors),max_absolute_error=max(e['absolute_error'] for e in errors),errors=errors,
                          version=a.version,
                          scope='exact EP script compatibility; legacy final-segment exclusion only here; frozen c4 leakage only here')
        else:
            with Trace(a.trace) as trace:
                if a.command == 'check':
                    if a.decode_only and trace.manifest['step_semantics'] != 'decode-verify':
                        raise ValueError('decode capture required; prefill proxy rejected')
                    result = trace.check(range(3,78) if a.all_75 else None,a.min_steps)
                elif a.train_trace:
                    with Trace(a.train_trace) as train:
                        result = replay(trace,a.cost_fit,a.samples,a.bootstrap,a.seed,a.replicate_h,a.details,a.extrapolate_layers,train,rows_per_request=a.rows_per_request)
                else:
                    result = replay(trace,a.cost_fit,a.samples,a.bootstrap,a.seed,a.replicate_h,a.details,a.extrapolate_layers,rows_per_request=a.rows_per_request)
        out = json.dumps(result,indent=2,allow_nan=False)+'\n'
        if getattr(a,'out',None):
            with open(a.out,'x') as f:
                f.write(out)
        else:
            print(out,end='')
        if a.command == 'regression' and result['check'] != 'PASS':
            raise ValueError('regression failed')
        return result
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:
        p.exit(2,'error: '+str(e)+'\n')



def main(argv=None):
    run_cli(argv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
