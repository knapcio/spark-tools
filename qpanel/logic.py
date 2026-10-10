# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Null-calibrated item-pass arithmetic from the existing quality panel."""
import math

def diff(a,b):
    lost = [x['key'] for x,y in zip(a['results'],b['results']) if x['pass'] and not y['pass']]
    gained = [x['key'] for x,y in zip(a['results'],b['results']) if not x['pass'] and y['pass']]
    changed = [x['key'] for x,y in zip(a['results'],b['results']) if x['response_sha256']!=y['response_sha256']]
    # Ignore response IDs/timestamps/usage for semantic changed-output counts.
    output_changed = [x['key'] for x,y in zip(a['results'],b['results']) if x['output']!=y['output']]
    return dict(pass_to_fail=len(lost),fail_to_pass=len(gained),pass_changed=len(lost)+len(gained),net_loss=len(lost)-len(gained),
                pass_to_fail_items=lost,fail_to_pass_items=gained,raw_changed=len(changed),output_changed=len(output_changed),output_changed_items=output_changed)



def decide(a, a2, b):
    runs = (a, a2, b)
    keys = [row['key'] for row in a['results']]
    n = len(keys)
    if not n or len(set(keys)) != n:
        raise ValueError('nonempty unique item keys required')
    for run in runs:
        if run.get('status') != 'COMPLETE' or [r['key'] for r in run['results']] != keys:
            raise ValueError('incomplete/mismatched quality panel')
        for row in run['results']:
            if type(row.get('pass')) is not bool or type(row.get('truncated')) is not bool:
                raise ValueError('Boolean item score and truncation required')
            if not row.get('response_sha256') or 'output' not in row:
                raise ValueError('output/hash required')
    null = diff(a, a2); candidate = diff(a, b)
    counts = [sum(r['pass'] for r in x['results']) for x in runs]
    p = counts[0] / n
    se = math.sqrt(n * p * (1-p)); noise = abs(null['net_loss']); threshold = max(noise, se)
    return dict(decision='PASS' if candidate['net_loss'] <= threshold else 'FAIL',
                A_pass=counts[0], A2_pass=counts[1], B_pass=counts[2], items=n, A_rate=p,
                noise_net_items=noise, binomial_SE_items=se, threshold_items=threshold,
                truncations={arm:sum(r['truncated'] for r in run['results']) for arm,run in zip(('A','A2','B'),runs)},
                A_vs_A2=null, A_vs_B=candidate,
                interpretation='Candidate loss is within the preregistered noise tolerance.' if candidate['net_loss'] <= threshold else 'Candidate loss exceeds the preregistered noise tolerance.')
