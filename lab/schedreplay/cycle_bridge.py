# SPDX-License-Identifier: Apache-2.0
"""Validation of caller-supplied cycle timing and acceptance contracts."""
import math


def validate_contract(point):
    """Reject costs/acceptance for a different execution contract."""
    c = point.get('c')
    k = 3 if c == 1 else 2
    policy = 'kstop' if c == 1 else 'uniform_k2'
    if (type(c) is not int or c not in (1,2,4) or type(point.get('k')) is not int
            or point['k'] != k or point.get('policy') != policy
            or type(point.get('M')) is not int or point['M'] != c*(k+1)
            or point.get('capture_layout') != 'reuse'):
        raise ValueError('incompatible serving cycle contract: c1 K3/kstop M4; c2/c4 K2/reuse M6/M12 required')
    survival = point.get('survival')
    committed = point.get('committed_per_cycle')
    def finite(x): return type(x) in (int,float) and math.isfinite(x)
    if (not isinstance(survival,list) or len(survival) != k or
            any(not finite(x) or not 0 <= x <= 1 for x in survival) or
            any(a < b for a,b in zip(survival,survival[1:])) or
            not finite(committed) or not math.isclose(committed,1+sum(survival),rel_tol=1e-9)):
        raise ValueError('invalid serving cycle acceptance contract')
    if not finite(point.get('ms')) or point['ms'] <= 0:
        raise ValueError('invalid cycle timing')

