# SPDX-License-Identifier: Apache-2.0
"""Retained quality-panel test counts and transport fixtures."""
import copy
import json
import math
from pathlib import Path
import pytest
from common.prereg import freeze, verify
from qpanel.cli import run
from qpanel.logic import decide, diff


def fixture(count):
    return dict(status='COMPLETE', results=[dict(key=str(i), **{'pass':i<count},
                response_sha256=str(i), output=dict(content=''), truncated=False) for i in range(116)])


def test_net_noise_directions_and_changed_output():
    a=fixture(100);b=copy.deepcopy(a)
    b['results'][0]['pass']=False;b['results'][1]['pass']=False;b['results'][100]['pass']=True
    d=diff(a,b)
    assert (d['pass_to_fail'],d['fail_to_pass'],d['net_loss']) == (2,1,1)
    assert d['output_changed'] == 0


@pytest.mark.parametrize('count', [100,98,97,96])
def test_original_threshold_boundary(count):
    result=decide(fixture(100),fixture(98),fixture(count))
    assert result['noise_net_items'] == 2
    assert result['binomial_SE_items'] == pytest.approx(math.sqrt(116*(100/116)*(16/116)))
    assert result['decision'] == ('PASS' if 100-count <= result['threshold_items'] else 'FAIL')


@pytest.mark.parametrize('fault', ['partial','duplicate','bad-score','bad-status'])
def test_invalid_panels(fault):
    a=fixture(100);b=copy.deepcopy(a)
    if fault=='partial':b['results'].pop()
    if fault=='duplicate':b['results'][0]['key']=b['results'][1]['key']
    if fault=='bad-score':b['results'][0]['pass']=1
    if fault=='bad-status':b['status']='INVALID'
    with pytest.raises(ValueError):decide(a,a,b)


class Transport:
    def __init__(self, content='', finish='length'):self.calls=[];self.content=content;self.finish=finish
    def ask(self,body,timeout):
        self.calls.append(body)
        return json.dumps(dict(choices=[dict(message=dict(content=self.content),finish_reason=self.finish)],
                               usage=dict(completion_tokens=1))).encode()


def test_portable_transport_keeps_frozen_body_and_truncation():
    body={'model':'example-model','messages':[{'role':'user','content':'Reply with exactly OK.'}],
          'max_tokens':4,'temperature':0,'stream':False}
    cfg={'tasks':[{'key':'probe','body':body,'expected':'OK'}]}
    t=Transport('OK')
    result=run(cfg,'A',t)
    assert t.calls==[body]
    assert result['results'][0]['pass'] is True
    assert result['results'][0]['truncated'] is True
    with pytest.raises(ValueError):run(cfg,'B',Transport('',finish='error'))


def test_preregistration_hash_source_and_create_only(tmp_path):
    src=tmp_path/'source.py';src.write_text('clean')
    p=tmp_path/'plan.json';cfg={'tasks':[]}
    freeze(cfg,p,[src]);assert verify(p,[src])[0]==cfg
    with pytest.raises(FileExistsError):freeze(cfg,p,[src])
    src.write_text('changed')
    with pytest.raises(ValueError,match='drift'):verify(p,[src])
    src.write_text('clean');p.write_text('{}')
    with pytest.raises(ValueError,match='drift'):verify(p,[src])
