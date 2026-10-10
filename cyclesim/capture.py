#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write planned HTTP bodies or hash an already-collected cell; never sends HTTP."""
import argparse
import json
import uuid
from pathlib import Path
from cyclesim.utilities import dump, sha256

PROMPTS = {
 'prose': [
  'Explain how a town can keep its public library useful for residents of different ages. Compare practical options and write a detailed, connected discussion.',
  'Describe how a small team should preserve knowledge when experienced members leave. Discuss concrete examples, tradeoffs, and a practical plan.',
  'Explain how to plan a community garden through its first year. Discuss weather, tools, volunteers, and ways to learn from mistakes in detail.',
  'Discuss how a family can organize and preserve old photographs. Explain the practical choices and develop a careful, detailed process.'],
 'code': [
  'Write and explain a Python standard-library streaming CSV aggregator with validation, bounded memory, error handling, and useful tests.',
  'Design and implement a Python standard-library task queue with retries and cancellation. Explain the state transitions and provide tests.',
  'Write a Python standard-library incremental JSON event reader. Explain handling of chunk boundaries and malformed input, and provide tests.',
  'Implement a Python standard-library dependency graph scheduler. Explain cycle detection, deterministic ordering, and test edge cases.']}


def plan(out):
    out.mkdir(parents=True, exist_ok=True)
    cells, request_catalog = [], []
    for domain in ('prose','code'):
        for c in (1,4):
            for arm in ('A1','P','A2'):
                cell='%s-c%d-%s'%(domain,c,arm)
                requests=[]
                for i in range(c):
                    rid=cell+'-s%d'%i
                    body={'model':'example-model','messages':[{'role':'user','content':PROMPTS[domain][i]}],
                          'max_tokens':640, 'min_tokens':640, 'ignore_eos':True, 'seed':i, 'stream':True,
                          'stream_options':{'include_usage':True}, 'cache_salt':rid+'-'+uuid.uuid4().hex,
                          'chat_template_kwargs':{'enable_thinking':True}}
                    dump(out/(rid+'.request.json'),body)
                    request_catalog.append({'file':rid+'.request.json','sha256':sha256(out/(rid+'.request.json')),
                                            'body':body,'status':'planned'})
                    requests.append(rid+'.request.json')
                cells.append({'id':cell,'domain':domain,'c':c,'arm':arm,'requests':requests,'status':'planned'})
    dump(out/'prompts.json', {'schema':1, 'status':'planned', 'prompts':PROMPTS, 'requests':request_catalog})
    dump(out/'capture-plan.json',{'schema':1,'status':'planned; offline generator sends no HTTP','cells':cells,
                                  'temperature':'omitted: server default', 'max_profiled_cells_per_boot':1,
                                  'routing_capture':None})


def manifest(root, mode, c, domain, unprofiled):
    files=[]
    companions={'prom-before.prom':'prom_before','prom-after.prom':'prom_after','client-timings.jsonl':'client_timings',
                'env.json':'env','graph-sizes.json':'graph_sizes','A1.json':'a1','A2.json':'a2','prompts.json':'prompts'}
    if mode=='profile':companions['P.json']='profile_client'
    for path,role in companions.items():
        p=root/path
        if not p.is_file():raise ValueError('missing companion '+path)
        files.append({'path':path,'role':role,'sha256':sha256(p)})
    if (root/'kernel-bindings.json').is_file():companions['kernel-bindings.json']='kernel_bindings'
    if 'kernel_bindings' in companions.values():
        path=root/'kernel-bindings.json'
        files.append({'path':'kernel-bindings.json','role':'kernel_bindings','sha256':sha256(path)})
    traces=sorted(set(root.rglob('*.pt.trace.json.gz'))|set(root.rglob('*.pt.trace.json')))
    import re
    for path in traces:
        m=re.search(r'rank(\d+)',str(path.relative_to(root)))
        if not m:raise ValueError('trace file or directory must identify rankN')
        files.append({'path':path.relative_to(root).as_posix(),'role':'trace','rank':int(m[1]),'sha256':sha256(path)})
    env=json.loads((root/'env.json').read_text())
    doc={'schema':1,'mode':mode,'c':c,'k':3 if c==1 else 2,'domain':domain,
         **{key:env[key] for key in ('boot','head','policy','capture_layout','pad_hygiene')},'unprofiled_cycle_ms':unprofiled,
         'files':files,'routing_capture':None,'status':'measured file hashes; metadata operator supplied'}
    dump(root/'manifest.json',doc)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--mode',choices=('profile','companions-only'),default='profile')
    p.add_argument('--c',type=int,choices=(1,2,4),default=1)
    p.add_argument('--domain',choices=('prose','code'),default='prose')
    p.add_argument('--unprofiled-cycle-ms',type=float)
    a=p.parse_args()
    if a.manifest:
        manifest(a.manifest,a.mode,a.c,a.domain,a.unprofiled_cycle_ms)
    elif a.out:plan(a.out)
    else:p.error('--out or --manifest required')


if __name__=='__main__':main()
