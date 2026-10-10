# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import tempfile
from pathlib import Path
from lab import utilities


def test_existing_trace_stream_handles_gzip_and_metadata(tmp_path):
    p=tmp_path/'trace.json.gz'
    value={'traceEvents':[{'ts':0,'dur':2},{'ts':1,'dur':3}], 'displayTimeUnit':'ns'}
    with gzip.open(p,'wt') as f:json.dump(value,f)
    meta={}
    assert list(utilities.trace_events(p,meta))==value['traceEvents']
    assert meta['displayTimeUnit']=='ns'
    assert utilities.union_us([(0,2),(1,4),(6,7)])==5
    assert utilities.stats([1,2,3])['median']==2


def test_existing_acceptance_counters(tmp_path):
    def text(drafts, accepted):
        return '\n'.join([
            f'vllm:spec_decode_num_drafts_total {drafts}',
            f'vllm:spec_decode_num_draft_tokens_total {drafts*2}',
            f'vllm:spec_decode_num_accepted_tokens_total {sum(accepted)}',
            *[f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}} {v}' for i,v in enumerate(accepted)]])
    a=tmp_path/'before.prom';b=tmp_path/'after.prom'
    a.write_text(text(0,[0,0]));b.write_text(text(10,[8,4]))
    result=utilities.acceptance(a,b,2)
    assert result['survival']==[.8,.4]
    assert result['committed']==2.2


def test_trace_stream_handles_numbers_split_at_chunk_boundaries():
    header = '{"traceEvents": [], "numeric": '
    for prefix, suffix, expected in (('1.', '25', 1.25), ('1e', '3', 1000),
                                     ('-1.25e-', '3', -.00125)):
        document = header + ' ' * (65536 - len(header) - len(prefix)) + prefix + suffix + '}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(document)
            metadata = {}
            assert list(utilities.trace_events(path, metadata)) == []
            assert metadata['numeric'] == expected
