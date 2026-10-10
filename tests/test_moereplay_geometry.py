# SPDX-License-Identifier: Apache-2.0
"""v0.3 geometry contract, including frozen v1 bytes and uint16 boundaries."""
from collections import Counter
import contextlib
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from moereplay.cli import run_cli
from moereplay.converters import convert_schema1, convert_pt
from moereplay.geometry import Geometry, LEGACY_GEOMETRY, Layout, PRESETS, load_geometry, resolve_layouts
from moereplay.model import Costs, placement, rank_work
from moereplay.replay import replay
from moereplay.routes import Writer, Trace, sha256, valid_ids


def make_trace(path, geometry=None, steps=4, request_prefix='r', semantics='decode-verify'):
    selected = geometry or LEGACY_GEOMETRY
    writer = Writer(path, [3], 'fixture', dict(status='assumed', kind='synthetic'), semantics, geometry)
    for step in range(steps):
        rows = [dict(request=request_prefix+str(step), position=i, padding=False,
                     phase='prefill' if semantics == 'prefill-proxy' else 'decode',
                     k=0 if semantics == 'prefill-proxy' else 1, concurrency=1)
                for i in range(4 if semantics == 'prefill-proxy' else 2)]
        experts = [[(step+i+j) % selected.num_experts for j in range(selected.top_k)] for i in range(len(rows))]
        writer.append(step, rows, {3:experts})
    return writer.finish()


def bound_fit(path, geometry, layouts=None, records=None):
    selected = resolve_layouts(geometry, layouts)
    payload = dict(schema='moe-cost-fit v2', geometry=geometry.to_dict(),
                   geometry_fingerprint=geometry.fingerprint, layouts=[l.to_dict() for l in selected],
                   per_layout_M=records or [dict(layout=selected[0].name, M=2, a_us=2,
                       b_us_per_distinct_local_expert=3, points=[dict(D=geometry.top_k), dict(D=geometry.num_experts)])])
    path.write_text(json.dumps(payload))
    return payload


@pytest.mark.parametrize('bad', [dict(num_experts=0, top_k=1, world_size=1),
    dict(num_experts=65537, top_k=1, world_size=1), dict(num_experts=True, top_k=1, world_size=1),
    dict(num_experts=8, top_k=0, world_size=2), dict(num_experts=8, top_k=9, world_size=2),
    dict(num_experts=8, top_k=2., world_size=2), dict(num_experts=8, top_k=2, world_size=0),
    dict(num_experts=8, top_k=2, world_size=False), dict(num_experts=8, top_k=2, world_size=2, expert_bytes=-1),
    dict(num_experts=8, top_k=2, world_size=2, expert_bytes=True)])
def test_invalid_geometry(bad):
    with pytest.raises(ValueError):
        Geometry.from_dict(bad)


def test_geometry_immutable_and_canonical(tmp_path):
    geometry = Geometry(8, 2, 2)
    with pytest.raises(FrozenInstanceError):
        geometry.top_k = 4
    assert load_geometry(geometry.to_dict()) == geometry
    assert load_geometry(json.dumps(dict(world_size=2, top_k=2, num_experts=8))).fingerprint == geometry.fingerprint
    path = tmp_path/'geometry.json'
    path.write_text(json.dumps(geometry.to_dict()))
    assert load_geometry(str(path)) == geometry
    for value in ({}, dict(geometry.to_dict(), hidden_size=3), []):
        with pytest.raises(ValueError):
            load_geometry(value)
    assert Geometry(8, 2, 2, 128).fingerprint != geometry.fingerprint
    assert Geometry(8, 2, 4).fingerprint != geometry.fingerprint


@pytest.mark.parametrize('bad', ['TP3', 'EP3', 'EP0', 'TP02', 'TP2/EP2',
    dict(name='bad', ep=True, tp=2), dict(name='bad', ep=2, tp=2),
    dict(name='bad', ep=3, tp=1), dict(name='bad', ep=2),
    dict(name='', ep=1, tp=2)])
def test_invalid_layouts(bad):
    with pytest.raises(ValueError):
        resolve_layouts(Geometry(8, 2, 2), [bad])


def test_layout_capacity_and_duplicates():
    with pytest.raises(ValueError, match='duplicate'):
        resolve_layouts(Geometry(8, 2, 2), ['TP2', 'TP2'])
    with pytest.raises(ValueError, match='exceed'):
        resolve_layouts(Geometry(2, 1, 4), ['EP4'])
    with pytest.raises(ValueError, match='divide'):
        resolve_layouts(Geometry(8, 2, 2, 127), ['TP2'])
    assert [l.to_dict() for l in resolve_layouts(Geometry(12, 2, 6))] == [
        dict(name='TP6', ep=1, tp=6), dict(name='EP2xTP3', ep=2, tp=3),
        dict(name='EP3xTP2', ep=3, tp=2), dict(name='EP6', ep=6, tp=1)]


@pytest.mark.parametrize('geometry,row', [(Geometry(512, 2, 2), [256, 511]),
                                         (Geometry(65536, 2, 1), [0, 65535]),
                                         (Geometry(1, 1, 1), [0])])
def test_uint16_roundtrip_and_byte_order(tmp_path, geometry, row):
    writer = Writer(tmp_path/'trace', [0], 'synthetic', dict(status='assumed'), 'decode-verify', geometry)
    writer.append(0, [dict(request='a', position=0, padding=False, phase='decode', k=0, concurrency=1)], {0:[row]})
    manifest = writer.finish()
    assert manifest['schema'] == 'moe-routes v2'
    assert manifest['data'][0]['byte_order'] == 'little'
    assert (tmp_path/'trace/layer-0.u16le').read_bytes() == struct.pack('<%dH' % len(row), *row)
    with Trace(tmp_path/'trace') as trace:
        assert trace.row(0, 0) == tuple(row)
        assert trace.geometry == geometry
        assert trace.check()['structural_check'] == 'PASS'


@pytest.mark.parametrize('row', [[0, 65536], [0, -1], [0, True], [0, 0], [0], [0, 1, 2]])
def test_uint16_invalid_ids(row):
    with pytest.raises(ValueError):
        valid_ids(row, Geometry(65536, 2, 1))


@pytest.mark.parametrize('field,value', [('geometry_fingerprint', '0'*64), ('top_k', 3), ('num_experts', 9)])
def test_manifest_geometry_tampering(tmp_path, field, value):
    make_trace(tmp_path/'trace', Geometry(8, 2, 2))
    path = tmp_path/'trace/manifest.json'
    payload = json.loads(path.read_text()); payload[field] = value; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='geometry'):
        Trace(tmp_path/'trace')


@pytest.mark.parametrize('kind', ['big-endian', 'length', 'ids'])
def test_v2_binary_rejections(tmp_path, kind):
    make_trace(tmp_path/'trace', Geometry(8, 2, 2))
    manifest_path = tmp_path/'trace/manifest.json'; payload = json.loads(manifest_path.read_text())
    data_path = tmp_path/'trace/layer-3.u16le'
    if kind == 'big-endian':
        payload['data'][0]['byte_order'] = 'big'
    elif kind == 'length':
        data_path.write_bytes(data_path.read_bytes()[:-1])
    else:
        data_path.write_bytes(struct.pack('<H', 8)+data_path.read_bytes()[2:])
    payload['data'][0]['sha256'] = sha256(data_path); manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        with Trace(tmp_path/'trace') as trace:
            trace.check()


@pytest.mark.parametrize('world', [1, 2, 3, 4, 6, 8])
def test_multi_world_hand_calculation(world):
    geometry = Geometry(12, 3, world, 480)
    for layout in resolve_layouts(geometry):
        owner = placement(layout.ep, 'contiguous', geometry=geometry)
        routes = [(0, 5, 11), (0, 5, 11)]
        result = rank_work(routes, layout, owner, geometry=geometry)
        group_distinct = [len({e for e in routes[0] if owner[e] == group}) for group in range(layout.ep)]
        assert result['D_r'] == [d for d in group_distinct for _ in range(layout.tp)]
        assert result['mean_rank_D'] == 3/layout.ep
        assert result['bytes_r'] == [d*480//layout.tp for d in result['D_r']]
        assert sum(result['routed_slots_r']) == 6*layout.tp
        assert len(result['D_r']) == world
        hot = rank_work(routes, layout, owner, hottest=(0, 5, 11), geometry=geometry)
        assert sum(hot['routed_slots_r']) == 6*layout.tp


def test_nondivisible_expert_count():
    geometry = Geometry(10, 2, 3)
    for name in ('contiguous', 'round-robin', 'greedy'):
        owners = placement(3, name, Counter(), geometry)
        assert len(owners) == 10
        assert max(Counter(owners).values()) == 4
    assert placement(3, 'greedy', Counter(), geometry) == placement(3, 'contiguous', geometry=geometry)


@pytest.mark.parametrize('owners,hot,routes', [([0]*7, (), [(0, 1)]),
    ([2]*8, (), [(0, 1)]), ([False]*8, (), [(0, 1)]), ([0]*8, (8,), [(0, 1)]),
    ([0]*8, (0, 0), [(0, 1)]), ([0]*8, (), [(0, 8)]), ([0]*8, (), [])])
def test_invalid_rank_work(owners, hot, routes):
    with pytest.raises(ValueError):
        rank_work(routes, 'EP2', owners, hot, Geometry(8, 2, 2))


def test_v1_byte_snapshot(tmp_path):
    # Recorded independently from the unmodified HEAD v1 writer.
    writer = Writer(tmp_path/'trace', [3], 'fixture', dict(status='assumed', kind='synthetic'), 'decode-verify')
    rows = [dict(request='a', position=i, padding=False, phase='decode', k=1, concurrency=1) for i in range(2)]
    writer.append(0, rows, {3:[list(range(8)), list(range(248, 256))]}); writer.finish()
    assert {p.name:sha256(p) for p in (tmp_path/'trace').iterdir()} == {
        'rows.jsonl':'39926be90a3046003a23a104a047fb649d0e63b388dcce9abd3fe569f35ff7eb',
        'manifest.json':'35582ef273e9e31aa11a5de7a03f835c61be0c8a6e2978c70a8f9e97278cbf08',
        'layer-3.u8':'b00d0296e5c88b2a7a6f50872e390cb34c1a28f11a5b1ce8a8d53689a4a4c2cf'}
    with Trace(tmp_path/'trace') as trace:
        assert trace.geometry == LEGACY_GEOMETRY
        assert type(trace.row(3, 0)) is bytes


def test_uncalibrated_replay_is_analysis_only(tmp_path):
    geometry = Geometry(8, 2, 2)
    make_trace(tmp_path/'trace', geometry)
    with Trace(tmp_path/'trace') as trace:
        result = replay(trace, bootstrap_samples=0, replicate_h=8, details=tmp_path/'details.jsonl')
    assert result['schema'] == 'moe-replay v2'
    assert result['cyclesim_what_if'] == []
    assert result['collective']['delta_ms'] is None
    for layer in result['per_layer'].values():
        for cell in layer.values():
            assert cell['layer_ms'] is None
            assert cell['D_union'] > 0
            assert len(cell['D_r']) == 2
            assert cell['bytes_r'] == [None, None]
    for cell in result['cycles'].values():
        assert cell['cycle_kernel_ms']['status'] == 'unavailable'
        assert cell['cycle_kernel_ms']['mean'] is None
        assert cell['delta_vs_baseline_ms']['mean'] is None
        assert not cell['delta_vs_baseline_ms']['calibration_eligible']
        assert cell['serving_compatible'] is None
    assert result['placement_fits']['3']['training_rows'] == 4
    assert len(result['placement_fits']['3']['frequencies']) == 8
    assert all(record['time']['layer_us'] is None for record in map(json.loads, (tmp_path/'details.jsonl').read_text().splitlines()))


def test_prefill_geometry_replay(tmp_path):
    make_trace(tmp_path/'trace', Geometry(16, 3, 3), semantics='prefill-proxy')
    with Trace(tmp_path/'trace') as trace:
        result = replay(trace, samples=2, bootstrap_samples=0)
    assert result['holdout']['unit'] == 'prompt'
    assert result['baseline_layout'] == 'TP3'
    assert len(result['cycles']) == 36


def test_cli_wide_geometry_replica_limit_and_layout(tmp_path):
    geometry = load_geometry('synthetic-wide')
    make_trace(tmp_path/'trace', geometry)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_cli(['replay', str(tmp_path/'trace'), '--preset', 'synthetic-wide',
                          '--replicate-h', '300', '--layout', 'EP8', '--bootstrap', '0'])
    assert result['layouts'] == [dict(name='EP8', ep=8, tp=1)]
    assert len(result['placement_fits']['3']['hottest']) == 300
    assert len(result['cycles']['observed/c1_M2_K1/EP8/replicate-hottest']['bank_bytes_r']) == 8


def test_v2_legacy_geometry_does_not_implicitly_transfer_prior(tmp_path):
    make_trace(tmp_path/'trace', LEGACY_GEOMETRY)
    with Trace(tmp_path/'trace') as trace:
        result = replay(trace, bootstrap_samples=0)
    assert all(row['cycle_kernel_ms']['mean'] is None for row in result['cycles'].values())
    assert all(row['bytes_status'] == 'estimated' for row in result['per_layer']['3'].values())


def test_explicit_layout_and_bytes_replay(tmp_path):
    make_trace(tmp_path/'trace', Geometry(10, 2, 3, 300))
    with Trace(tmp_path/'trace') as trace:
        result = replay(trace, bootstrap_samples=0, layouts=[Layout('custom', 3, 1)])
    assert result['layouts'] == [dict(name='custom', ep=3, tp=1)]
    assert result['baseline_layout'] == 'custom'
    assert result['cycles']['observed/c1_M2_K1/custom/contiguous']['bank_bytes_r'] == [1200, 900, 900]


@pytest.mark.parametrize('train_geometry', [Geometry(8, 3, 2), Geometry(8, 2, 4), Geometry(9, 2, 2), Geometry(8, 2, 2, 128)])
def test_train_geometry_mismatch(tmp_path, train_geometry):
    make_trace(tmp_path/'test', Geometry(8, 2, 2))
    make_trace(tmp_path/'train', train_geometry, request_prefix='other')
    with Trace(tmp_path/'test') as trace, Trace(tmp_path/'train') as train, pytest.raises(ValueError, match='geometry'):
        replay(trace, train_trace=train, bootstrap_samples=0)


def test_v1_train_v2_geometry_mismatch(tmp_path):
    make_trace(tmp_path/'test')
    make_trace(tmp_path/'train', Geometry(256, 8, 4), request_prefix='other')
    with Trace(tmp_path/'test') as trace, Trace(tmp_path/'train') as train, pytest.raises(ValueError, match='geometry'):
        replay(trace, train_trace=train, bootstrap_samples=0)


@pytest.mark.parametrize('kind', ['geometry', 'fingerprint', 'layout', 'unbound', 'schema', 'missing-layouts'])
def test_cost_binding_rejections(tmp_path, kind):
    geometry = Geometry(8, 2, 2)
    path = tmp_path/'fit.json'; payload = bound_fit(path, geometry)
    if kind == 'geometry':
        payload['geometry'] = Geometry(9, 2, 2).to_dict()
        payload['geometry_fingerprint'] = Geometry(9, 2, 2).fingerprint
    elif kind == 'fingerprint':
        payload['geometry_fingerprint'] = 'x'
    elif kind == 'layout':
        payload['layouts'][0] = dict(name='TP2', ep=2, tp=1)
    elif kind == 'unbound':
        del payload['geometry']; del payload['geometry_fingerprint']
    elif kind == 'schema':
        payload['schema'] = 'unknown'
    else:
        del payload['layouts']
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        Costs(path, geometry)


def test_geometry_bound_fit_and_missing_cells(tmp_path):
    geometry = Geometry(8, 2, 2)
    path = tmp_path/'fit.json'
    bound_fit(path, geometry, records=[dict(layout='TP2', M=2, a_us=2,
        b_us_per_distinct_local_expert=3, points=[dict(D=2), dict(D=4)])])
    costs = Costs(path, geometry)
    assert costs.evaluate('TP2', 2, [3, 3])['rank_us'] == [11, 11]
    assert not costs.evaluate('TP2', 2, [3, 3])['calibration_eligible']
    assert costs.evaluate('EP2', 2, [3, 0])['layer_us'] is None
    assert costs.evaluate('TP2', 4, [3, 3])['layer_us'] is None
    make_trace(tmp_path/'trace', geometry)
    with Trace(tmp_path/'trace') as trace:
        result = replay(trace, cost_fit=path, bootstrap_samples=0)
    assert result['cycles']['observed/c1_M2_K1/TP2/contiguous']['cycle_kernel_ms']['mean'] is not None
    assert result['cycles']['observed/c1_M2_K1/EP2/contiguous']['cycle_kernel_ms']['mean'] is None
    assert not result['cyclesim_what_if']


def test_geometry_fit_requires_matched_occupancy(tmp_path):
    geometry = Geometry(8, 2, 2)
    owners = placement(1, 'contiguous', geometry=geometry)
    work = rank_work([(0, 1), (1, 2)], 'TP2', owners, geometry=geometry)
    other = rank_work([(0, 1), (2, 3)], 'TP2', owners, geometry=geometry)
    points = []
    for signature, latency in ((work, 10), (other, 14)):
        points.append(dict(D=signature['D_r'][0], median_us=latency,
            local_slots=signature['routed_slots_r'][0], local_slots_per_row=signature['local_slots_per_row_r'][0],
            expert_multiplicities=signature['expert_multiplicities_r'][0],
            local_route_sha256=signature['local_route_sha256_r'][0], duplicate_rows=0,
            matched_route_validation=True))
    path = tmp_path/'fit.json'
    bound_fit(path, geometry, records=[dict(layout='TP2', M=2, a_us=0,
        b_us_per_distinct_local_expert=0, status='measured', points=points)])
    costs = Costs(path, geometry)
    estimate = costs.evaluate('TP2', 2, work['D_r'], work)
    assert estimate['calibration_eligible']
    assert estimate['rank_us'] == [10, 10]
    changed = rank_work([(0, 1), (0, 2)], 'TP2', owners, geometry=geometry)
    assert not costs.evaluate('TP2', 2, changed['D_r'], changed)['calibration_eligible']


def test_pt_converter_geometry_without_tensor_dependency(tmp_path, monkeypatch):
    class Tensor:
        ndim = 2
        dtype = SimpleNamespace(is_floating_point=False, is_complex=False)

        def __init__(self, rows):
            self.data = rows
            self.shape = (len(rows), len(rows[0]))

        def __getitem__(self, key):
            return Tensor(self.data[key])

        def tolist(self):
            return self.data

    calls = []
    def load(path, **options):
        calls.append(options)
        return dict(ids=Tensor([[0, 256], [511, 256], [0, 1], [255, 256]]))

    monkeypatch.setitem(__import__('sys').modules, 'torch', SimpleNamespace(load=load, bool='boolean'))
    path = tmp_path/'source.pt'; path.write_bytes(b'synthetic')
    result = convert_pt(path, tmp_path/'trace', prompt_rows=2, geometry=Geometry(512, 2, 2))
    assert calls == [dict(map_location='cpu', weights_only=True)]
    assert result['schema'] == 'moe-routes v2'
    with Trace(tmp_path/'trace') as trace:
        assert trace.row(3, 1) == (511, 256)
        assert trace.check()['rows'] == 4


def test_legacy_cost_rejects_other_bound_geometry(tmp_path):
    path = tmp_path/'fit.json'; bound_fit(path, Geometry(8, 2, 2))
    with pytest.raises(ValueError, match='geometry'):
        Costs(path)


def test_converter_geometry_and_cli(tmp_path):
    capture = dict(schema=1, step=0, layers=[3], physical_rows=2, k=1, concurrency=1, accepted={'a':1},
        rows=[dict(request_id='a', position=i, padding=False) for i in range(2)], experts=[[[256, 511], [0, 256]]])
    path = tmp_path/'capture.jsonl'; path.write_text(json.dumps(capture)+'\n')
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_cli(['convert-schema1', str(path), '--out', str(tmp_path/'wrong'),
                 '--source-status', 'assumed', '--preset', 'synthetic-wide'])
    assert not (tmp_path/'wrong').exists()
    geometry = Geometry(512, 2, 8)
    with contextlib.redirect_stdout(io.StringIO()):
        converted = run_cli(['convert-schema1', str(path), '--out', str(tmp_path/'trace'),
                             '--source-status', 'assumed', '--geometry', json.dumps(geometry.to_dict())])
        checked = run_cli(['check', str(tmp_path/'trace'), '--geometry', json.dumps(geometry.to_dict())])
    assert converted['schema'] == 'moe-routes v2'
    assert converted['model'] == 'unspecified-model'
    assert checked['structural_check'] == 'PASS'
    with Trace(tmp_path/'trace') as trace:
        assert trace.row(3, 0) == (256, 511)
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_cli(['check', str(tmp_path/'trace'), '--preset', 'synthetic-small'])
    capture['geometry'] = Geometry(512, 2, 4).to_dict()
    path.write_text(json.dumps(capture)+'\n')
    with pytest.raises(ValueError, match='geometry'):
        convert_schema1(path, tmp_path/'mismatch', geometry=geometry)


def test_presets_are_honest_and_cli_lists_provenance():
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_cli(['presets'])
    assert result['presets'] == PRESETS
    for name, metadata in PRESETS.items():
        assert load_geometry(name).num_experts > 0
        assert metadata['runtime_support'] == 'unknown'
        if name != 'glm53-legacy':
            assert metadata['calibration'] == 'unavailable'
            assert 'Synthetic offline' in metadata['provenance']
