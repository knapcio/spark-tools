# SPDX-License-Identifier: Apache-2.0
"""Immutable offline MoE geometry; presets do not assert runtime support."""
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from pathlib import Path


@dataclass(frozen=True)
class MoEGeometry:
    num_experts: int
    top_k: int
    world_size: int
    # Full unsharded packed weight + scale footprint, not measured DRAM traffic.
    expert_bytes: int = None

    def __post_init__(self):
        for name in ('num_experts', 'top_k', 'world_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.num_experts > 65536 or self.top_k > self.num_experts:
            raise ValueError('num_experts <=65536 and top_k <=num_experts required')
        if self.expert_bytes is not None and (type(self.expert_bytes) is not int or self.expert_bytes < 1):
            raise ValueError('expert_bytes must be a positive integer or null')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {'num_experts', 'top_k', 'world_size', 'expert_bytes'}:
            raise ValueError('invalid geometry fields')
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError('geometry requires num_experts, top_k and world_size') from error

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


Geometry = MoEGeometry
LEGACY_GEOMETRY = MoEGeometry(256, 8, 4, 4866048 * 4)
PRESETS = {
    'glm53-legacy': dict(geometry=LEGACY_GEOMETRY.to_dict(),
        provenance='Legacy GLM-5.3 256/top8/four-rank geometry; historical TP4 INT4+BF16 g128 weight footprint and seven timing cells only.',
        calibration='legacy TP4 prior only; EP scaling remains estimated', runtime_support='unknown'),
    'synthetic-small': dict(geometry=MoEGeometry(8, 2, 2).to_dict(),
        provenance='Synthetic offline fixture; no model or hardware measurements.',
        calibration='unavailable', runtime_support='unknown'),
    'synthetic-wide': dict(geometry=MoEGeometry(512, 4, 8).to_dict(),
        provenance='Synthetic offline uint16 fixture; no model or hardware measurements.',
        calibration='unavailable', runtime_support='unknown'),
}


def load_geometry(value):
    """Accept an object, mapping, preset, JSON object, or geometry JSON file."""
    if isinstance(value, MoEGeometry):
        return value
    if isinstance(value, dict):
        return MoEGeometry.from_dict(value)
    if isinstance(value, str):
        if value in PRESETS:
            return MoEGeometry.from_dict(PRESETS[value]['geometry'])
        data = json.loads(value if value.lstrip().startswith('{') else Path(value).read_text())
        return MoEGeometry.from_dict(data)
    raise ValueError('explicit geometry required')


@dataclass(frozen=True)
class Layout:
    name: str
    ep: int
    tp: int

    def validate(self, geometry):
        if not isinstance(self.name, str) or not self.name or '/' in self.name:
            raise ValueError('layout needs a nonempty name without slash')
        if any(type(n) is not int or n < 1 for n in (self.ep, self.tp)) or self.ep * self.tp != geometry.world_size:
            raise ValueError('layout ep*tp must equal geometry world_size')
        if self.ep > geometry.num_experts:
            raise ValueError('layout EP groups exceed expert count')
        if geometry.expert_bytes is not None and geometry.expert_bytes % self.tp:
            raise ValueError('expert_bytes must divide evenly across TP shards')
        return self

    def to_dict(self):
        return asdict(self)


def resolve_layouts(geometry, layouts=None):
    geometry = load_geometry(geometry)
    if layouts is None:
        layouts = [Layout('TP%d' % geometry.world_size if ep == 1 else
                          'EP%d' % ep if ep == geometry.world_size else 'EP%dxTP%d' % (ep, geometry.world_size // ep),
                          ep, geometry.world_size // ep)
                   for ep in range(1, min(geometry.world_size, geometry.num_experts) + 1)
                   if geometry.world_size % ep == 0 and
                   (geometry.expert_bytes is None or geometry.expert_bytes % (geometry.world_size // ep) == 0)]
    if not isinstance(layouts, (list, tuple)) or not layouts:
        raise ValueError('nonempty explicit layouts required')
    result = []
    for value in layouts:
        if isinstance(value, Layout):
            layout = value
        elif isinstance(value, str):
            match = re.fullmatch(r'TP([1-9][0-9]*)|EP([1-9][0-9]*)(?:xTP([1-9][0-9]*))?', value)
            if not match:
                raise ValueError('layout must be TPn, EPn, EPnxTPn or an explicit layout object')
            layout = Layout(value, 1 if match[1] else int(match[2]), int(match[1] or match[3] or 1))
        elif isinstance(value, dict) and set(value) == {'name', 'ep', 'tp'}:
            layout = Layout(**value)
        else:
            raise ValueError('invalid layout')
        result.append(layout.validate(geometry))
    if len({layout.name for layout in result}) != len(result):
        raise ValueError('duplicate layout name')
    return tuple(result)
