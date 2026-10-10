# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""CPU reference for the diagnostic post-remap capture contract, NOT a GPU hook.

The coordinator must implement the predicated copy as a graph-safe device op.
This reference tests selection, epochs and complete-layer export on CPU only.
"""
from moereplay.routes import valid_ids

ROUTE_STAGE = 'post-kstop-remap'


def remap_reference(ids, sources):
    """Snapshot semantics of glm_deadrow::remap_; includes dead AND padded rows."""
    snapshot = [valid_ids(list(row)) for row in ids]
    if len(sources)!=len(snapshot) or any(type(i) is not int or not 0<=i<len(snapshot) for i in sources):
        raise ValueError('exact per-step source row map required')
    return [snapshot[i] for i in sources]


class CapturePolicy:
    def __init__(self, layers=range(3,78), max_rows=16, max_steps=2000):
        self.layers = tuple(layers)
        self.max_rows, self.max_steps = max_rows, max_steps
        self.graphs = set()
        self.records = self.drops = self.overflows = 0
        self.enabled = False
        self.scratch = {}

    def install(self, graph, phase, rows, target=True, profiling=False):
        # Called before capture of a specifically classified decode graph.
        # Profile/prefill/draft graphs get no copy nodes or scratch accesses.
        if not target or profiling or phase!='decode-verify' or not 0<rows<=self.max_rows:
            return False
        self.graphs.add((graph,rows))
        return True

    def begin(self, step, graph, phase, rows, target=True, profiling=False, transition=False):
        self.enabled = False
        self.scratch = {}
        if profiling or transition or not target or phase!='decode-verify' or self.records>=self.max_steps:
            return False
        if not 0<rows<=self.max_rows:
            self.overflows += 1
            self.drops += 1
            return False
        if graph is not None and (graph,rows) not in self.graphs:
            self.drops += 1
            return False
        self.step, self.rows = step, rows
        self.enabled = True
        return True

    def copy_after_remap(self, layer, ids, stage=ROUTE_STAGE):
        # CPU stand-in for the DEVICE enable/epoch-predicated op in a graph.
        if not self.enabled:
            return False
        if stage!=ROUTE_STAGE or layer not in self.layers or layer in self.scratch or len(ids)!=self.rows:
            self.enabled = False
            self.drops += 1
            return False
        try:
            self.scratch[layer] = [valid_ids(list(row)) for row in ids]
        except ValueError:
            self.enabled = False
            self.drops += 1
            return False
        return True

    def finish(self):
        if not self.enabled:
            return None
        self.enabled = False
        if set(self.scratch)!=set(self.layers):
            self.drops += 1
            return None
        self.records += 1
        return dict(step=self.step,route_stage=ROUTE_STAGE,
                    experts=[self.scratch[l] for l in self.layers])
