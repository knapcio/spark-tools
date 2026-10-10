# SPDX-License-Identifier: Apache-2.0
"""Negative controls: local MemAvailable shapes and skipped collectives."""
from lab.rankharness.api import Fault, digest_guard

FAULTS = [Fault("guarded", "rank_local:memory", value=7890),
          Fault("skipped", "rank_local:memory", value=7890)]
EXPECTED = dict(guarded="DIVERGED", skipped="HANG")


def worker(ctx):
    import torch.distributed as dist
    memory = ctx.rank_local(8510, "memory")
    # Represents the unsafe MemAvailable-derived batch shape.
    decision = memory // 1000
    if ctx.fault["name"] == "guarded":
        agreed = digest_guard(dict(chunk=decision))
        verdict = "PASS" if agreed else "DIVERGED"
    else:
        ctx.stage("conditional_collective")
        if decision >= 8:
            dist.barrier()
        verdict = "PASS"
    return dict(verdict=verdict, decision=decision, state=dict(chunk=decision))
