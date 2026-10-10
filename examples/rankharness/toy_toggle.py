# SPDX-License-Identifier: Apache-2.0
"""Sequence-numbered sidecar lever, collective fallback and retention faults."""
import json
from pathlib import Path

from lab.rankharness.api import Fault, all_rank_vote, digest_guard, gather_results, leak_probe

FAULTS = [
    Fault("none", ranks=[]),
    Fault("invalid_one", "validate"),
    Fault("invalid_subset", "validate", [1, 3]),
    Fault("invalid_all", "validate", "all"),
    Fault("payload", "payload"),
    Fault("stale_sequence", "sidecar", "all"),
    Fault("malformed_file", "sidecar", "all"),
    Fault("missing_file", "sidecar", "all"),
    Fault("read_exception", "read", action="raise"),
    Fault("memory", "rank_local:memory", value=4),
    Fault("timer", "rank_local:timer", value=99),
    Fault("local_file", "rank_local:control", value={"sequence": 99, "on": False}),
    Fault("skip_collective", "before_vote"),
    Fault("final_state", "apply"),
    Fault("fatal", "fatal", "all", action="raise"),
    Fault("python_leak", "retention"),
    Fault("cpu_leak", "retention"),
]
EXPECTED = {f.name: "FALLBACK" for f in FAULTS}
EXPECTED.update(none="PASS", skip_collective="HANG", final_state="DIVERGED",
                fatal="FATAL", python_leak="FATAL", cpu_leak="FATAL")


def worker(ctx):
    import torch
    import torch.distributed as dist
    state = dict(on=False, ready=True, sequence=0)
    verdict = "PASS"
    path = Path(ctx.shared_dir) / "control.json"
    result_records = []
    ctx.fault_point("fatal")
    for sequence in range(1, 5):
        ctx.fault_point("sidecar")
        if ctx.rank == 0:
            if ctx.fault["name"] == "malformed_file":
                path.write_text("{")
            elif ctx.fault["name"] != "missing_file":
                path.write_text(json.dumps(dict(sequence=0 if ctx.fault["name"] == "stale_sequence" else sequence,
                                                on=bool(sequence % 2))))
        ctx.stage("sidecar_barrier")
        dist.barrier()
        candidate = None
        valid = True
        try:
            ctx.fault_point("read")
            candidate = ctx.rank_local(json.loads(path.read_text()), "control")
            valid = (set(candidate) == {"sequence", "on"} and type(candidate["sequence"]) is int
                     and type(candidate["on"]) is bool and candidate["sequence"] > state["sequence"])
        except (OSError, ValueError, TypeError, RuntimeError):
            valid = False
        # A local memory input becomes a common MIN before affecting the lever.
        memory = torch.tensor(ctx.rank_local(8, "memory"), dtype=torch.int64)
        ctx.stage("memory_min")
        dist.all_reduce(memory, op=dist.ReduceOp.MIN)
        timer = ctx.rank_local(0, "timer")
        valid = valid and memory.item() >= 6
        valid = valid and not ctx.fault_point("validate")
        payload = dict(control=candidate, memory=memory.item(), timer=timer)
        if ctx.fault_point("payload"):
            payload["memory"] = 100
        if ctx.fault_point("before_vote"):
            # Return without closing the collective; the parent must report HANG.
            return dict(verdict="PASS", decision="skipped", state=state)
        approved = all_rank_vote(valid, payload)
        result_records = gather_results(dict(rank=ctx.rank, approved=approved, valid=valid))
        if not approved:
            state.update(on=False, ready=False)
            verdict = "FALLBACK"
            break
        state.update(on=candidate["on"], sequence=candidate["sequence"])
        if ctx.fault_point("apply"):
            state["on"] = not state["on"]
        if not digest_guard(state):
            verdict = "DIVERGED"
            break
    retained = []
    def toggle(i):
        # Balanced arm/disarm; all retained CPU data should disappear after disarm.
        allocation = torch.zeros(32768)
        if ctx.fault_point("retention"):
            if ctx.fault["name"] == "python_leak":
                retained.append(bytearray(131072))
            elif ctx.fault["name"] == "cpu_leak":
                retained.append(allocation)
    probe = leak_probe(toggle, toggles=4, warmup=1)
    return dict(verdict=verdict, decision=state["on"], state=state,
                result_records=result_records, leak_probe=probe)
