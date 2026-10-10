# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from lab.rankharness import api as rh


def worker(ctx):
    name = ctx.fault["name"]
    ctx.fault_point("fatal")
    hit = ctx.fault_point("vote")
    if name == "skip" and hit:
        return dict(verdict="PASS", decision="skip", state={})
    if name == "hang" and hit:
        ctx.fault_point("blocked")
    payload = {"on": hit} if name == "payload" else {"on": True}
    if name == "unserializable" and hit:
        payload = {"nan": float("nan")}
    approved = rh.all_rank_vote(not ((name.startswith("invalid") or name == "unsafe") and hit), payload)
    peers = rh.gather_results(dict(rank=ctx.rank, approved=approved))
    assert len(peers) == ctx.world and len({r["approved"] for r in peers}) == 1
    state = dict(on=approved, ready=approved)
    if name == "unsafe" and hit:
        state["on"] = True
    if name == "local":
        state["memory"] = ctx.rank_local(8, "memory")
    verdict = "PASS" if approved else "FALLBACK"
    if name == "guard":
        assert not rh.digest_guard(dict(chunk=ctx.rank))
    if name == "alternating" and ctx.output.name == "repeat-2":
        verdict = "FALLBACK"
    return dict(verdict=verdict, decision=approved, state=state, rank=-1, pid=1, measurement="assumed")


class RankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        faults = [rh.Fault("good", ranks=[]), rh.Fault("invalid_one", "vote"),
                  rh.Fault("invalid_subset", "vote", [0, 3]), rh.Fault("invalid_all", "vote", "all"),
                  rh.Fault("payload", "vote"), rh.Fault("unserializable", "vote"),
                  rh.Fault("unsafe", "vote"), rh.Fault("guard"), rh.Fault("skip", "vote"),
                  rh.Fault("fatal", "fatal", "all", "raise"),
                  rh.Fault("local", "rank_local:memory", value=4), rh.Fault("alternating")]
        cls.expected = dict(good="PASS", invalid_one="FALLBACK", invalid_subset="FALLBACK",
                            invalid_all="FALLBACK", payload="FALLBACK", unserializable="FALLBACK",
                            unsafe="DIVERGED", guard="DIVERGED", skip="HANG", fatal="FATAL",
                            local="DIVERGED", alternating="PASS")
        cls.started = time.monotonic()
        cls.result = rh.run_matrix(worker, faults, timeout_s=.5, expected=cls.expected,
                                   out=Path(cls.temp.name) / "matrix")
        cls.elapsed = time.monotonic() - cls.started
        cls.cases = {c["fault"]: c for c in cls.result.cases}
        cls.runs = [json.loads(line) for line in (cls.result.out / "verdicts.jsonl").read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_matrix_classification(self):
        for name, expected in self.expected.items():
            if name != "alternating":
                self.assertEqual(self.cases[name]["verdicts"], [expected, expected])

    def test_refused_vote_safe_state(self):
        for run in self.runs:
            if run["fault"] in {"invalid_one", "invalid_subset", "invalid_all", "payload", "unserializable"}:
                self.assertEqual([r["state"] for r in run["results"]], [dict(on=False, ready=False)] * 4)

    def test_hang_deadline_stages_finished_and_no_orphans(self):
        for run in self.runs:
            if run["fault"] == "skip":
                self.assertEqual(run["verdict"], "HANG")
                self.assertEqual(run["timeout_phase"], "worker")
                self.assertEqual(run["finished_ranks"], [2])
                self.assertEqual(run["last_stages"]["0"], "all_rank_vote")
                self.assertLess(run["elapsed_s"], 10)
            for row in run["results"]:
                self.assertTrue(row["process_reaped"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(row["pid"], 0)
        self.assertFalse(multiprocessing.active_children())

    def test_determinism_and_nonzero_exit(self):
        self.assertTrue(self.cases["alternating"]["nondeterministic"])
        self.assertTrue(self.cases["alternating"]["mismatch"])
        self.assertEqual(self.result.exit_code, 1)

    def test_record_layout(self):
        self.assertEqual(len(self.runs), 24)
        self.assertEqual(len(list((self.result.out / "init-gloo" / "good").glob("[0-3].json"))), 4)
        self.assertEqual(json.loads((self.result.out / "matrix.json").read_text())["exit_code"], 1)
        for run in self.runs:
            self.assertEqual([row["rank"] for row in run["results"]], list(range(4)))
            self.assertTrue(all(row["measurement"] == "measured" for row in run["results"]))

    def test_suite_spawn_budget(self):
        self.assertLess(self.elapsed, 180)


class PureTests(unittest.TestCase):
    def test_divergent_final_state_and_verdict(self):
        rows = [dict(verdict="FALLBACK", decision=False, state=dict(on=False)) for _ in range(4)]
        rows[2]["state"]["on"] = True
        self.assertEqual(rh._classify(rows, False), "DIVERGED")
        self.assertEqual(rh._classify(rows, True), "HANG")
        rows[2]["state"]["on"] = False
        rows[2]["verdict"] = "PASS"
        self.assertEqual(rh._classify(rows, False), "DIVERGED")

    def test_validation_before_spawning(self):
        for faults in ([rh.Fault("../bad")], [rh.Fault("rank", ranks=[4])],
                       [rh.Fault("duplicate"), rh.Fault("duplicate")],
                       [{"name": "unknown_field", "unsupported": True}]):
            with self.assertRaises(ValueError):
                rh.run_matrix(worker, faults)
        with self.assertRaises(ValueError):
            rh.run_matrix(worker, [rh.Fault("ok")], timeout_s=float("nan"))
        with self.assertRaises(ValueError):
            rh.run_matrix(worker, [rh.Fault("ok")], expected={})

    def test_retention_probe_python_cpu_and_clean(self):
        import torch
        retained = []
        py = rh.leak_probe(lambda i: retained.append(bytearray(131072)), toggles=4)
        self.assertTrue(py["leak_detected"])
        self.assertGreater(py["bytes_per_toggle"]["python_bytes"], 65536)
        retained.clear()
        cpu = rh.leak_probe(lambda i: retained.append(torch.zeros(32768)), toggles=4)
        self.assertTrue(cpu["leak_detected"])
        self.assertGreater(cpu["bytes_per_toggle"]["cpu_tensor_bytes"], 120000)
        retained.clear()
        clean = rh.leak_probe(lambda i: torch.zeros(32768), toggles=4)
        self.assertFalse(clean["leak_detected"])

    def test_cpu_storage_views_counted_once(self):
        import torch
        baseline = rh.cpu_tensor_bytes()
        tensor = torch.zeros(1000)
        view = tensor[:500]
        self.assertEqual(rh.cpu_tensor_bytes() - baseline, 4000)
        self.assertEqual(view.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())



if __name__ == "__main__":
    unittest.main()
