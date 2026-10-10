# SPDX-License-Identifier: Apache-2.0
"""Bounded, isolated CPU Gloo fault matrices for rank-invariant control logic."""
import argparse
import contextlib
import datetime
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
import traceback
import tracemalloc
import warnings
from dataclasses import dataclass

VERDICTS = {"PASS", "FALLBACK", "FATAL", "HANG", "DIVERGED"}
_context = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json(value) + "\n")
    tmp.replace(path)


def _load_worker(path, name):
    module_name = "rankharness_worker_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, getattr(module, name)


@dataclass(frozen=True)
class Fault:
    name: str
    stage: str = "none"
    ranks: object = (2,)
    action: str = "flag"
    value: object = None


class Context:
    def __init__(self, rank, world, fault, output, shared_dir, options):
        self.rank, self.world, self.fault = rank, world, fault
        self.output, self.shared_dir = Path(output), Path(shared_dir)
        self.options = options
        self.last_stage = "starting"
        self.history = []
        self.guard_diverged = False

    def stage(self, name):
        self.last_stage = name
        self.history.append(name)
        _write(self.output / (str(self.rank) + ".progress.json"),
               dict(rank=self.rank, last_stage=name, history=self.history,
                    measurement="measured", pid=os.getpid()))

    def fault_point(self, stage):
        self.stage(stage)
        hit = self.fault["stage"] == stage and (
            self.fault["ranks"] == "all" or self.rank in self.fault["ranks"])
        if hit and self.fault["action"] == "raise":
            raise RuntimeError("injected fault at " + stage)
        if hit and self.fault["action"] == "hang":
            while True:
                time.sleep(1)
        return hit

    def rank_local(self, value, key="input"):
        """Expose a local input; injection does NOT silently synchronize it."""
        if self.fault_point("rank_local:" + key):
            return self.fault["value"]
        return value


def fault_point(stage):
    return _context.fault_point(stage)


def rank_local(value, key="input"):
    return _context.rank_local(value, key)


def _stage(name):
    if _context is not None:
        _context.stage(name)


def _digest_packet(value):
    # Serialization failures must still enter the same fixed-size collective.
    try:
        return hashlib.sha256(_json(value).encode()).digest(), True
    except Exception:
        return bytes(32), False


def _digest_agreement(value, group=None):
    import torch
    import torch.distributed as dist
    digest, valid = _digest_packet(value)
    packet = torch.tensor(list(digest) + [int(valid)], dtype=torch.uint8)
    peers = [torch.empty_like(packet) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, packet, group=group)
    return all(bool(p[-1]) and torch.equal(p, peers[0]) for p in peers)


def digest_guard(decision, group=None):
    """33-byte fixed-size digest all-gather; records disagreement on every rank."""
    _stage("digest_guard")
    agreed = _digest_agreement(decision, group)
    if not agreed and _context is not None:
        _context.guard_diverged = True
    return agreed


def all_rank_vote(valid, payload, group=None):
    """Completed MIN validity vote AND digest agreement; return a shared bool."""
    import torch
    import torch.distributed as dist
    _stage("all_rank_vote")
    flag = torch.tensor(int(bool(valid)), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    # Never branch on local validity before the payload collective.
    matches = _digest_agreement(payload, group)
    return bool(flag.item()) and matches


def gather_results(result_record, group=None):
    import torch.distributed as dist
    _stage("gather_results")
    rows = [None] * dist.get_world_size(group)
    dist.all_gather_object(rows, result_record, group=group)
    return rows


def cpu_tensor_bytes():
    """Count unique live, GC-visible CPU storages (views counted once)."""
    import torch
    storages = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                storage = None
                if isinstance(obj, torch.Tensor):
                    if obj.device.type == "cpu":
                        storage = obj.untyped_storage()
                elif isinstance(obj, torch.UntypedStorage) and obj.device.type == "cpu":
                    storage = obj
                if storage is not None:
                    storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
            except (RuntimeError, NotImplementedError, ReferenceError):
                continue
    return sum(storages.values())


def _slope(values):
    n = len(values)
    center = (n - 1) / 2
    mean = sum(values) / n
    return sum((i - center) * (v - mean) for i, v in enumerate(values)) / sum(
        (i - center) ** 2 for i in range(n))


def leak_probe(toggle_fn, toggles=8, warmup=2, python_limit=65536, tensor_limit=4096):
    """Measure retention per toggle; no GPU memory/allocator coverage."""
    if toggles < 2 or warmup < 0 or min(python_limit, tensor_limit) < 0:
        raise ValueError("need >=2 toggles, nonnegative warmup and thresholds")
    owned = not tracemalloc.is_tracing()
    if owned:
        tracemalloc.start()
    try:
        for i in range(warmup):
            toggle_fn(i)
        samples = []
        for i in range(toggles + 1):
            if i:
                toggle_fn(warmup + i - 1)
            gc.collect()
            samples.append(dict(python_bytes=tracemalloc.get_traced_memory()[0],
                                cpu_tensor_bytes=cpu_tensor_bytes()))
        slopes = {k: _slope([s[k] for s in samples]) for k in samples[0]}
        detected = any(slopes[k] > limit and samples[-1][k] - samples[0][k] > limit * toggles
                       for k, limit in (("python_bytes", python_limit), ("cpu_tensor_bytes", tensor_limit)))
        return dict(measurement="measured", scope="Python and GC-visible CPU tensor/storage retention only; no GPU memory",
                    toggles=toggles, warmup=warmup, samples=samples,
                    bytes_per_toggle=slopes, limits=dict(python_bytes=python_limit, cpu_tensor_bytes=tensor_limit),
                    leak_detected=detected)
    finally:
        if owned:
            tracemalloc.stop()


def _rank_main(rank, world, spec, fault, output, shared, rendezvous, options):
    # Descendants inherit this session/group; the parent owns whole-group cleanup.
    os.setsid()
    with (Path(output) / (str(rank) + ".log")).open("w", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            _rank_body(rank, world, spec, fault, output, shared, rendezvous, options)


def _rank_body(rank, world, spec, fault, output, shared, rendezvous, options):
    global _context
    sys.modules["lab.rankharness.api"] = sys.modules[__name__]
    # Also effective before importing torch in a spawned child.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    _context = Context(rank, world, fault, output, shared, options)
    row = dict(rank=rank, fault=fault["name"], measurement="measured", finished=False, pid=os.getpid())
    try:
        _context.stage("import")
        import torch
        import torch.distributed as dist
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _context.stage("init_process_group")
        # Parent owns the timeout; Gloo must not turn a skipped collective into an early error.
        dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                                world_size=world, timeout=datetime.timedelta(days=1))
        _context.stage("ready")
        _write(Path(output) / (str(rank) + ".ready.json"), dict(rank=rank))
        _, worker = _load_worker(*spec)
        result = worker(_context)
        if not isinstance(result, dict) or result.get("verdict") not in {"PASS", "FALLBACK", "FATAL", "DIVERGED"}:
            raise ValueError("worker must return a dict with a supported verdict")
        if "decision" not in result or "state" not in result:
            raise ValueError("worker must report decision and final state")
        _json(result)
        row.update(result)
        row.update(rank=rank, fault=fault["name"], measurement="measured", pid=os.getpid())
        if _context.guard_diverged:
            row["verdict"] = "DIVERGED"
        if result.get("leak_probe", {}).get("leak_detected"):
            row["verdict"] = "FATAL"
        row["finished"] = True
    except BaseException as exc:
        row.update(verdict="FATAL", finished=True, decision=None, state=None,
                   error=dict(type=type(exc).__name__, text=str(exc), traceback=traceback.format_exc()))
    finally:
        row.update(last_stage=_context.last_stage, history=_context.history)
        _write(Path(output) / (str(rank) + ".json"), row)
        # Keep transports alive after recording completion. Otherwise an early
        # return can close a peer socket and disguise a skipped collective as
        # a transport error. This is a file poll, not an extra Gloo collective.
        if "dist" in locals() and dist.is_initialized():
            while not all((Path(output) / (str(peer) + ".json")).exists() for peer in range(world)):
                time.sleep(.02)
            dist.destroy_process_group()


def _classify(rows, timed_out):
    if timed_out:
        return "HANG"
    if any(r["verdict"] == "FATAL" for r in rows):
        return "FATAL"
    if any(r["verdict"] == "DIVERGED" for r in rows):
        return "DIVERGED"
    if len({_json([r["verdict"], r["decision"], r["state"]]) for r in rows}) != 1:
        return "DIVERGED"
    return rows[0]["verdict"]


def _run_case(spec, fault, world, timeout_s, startup_timeout_s, output, options):
    output.mkdir(parents=True)
    started = time.monotonic()
    processes = []
    timed_out = False
    premature_exits = {}
    with tempfile.TemporaryDirectory(prefix="rankharness-") as tmp:
        ctx = mp.get_context("spawn")
        try:
            for rank in range(world):
                p = ctx.Process(target=_rank_main, args=(rank, world, spec, fault, str(output), tmp,
                                                       str(Path(tmp) / "gloo"), options))
                p.start()
                processes.append(p)
            deadline = started + startup_timeout_s
            ready = False
            while any(p.is_alive() for p in processes):
                premature_exits = {str(rank): p.exitcode for rank, p in enumerate(processes)
                                   if p.exitcode is not None and
                                   not (output / (str(rank) + ".json")).exists()}
                if premature_exits:
                    break
                if not ready and all((output / (str(r) + ".ready.json")).exists() for r in range(world)):
                    ready = True
                    deadline = time.monotonic() + timeout_s
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(.02)
            # Also cover every rank exiting between polls, including exit(0).
            premature_exits.update({str(rank): p.exitcode for rank, p in enumerate(processes)
                                    if p.exitcode is not None and
                                    not (output / (str(rank) + ".json")).exists()})
        finally:
            # Signal every rank group, even after its leader has already exited.
            for p in processes:
                _signal_rank_group(p, signal.SIGTERM)
                if p.is_alive():
                    # Startup may have been interrupted before setsid().
                    p.terminate()
            for p in processes:
                p.join(1)
            for p in processes:
                _signal_rank_group(p, signal.SIGKILL)
                if p.is_alive():
                    p.kill()
                p.join()
        rows = []
        for rank, p in enumerate(processes):
            path = output / (str(rank) + ".json")
            if path.exists():
                row = json.loads(path.read_text())
            else:
                progress = output / (str(rank) + ".progress.json")
                row = json.loads(progress.read_text()) if progress.exists() else dict(rank=rank, last_stage="spawn", history=[])
                row.update(fault=fault["name"], verdict="HANG" if timed_out else "FATAL",
                           finished=False, decision=None, state=None, measurement="measured", pid=p.pid)
            row["process_exitcode"] = p.exitcode
            row["process_reaped"] = not p.is_alive()
            if not timed_out and p.exitcode != 0:
                row["verdict"] = "FATAL"
            _write(path, row)
            rows.append(row)
    return dict(fault=fault["name"], verdict="FATAL" if premature_exits else _classify(rows, timed_out), results=rows,
                premature_exits=premature_exits,
                finished_ranks=[r["rank"] for r in rows if r["finished"]],
                last_stages={str(r["rank"]): r["last_stage"] for r in rows},
                elapsed_s=time.monotonic() - started, timeout_phase=None if not timed_out else (
                    "worker" if ready else "startup"), measurement="measured")


def _signal_rank_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can report EPERM for an empty group after its leader is reaped.
        # An interrupted startup may also precede the child's setsid().
        try:
            isolated = os.getpgid(process.pid) == process.pid
        except ProcessLookupError:
            isolated = False
        if isolated and process.is_alive():
            raise


class MatrixResult:
    def __init__(self, cases, exit_code):
        self.cases, self.exit_code = cases, exit_code

    def __int__(self):
        return self.exit_code


def run_matrix(worker_fn, faults, world=4, timeout_s=5, expected=None, *, out=None,
               repeat=2, startup_timeout_s=25, options=None):
    """Spawn a fresh group per fault/repeat; return MatrixResult with exit_code."""
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ValueError("world must be a positive integer")
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1 or not all(math.isfinite(x) and x > 0 for x in (timeout_s, startup_timeout_s)):
        raise ValueError("repeat and finite timeouts must be positive")
    module = sys.modules[worker_fn.__module__]
    if "<locals>" in worker_fn.__qualname__ or worker_fn.__name__ == "<lambda>":
        raise ValueError("worker must be a top-level function in an importable Python file")
    spec = (str(Path(module.__file__).resolve()), worker_fn.__name__)
    normalized = []
    for fault in faults:
        f = vars(fault).copy() if isinstance(fault, Fault) else dict(fault)
        if "name" not in f or set(f) - {"name", "stage", "ranks", "action", "value"}:
            raise ValueError("fault requires name and only supported fields")
        f = dict(vars(Fault(f["name"])), **f)
        if not isinstance(f["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", f["name"]):
            raise ValueError("case names must be safe path components")
        ranks = f["ranks"]
        if ranks != "all":
            ranks = [ranks] if isinstance(ranks, int) else list(ranks)
            if any(isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < world for r in ranks):
                raise ValueError("fault rank outside world")
            f["ranks"] = ranks
        if f["action"] not in {"flag", "raise", "hang", "local"} or not isinstance(f["stage"], str):
            raise ValueError("invalid fault action or stage")
        _json(f)
        normalized.append(f)
    if not normalized or len({f["name"] for f in normalized}) != len(normalized):
        raise ValueError("need nonempty, uniquely named faults")
    expected = {f["name"]: "PASS" for f in normalized} if expected is None else dict(expected)
    if any(v not in VERDICTS for v in expected.values()):
        raise ValueError("unsupported expected verdict")
    if any(f["name"] not in expected for f in normalized):
        raise ValueError("expected table must cover every selected case")
    output = Path(out) if out else Path(tempfile.mkdtemp(prefix="rankharness-results-"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verdicts.jsonl").exists() or (output / "init-gloo").exists():
        raise ValueError("output already contains harness results; choose a fresh directory")
    _write(output / "manifest.json", dict(schema=1, world=world, repeat=repeat, timeout_s=timeout_s,
           startup_timeout_s=startup_timeout_s, faults=normalized, expected=expected,
           options=options or {}, measurement="measured", expected_basis="assumed test contract",
           timeout_basis="configured limit, not real engine timing", python=sys.version))
    cases = []
    with (output / "verdicts.jsonl").open("w") as log:
        for f in normalized:
            runs = []
            for index in range(repeat):
                run = _run_case(spec, f, world, timeout_s, startup_timeout_s,
                                output / "init-gloo" / f["name"] / ("repeat-" + str(index + 1)), options or {})
                run["repeat"] = index + 1
                runs.append(run)
                log.write(_json(run) + "\n")
                log.flush()
            for row in runs[0]["results"]:
                _write(output / "init-gloo" / f["name"] / (str(row["rank"]) + ".json"), row)
            verdicts = [r["verdict"] for r in runs]
            case = dict(fault=f["name"], verdicts=verdicts,
                        measurement="measured", expected_basis="assumed test contract",
                        nondeterministic=len(set(verdicts)) != 1,
                        determinism="measured" if repeat >= 2 else "unknown",
                        expected=expected.get(f["name"], "unknown"),
                        mismatch=any(v != expected[f["name"]] for v in verdicts) if f["name"] in expected else False)
            cases.append(case)
            print(f["name"], verdicts, flush=True)
    exit_code = int(any(c["mismatch"] or c["nondeterministic"] for c in cases))
    summary = [c["fault"] + " " + str(sorted(set(c["verdicts"]))) +
               (" NONDETERMINISTIC" if c["nondeterministic"] else "") +
               (" EXPECTED_MISMATCH" if c["mismatch"] else "") for c in cases]
    (output / "summary.txt").write_text("\n".join(summary) + "\n")
    _write(output / "matrix.json", dict(schema=1, cases=cases, exit_code=exit_code, measurement="measured"))
    result = MatrixResult(cases, exit_code)
    result.out = output
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("example", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--cases")
    run.add_argument("--repeat", type=int, default=2)
    run.add_argument("--world", type=int, default=4)
    run.add_argument("--timeout", type=float, default=5)
    run.add_argument("--startup-timeout", type=float, default=25)
    run.add_argument("--overlay-dir", type=Path)
    run.add_argument("--fixtures-dir", type=Path)
    run.add_argument("--reference-summary", type=Path)
    args = parser.parse_args(argv)
    # Ensure examples importing rankharness share this module's context under script execution.
    sys.modules["lab.rankharness.api"] = sys.modules[__name__]
    try:
        module, worker = _load_worker(args.example.resolve(), "worker")
        options = {k: str(getattr(args, k).resolve()) for k in ("overlay_dir", "fixtures_dir", "reference_summary")
                   if getattr(args, k) is not None}
        if hasattr(module, "prepare"):
            reason = module.prepare(options)
            if reason:
                args.out.mkdir(parents=True, exist_ok=True)
                if any(args.out.iterdir()):
                    raise ValueError("skip output must be empty")
                _write(args.out / "skip.json", dict(status="SKIP", reason=reason, measurement="unknown"))
                (args.out / "summary.txt").write_text("SKIP " + reason + "\n")
                print("SKIP " + reason)
                return 0
        faults = module.FAULTS
        if args.cases:
            selected = args.cases.split(",")
            faults = [f for f in faults if (f.name if isinstance(f, Fault) else f["name"]) in selected]
            found = {f.name if isinstance(f, Fault) else f["name"] for f in faults}
            if found != set(selected):
                raise ValueError("unknown selected cases: " + str(set(selected) - found))
        return run_matrix(worker, faults, args.world, args.timeout, module.EXPECTED, out=args.out,
                          repeat=args.repeat, startup_timeout_s=args.startup_timeout, options=options).exit_code
    except (ValueError, OSError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
