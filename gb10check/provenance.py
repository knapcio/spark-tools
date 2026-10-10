# SPDX-License-Identifier: Apache-2.0
"""Bounded build provenance, not proof of reproducible output."""
import hashlib
import platform
from pathlib import Path
import shlex
from common.config import digest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_hashes(text):
    """Hash the compiler-emitted include closure without path normalization."""
    target, separator, dependencies = text.replace("\\\n", " ").partition(":")
    if not separator or not target.strip():
        raise ValueError("compiler dependency output lacks a target")
    paths = shlex.split(dependencies)
    if not paths or len(paths) > 4096:
        raise ValueError("compiler dependency closure must contain 1..4096 files")
    result = {}
    for name in paths:
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("compiler dependency must resolve to an absolute file")
        result[name] = file_hash(path)
    return result


def declared_inputs(config, root, resolve):
    """Hash explicit files only; do not claim transitive include coverage."""
    def inputs(value):
        if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
            raise ValueError("declared inputs must be a list of filenames")
        return value
    names = list(inputs(config.get("inputs", [])))
    for entry in config.get("entries", []):
        names.extend(inputs(entry.get("inputs", [])))
        names.extend(entry[key] for key in ("source", "baseline_ptx", "baseline_sass") if entry.get(key))
    if any(not isinstance(name, str) for name in names):
        raise ValueError("declared inputs must be filenames")
    return {name: file_hash(resolve(root, name)) for name in sorted(set(names))}


def build_provenance(config, root, resolve, image=None, requested_platform=None):
    return {
        "schema": 1,
        "kind": "build provenance",
        "limitation": "Not reproducible-build proof; only explicit input files are hashed. "
                      "Transitive includes, compiler caches and image contents are not independently audited. "
                      "Commands record adapter-issued processes; compiler-internal subprocesses are not traced.",
        "config_digest": digest(config),
        "declared_inputs_sha256": declared_inputs(config, root, resolve),
        "target": config["target"],
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "requested": requested_platform or "unknown"},
        "image": image or {"status": "unknown"},
        "commands": [],
        "toolkit_versions": {},
        "occupancy_header": {"path": "/usr/local/cuda/include/cuda_occupancy.h", "sha256": "unknown"},
        "occupancy_source_sha256": file_hash(Path(__file__).with_name("occupancy.cpp")),
    }
