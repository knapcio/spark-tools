# SPDX-License-Identifier: MIT
# Copyright (c) 2026 knapcio
"""Portable adaptation of the static CUDA resource parser; no driver calls."""
from dataclasses import asdict, dataclass
import math
import re
from common.config import finite, integer


@dataclass(frozen=True)
class Resource:
    name: str
    target: str
    registers: int
    stack_bytes: int
    spill_store_bytes: int
    spill_load_bytes: int
    static_shared_bytes: int
    barriers: int


def parse_ptxas(text):
    """Require entry identity, spill report and register report for every entry."""
    if re.search(r"ptxas(?:-blackwell)?\s+(?:fatal|error)\s*:", text, re.I):
        raise ValueError("compiler reported an error")
    headers = list(re.finditer(r"Compiling entry function '([^']+)' for '(sm_\d+[af]?)'", text))
    if not headers:
        raise ValueError("no ptxas entry records")
    records = []
    for index, header in enumerate(headers):
        body = text[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)]
        name, target = header.groups()
        if target not in {"sm_121", "sm_121a"}:
            raise ValueError("unsupported compiler target")
        props = re.search(r"Function properties for\s+(\S+)", body)
        used = re.search(r"Used\s+(\d+)\s+registers([^\n]*)", body, re.I)
        spills = re.search(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads", body)
        if not used or not spills or not props or props[1] != name:
            raise ValueError("incomplete or mismatched ptxas entry")
        smem = re.search(r"(\d+) bytes smem", used[2])
        barriers = re.search(r"(?:used\s+)?(\d+)\s+barriers", used[2], re.I)
        records.append(Resource(name, target, int(used[1]), *map(int, spills.groups()),
                                int(smem[1]) if smem else 0, int(barriers[1]) if barriers else 0))
    if len({(row.name, row.target) for row in records}) != len(records):
        raise ValueError("duplicate entry/target records")
    return records


def parse_cuobjdump(text):
    records = {}
    for name, body in re.findall(r"Function\s+(\S+)\s*:\s*([^\n]*REG:[^\n]*)", text):
        if name in records:
            raise ValueError("duplicate binary resource entry")
        records[name] = {key: int(value) for key, value in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", body)}
    if not records:
        raise ValueError("no binary resource records")
    return records


def occupancy(resource, threads=None, dynamic_shared=None, hardware=None):
    """Rounded CPU approximation; the toolkit calculator models more rules.

    The fallback omits subpartition and barrier allocation rules. It is an
    approximation that can overestimate occupancy, not the CUDA runtime API.
    """
    if threads is None or dynamic_shared is None:
        return {"status": "unknown", "reason": "launch geometry not supplied"}
    integer(threads, "threads", 1, 1024)
    integer(dynamic_shared, "dynamic shared memory")
    h = {"sms": 48, "registers_per_sm": 65536, "warps_per_sm": 48,
         "blocks_per_sm": 24, "shared_per_sm": 102400, "shared_per_block": 101376,
         "shared_reserved": 1024, "register_unit": 256, "shared_unit": 128,
         "warp_group": 4}
    if hardware:
        if set(hardware) - set(h):
            raise ValueError("unknown hardware property")
        h.update(hardware)
    for key, value in h.items():
        integer(value, key, 0 if key == "shared_reserved" else 1)
    warps = math.ceil(threads / 32)
    allocated_warps = math.ceil(warps / h["warp_group"]) * h["warp_group"]
    registers = math.ceil(resource.registers * 32 / h["register_unit"]) * h["register_unit"] * allocated_warps
    user_shared = resource.static_shared_bytes + dynamic_shared
    shared = math.ceil((user_shared + h["shared_reserved"]) / h["shared_unit"]) * h["shared_unit"]
    limits = {"registers": h["registers_per_sm"] // registers if registers else h["blocks_per_sm"],
              "shared": h["shared_per_sm"] // shared if shared else h["blocks_per_sm"],
              "warps": h["warps_per_sm"] // warps, "blocks": h["blocks_per_sm"]}
    blocks = min(limits.values()) if user_shared <= h["shared_per_block"] and resource.registers <= 255 else 0
    return {"status": "estimated", "method": "rounded CPU approximation",
            "active_blocks_per_sm": blocks, "active_blocks_total": blocks * h["sms"],
            "warp_occupancy_pct": 100 * blocks * warps / h["warps_per_sm"],
            "allocated_registers_per_block": registers, "allocated_shared_per_block": shared,
            "limits": limits, "limitation": "subpartition and barrier limits omitted; may overestimate occupancy"}


def check_resources(resources, config, binary=None):
    launches = config.get("kernels", {})
    if not launches or set(launches) != {row.name for row in resources}:
        raise ValueError("launch configuration must cover exactly the parsed entries")
    thresholds = config.get("thresholds", {})
    defaults = {"max_registers": 128, "max_spill_bytes": 0, "max_shared_bytes": 101376,
                "min_occupancy_pct": 0}
    if set(thresholds) - set(defaults):
        raise ValueError("unknown resource threshold")
    defaults.update(thresholds)
    for key, value in defaults.items():
        finite(value, key, 0, 100 if key == "min_occupancy_pct" else math.inf)
    rows = []
    for resource in resources:
        spec = launches[resource.name]
        if resource.target != config.get("target"):
            raise ValueError("compiler target differs from configuration")
        occ = occupancy(resource, spec.get("threads"), spec.get("dynamic_shared_bytes"), config.get("hardware"))
        shared = None if spec.get("dynamic_shared_bytes") is None else resource.static_shared_bytes + spec["dynamic_shared_bytes"]
        checks = {"registers": resource.registers <= defaults["max_registers"],
                  "spills": resource.spill_store_bytes + resource.spill_load_bytes <= defaults["max_spill_bytes"],
                  "shared_memory": shared is not None and shared <= defaults["max_shared_bytes"],
                  "launch_geometry": occ["status"] != "unknown" and occ["active_blocks_per_sm"] > 0,
                  "occupancy": occ.get("warp_occupancy_pct", -1) >= defaults["min_occupancy_pct"]}
        if binary is not None:
            entry = binary.get(resource.name)
            # ELF SHARED may include the toolkit's reserved allocation; ptxas
            # reports user static memory. Keep both, do not subtract implicitly.
            checks["binary_registers"] = entry is not None and entry.get("REG") == resource.registers
        rows.append({**asdict(resource), "total_shared_bytes": shared, "occupancy": occ,
                     "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"})
    return {"schema": 1, "mode": "STATIC_NO_GPU", "target": config["target"],
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "resources": rows}
