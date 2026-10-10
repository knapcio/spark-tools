# SPDX-License-Identifier: Apache-2.0
"""CPU exact-byte fixtures, with fixed upstream hashes and negative controls."""
import json
import math
import struct

from . import SCOPE
from .golden import DENSE, TILES
from .manifests import UPSTREAMS
from .reference import (CODEBOOKS, E4M3_POSITIVE, GOLDEN, WIDTHS, codebook_values,
                        decode_tile, dense_fixture, digest, e4m3_round_bits,
                        e4m3_round_bits_arithmetic, exlr8_symbol_states,
                        synthetic_raw, tile_states, value_bytes)


def fixtures(format_name="all"):
    if format_name not in ("all", "exl3", "exlr8"):
        raise ValueError("format must be all, exl3 or exlr8")
    rows = []
    if format_name in ("all", "exl3"):
        for codebook in CODEBOOKS:
            for bits in WIDTHS:
                if not float(bits).is_integer() and codebook != "mul1":
                    continue
                raw = synthetic_raw(bits)
                key = f"{codebook}/{bits:g}"
                actual = digest(value_bytes(decode_tile(raw, bits, codebook)))
                rows.append({"id": key, "format": "EXL3", "bits": bits,
                             "codebook": codebook, "shape": [16, 16], "orientation": "input,output",
                             "dtype": "fp16-le", "domain": "rotated", "raw_hex": raw.hex(),
                             "raw_sha256": digest(raw), "decoded_sha256": actual,
                             "expected_sha256": TILES[key], "matches": actual == TILES[key]})
    if format_name in ("all", "exlr8"):
        for bits in (2, 3, 4):
            for side in ("in", "out"):
                raw, matrix = dense_fixture(bits, side, 256, 256)
                actual = digest(value_bytes(matrix, "fp32"))
                rows.append({"id": f"exlr8/{bits}/{side}", "format": "EXLR8", "bits": bits,
                             "codebook": "mcg-clamp448-e4m3fn-rne", "shape": [256, 256],
                             "orientation": "HF out,in", "entry_side": side,
                             "dtype": "fp32-le", "domain": "rotated; before Hadamard/scales",
                             "raw_pattern": "synthetic_raw-v1; consecutive tile_index in stored entry order",
                             "raw_bytes": len(raw), "raw_sha256": digest(raw),
                             "decoded_sha256": actual, "expected_sha256": DENSE[f"{bits}/{side}"],
                             "matches": actual == DENSE[f"{bits}/{side}"]})
    identity = digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    return {"schema": 1, "scope": SCOPE, "fixture_version": 1, "format": format_name,
            "status": "PASS" if all(row["matches"] for row in rows) else "FAIL",
            "fixtures": rows, "fixture_set_sha256": identity,
            "limits": {"max_matrix_dimension": 256, "max_states": 65536},
            "upstream_revisions": {name: pin["revision"] for name, pin in UPSTREAMS.items()}}


def _rounding_checks():
    checked = 0
    mismatches = 0
    for state_value in codebook_values("mcg"):
        mismatches += e4m3_round_bits(state_value) != e4m3_round_bits_arithmetic(state_value)
        checked += 1
    for lower in range(126):
        midpoint = (E4M3_POSITIVE[lower] + E4M3_POSITIVE[lower + 1]) / 2
        tie = lower if lower % 2 == 0 else lower + 1
        for sign, sign_bit in ((1, 0), (-1, 128)):
            for value, expected in ((math.nextafter(midpoint, 0), lower), (midpoint, tie),
                                    (math.nextafter(midpoint, math.inf), lower + 1)):
                got = e4m3_round_bits(sign * value)
                mismatches += got != (expected | sign_bit)
                mismatches += got != e4m3_round_bits_arithmetic(sign * value)
                checked += 1
    for value, expected in ((0.0, 0), (-0.0, 128), (448, 126), (-448, 254),
                            (449, 126), (-449, 254), (1e300, 126), (-1e300, 254)):
        mismatches += e4m3_round_bits(value) != expected
        checked += 1
    rejected = 0
    for value in (float("nan"), float("inf"), -float("inf")):
        try:
            e4m3_round_bits(value)
        except ValueError:
            rejected += 1
    return {"state_inputs": 65536, "halfway_neighbor_inputs": 756,
            "finite_inputs_checked": checked, "mismatches": mismatches,
            "nonfinite_inputs_rejected": rejected, "signed_zero_preserved": True,
            "nonfinite_policy": "reject fixture input; generated MCG values are finite"}


def cpu_report(format_name="all"):
    result = fixtures(format_name)
    report = {"schema": 1, "scope": SCOPE, "format": format_name, "backend": "stdlib-cpu",
              "comparison": "exact little-endian decoder bytes; zero tolerance",
              "fixture_set_sha256": result["fixture_set_sha256"],
              "fixture_count": len(result["fixtures"]), "fixture_failures": [row["id"] for row in result["fixtures"] if not row["matches"]],
              "codebooks": [], "errors": [], "gpu_status": "UNAVAILABLE",
              "matmul_status": "NOT_RUN", "dense_reconstruction_status": "EXTERNAL_ONLY",
              "claims_excluded": ["model", "quality", "collective", "whole-engine", "performance"]}
    if format_name in ("all", "exl3"):
        for name in CODEBOOKS:
            actual = digest(value_bytes(codebook_values(name)))
            report["codebooks"].append({"name": name, "states": 65536, "dtype": "fp16-le",
                                        "sha256": actual, "expected_sha256": GOLDEN[name], "matches": actual == GOLDEN[name]})
    if format_name in ("all", "exlr8"):
        values = codebook_values("exlr8")
        for name, blob in (("exlr8-fp32", value_bytes(values, "fp32")),
                           ("exlr8-e4m3", bytes(e4m3_round_bits(value) for value in values))):
            actual = digest(blob)
            report["codebooks"].append({"name": name, "states": 65536, "sha256": actual,
                                        "expected_sha256": GOLDEN[name], "matches": actual == GOLDEN[name]})
        rounding = _rounding_checks()
        report["rounding"] = rounding
        if rounding["mismatches"] or rounding["nonfinite_inputs_rejected"] != 3:
            report["errors"].append("E4M3 rounding policy mismatch")
        state_failures = []
        for width in (2, 3, 4):
            for tile in range(8):
                raw = synthetic_raw(width, tile)
                if tile_states(raw, width) != exlr8_symbol_states(raw, width):
                    state_failures.append([width, tile])
        report["trellis_order"] = {"cases": 24, "states_per_case": 256, "failures": state_failures}
        if state_failures:
            report["errors"].append("independent K2/K3/K4 state decoders disagree")
        mismatched = sum(struct.pack("<f", stock) != struct.pack("<f", rounded)
                         for stock, rounded in zip(codebook_values("mcg"), values))
        report["negative_control"] = {"name": "stock-unrounded-EXL3-MCG", "states": 65536,
                                      "different_fp32_states": mismatched,
                                      "expected": "REJECT", "status": "REJECTED" if mismatched else "FAIL"}
        if not mismatched:
            report["errors"].append("stock unrounded EXL3 control unexpectedly accepted")
    if report["fixture_failures"]:
        report["errors"].append("pinned fixture hash mismatch")
    if any(not row["matches"] for row in report["codebooks"]):
        report["errors"].append("pinned exhaustive codebook hash mismatch")
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report
