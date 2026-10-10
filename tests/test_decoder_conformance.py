# SPDX-License-Identifier: Apache-2.0
"""Exact arithmetic and tamper controls; no GPU, network, model or engine."""
import json
import math
from pathlib import Path
import struct
import subprocess
from unittest import mock

import pytest

from common.lab_cli import main as lab_main
from lab.conformance import SCOPE
from lab.conformance.cli import main
from lab.conformance.golden import DENSE, TILES
from lab.conformance.manifests import UPSTREAMS, check_source, runner_manifest
from lab.conformance.reference import (
    CODEBOOKS, E4M3_POSITIVE, GOLDEN, WIDTHS, codebook_values, decode_tile,
    dense_fixture, digest, e4m3_round_bits, e4m3_round_bits_arithmetic,
    e4m3_value, exl3_value, exlr8_symbol_states, synthetic_raw,
    tile_coordinates, tile_states, value_bytes,
)
from lab.conformance.suite import _rounding_checks, cpu_report, fixtures


@pytest.mark.parametrize("codebook", CODEBOOKS)
def test_all_65536_exl3_states_match_pinned_upstream(codebook):
    values = codebook_values(codebook)
    assert len(values) == 65536
    assert all(math.isfinite(value) for value in values)
    assert digest(value_bytes(values)) == GOLDEN[codebook]


def test_all_65536_exlr8_states_match_fp32_and_e4m3_upstream_bits():
    values = codebook_values("exlr8")
    assert digest(value_bytes(values, "fp32")) == GOLDEN["exlr8-fp32"]
    assert digest(bytes(e4m3_round_bits(value) for value in values)) == GOLDEN["exlr8-e4m3"]
    result = _rounding_checks()
    assert result["mismatches"] == 0
    assert result["finite_inputs_checked"] == 66300
    assert result["nonfinite_inputs_rejected"] == 3


@pytest.mark.parametrize("lower", range(126))
def test_every_halfway_rounds_even_with_neighbor_and_sign_controls(lower):
    midpoint = (E4M3_POSITIVE[lower] + E4M3_POSITIVE[lower + 1]) / 2
    for sign, sign_bit in ((1, 0), (-1, 128)):
        for value, code in ((math.nextafter(midpoint, 0), lower),
                            (midpoint, lower if lower % 2 == 0 else lower + 1),
                            (math.nextafter(midpoint, math.inf), lower + 1)):
            assert e4m3_round_bits(sign * value) == (code | sign_bit)
            assert e4m3_round_bits_arithmetic(sign * value) == (code | sign_bit)


@pytest.mark.parametrize("value,code", [(0.0, 0), (-0.0, 128), (448.0, 126),
                                       (-448.0, 254), (449, 126), (-449, 254),
                                       (1e300, 126), (-1e300, 254)])
def test_clamp_and_signed_zero(value, code):
    assert e4m3_round_bits(value) == code
    assert e4m3_round_bits_arithmetic(value) == code
    assert math.copysign(1, e4m3_value(code)) == math.copysign(1, value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fixture_policy_rejects(value):
    with pytest.raises(ValueError, match="nonfinite"):
        e4m3_round_bits(value)
    with pytest.raises(ValueError, match="nonfinite"):
        e4m3_round_bits_arithmetic(value)


def test_both_e4m3_nan_encodings_are_not_finite():
    assert math.isnan(e4m3_value(127))
    assert math.isnan(e4m3_value(255))
    assert all(math.isfinite(e4m3_value(code)) for code in range(256) if code & 127 != 127)


@pytest.mark.parametrize("key,want", TILES.items())
def test_exact_exl3_trellis_tile_matches_pinned_upstream(key, want):
    codebook, bits = key.split("/")
    bits = float(bits)
    assert digest(value_bytes(decode_tile(synthetic_raw(bits), bits, codebook))) == want


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_exlr8_symbol_history_equals_exl3_integer_span(bits):
    for tile in range(8):
        raw = synthetic_raw(bits, tile)
        assert tile_states(raw, bits) == exlr8_symbol_states(raw, bits)


@pytest.mark.parametrize("key,want", DENSE.items())
def test_exact_exlr8_k2_k3_k4_both_shard_orientations(key, want):
    bits, side = key.split("/")
    raw, matrix = dense_fixture(int(bits), side, 256, 256)
    assert len(raw) == int(bits) * 32 * 256
    assert digest(value_bytes(matrix, "fp32")) == want


def test_tile_coordinate_order_is_bijection_with_known_lane_boundaries():
    assert len({tile_coordinates(index) for index in range(256)}) == 256
    assert [tile_coordinates(index) for index in range(8)] == [
        (0, 0), (1, 0), (8, 0), (9, 0), (0, 8), (1, 8), (8, 8), (9, 8)]
    assert tile_coordinates(255) == (15, 15)


def test_tail_biting_one_bit_fixture_wraps_final_stream_into_first_state():
    raw = bytes(32)
    assert tile_states(raw, 1) == [0] * 256
    raw = raw[:-4] + bytes([1, 0, 0, 0])  # final stream bit: LSB of last LE32 word
    states = tile_states(raw, 1)
    assert any(states[:16])  # early states include the end of the ring
    assert any(states[-32:])
    assert len(states) == 256


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_unswapped_words_and_reversed_bits_are_rejected(bits):
    raw = synthetic_raw(bits)
    swapped = b"".join(raw[index + 2:index + 4] + raw[index:index + 2]
                       for index in range(0, len(raw), 4))
    reversed_bits = bytes(int(f"{byte:08b}"[::-1], 2) for byte in raw)
    want = value_bytes(decode_tile(raw, bits, exlr8=True), "fp32")
    assert value_bytes(decode_tile(swapped, bits, exlr8=True), "fp32") != want
    assert value_bytes(decode_tile(reversed_bits, bits, exlr8=True), "fp32") != want


def test_exlr8_decode_uses_its_symbol_path():
    with mock.patch("lab.conformance.reference.exlr8_symbol_states", return_value=[0] * 256) as spy:
        result = decode_tile(synthetic_raw(2), 2, exlr8=True)
    spy.assert_called_once()
    assert result == [codebook_values("exlr8")[0]] * 256


def test_stock_unrounded_exl3_is_a_negative_control():
    report = cpu_report("exlr8")
    assert report["status"] == "PASS"
    assert report["negative_control"]["different_fp32_states"] == 64372
    assert report["negative_control"]["status"] == "REJECTED"
    for bits in (2, 3, 4):
        raw = synthetic_raw(bits)
        assert value_bytes(decode_tile(raw, bits), "fp32") != value_bytes(decode_tile(raw, bits, exlr8=True), "fp32")


@pytest.mark.parametrize("inp,out,side", [(128, 256, "in"), (256, 128, "out"),
                                         (128, 256, "out"), (256, 128, "in")])
def test_non_square_fixture_has_huggingface_orientation_and_first_tile(inp, out, side):
    _, matrix = dense_fixture(3, side, inp, out)
    tile = decode_tile(synthetic_raw(3), 3, exlr8=True)
    assert len(matrix) == inp * out
    for r in range(16):
        for c in range(16):
            assert struct.pack("<f", matrix[c * inp + r]) == struct.pack("<f", tile[r * 16 + c])


@pytest.mark.parametrize("value", [-1, 65536, True, 1.5])
def test_invalid_states_rejected(value):
    with pytest.raises(ValueError):
        exl3_value(value)


@pytest.mark.parametrize("bits", [0, 9, float("nan"), True])
def test_invalid_widths_rejected(bits):
    with pytest.raises(ValueError):
        tile_states(bytes(64), bits)


@pytest.mark.parametrize("raw", [b"", bytes(63), bytearray(64)])
def test_payload_length_and_type_rejected(raw):
    with pytest.raises(ValueError):
        tile_states(raw, 2)
    with pytest.raises(ValueError):
        exlr8_symbol_states(raw, 2)


def test_half_width_and_exlr8_contracts_rejected():
    with pytest.raises(ValueError, match="half widths"):
        decode_tile(synthetic_raw(2.5), 2.5, "mcg")
    with pytest.raises(ValueError):
        decode_tile(synthetic_raw(4), 4, "mul1", exlr8=True)
    with pytest.raises(ValueError):
        decode_tile(synthetic_raw(8), 8, exlr8=True)
    with pytest.raises(ValueError):
        dense_fixture(2, "in", 1024, 128)


def test_fixture_hash_stable_and_cpu_claims_bounded():
    first = fixtures()
    assert first == fixtures()
    assert len(first["fixtures"]) == 33
    assert first["fixture_set_sha256"] == "7b0d2f9f27cb5be7ef076565a53474a76c4e613498ddb620cc7b80209d6f4d5e"
    report = cpu_report()
    assert report["status"] == "PASS"
    assert report["scope"] == SCOPE
    assert report["gpu_status"] == "UNAVAILABLE"
    assert report["matmul_status"] == "NOT_RUN"
    assert report["dense_reconstruction_status"] == "EXTERNAL_ONLY"


def test_pinned_fixture_tamper_fails_cpu_report(monkeypatch):
    monkeypatch.setitem(TILES, "mcg/2", "0" * 64)
    report = cpu_report("exl3")
    assert report["status"] == "FAIL"
    assert "mcg/2" in report["fixture_failures"]


def test_pinned_codebook_tamper_fails_cpu_report(monkeypatch):
    monkeypatch.setitem(GOLDEN, "mcg", "0" * 64)
    report = cpu_report("exl3")
    assert report["status"] == "FAIL"
    assert "pinned exhaustive codebook hash mismatch" in report["errors"]


@pytest.mark.parametrize("runner", ["tensorfold-cuda", "gr8-dense"])
def test_manifest_is_pinned_plan_only_and_bounded(runner):
    report = runner_manifest(runner)
    assert report["status"] == "PLAN_ONLY"
    assert report["executes"] is False
    assert report["scope"] == SCOPE
    assert sum(command["max_seconds"] for command in report["commands"]) <= report["max_seconds"]
    for upstream in report["upstreams"].values():
        assert len(upstream["revision"]) == 40
        assert all(len(sha) == 64 for sha in upstream["files"].values())
    if runner == "tensorfold-cuda":
        assert report["toleranced_matmul"]["threshold"] == 0.002
        assert report["exact_decoder"]["comparison"] == "fp16 bit patterns; zero tolerance"
    else:
        assert report["toleranced_matmul"] is None
        assert report["exact_decoder"]["matrices"] == 11


def _temporary_pinned_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "oracle.py").write_text("# synthetic pin check fixture\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True, check=True).stdout.strip()
    git("init")
    git("add", "oracle.py")
    git("-c", "user.name=Conformance test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    pin = {"revision": git("rev-parse", "HEAD"), "files": {"oracle.py": digest((source / "oracle.py").read_bytes())}}
    monkeypatch.setitem(UPSTREAMS, "exlr8", pin)
    return source


def test_source_checker_fails_on_revision_and_tracked_content_drift(tmp_path, monkeypatch):
    source = _temporary_pinned_source(tmp_path, monkeypatch)
    assert check_source("exlr8", source)["status"] == "PASS"
    (source / "oracle.py").write_text("# altered\n")
    report = check_source("exlr8", source)
    assert report["status"] == "FAIL"
    assert "tracked source differs from HEAD" in report["errors"]
    assert "file pin mismatch: oracle.py" in report["errors"]
    monkeypatch.setitem(UPSTREAMS["exlr8"], "revision", "0" * 40)
    assert "HEAD differs from pinned revision" in check_source("exlr8", source)["errors"]


def test_source_checker_rejects_symlink_file(tmp_path, monkeypatch):
    source = _temporary_pinned_source(tmp_path, monkeypatch)
    (source / "oracle.py").unlink()
    (source / "oracle.py").symlink_to(Path(__file__))
    with pytest.raises(ValueError, match="symlinks"):
        check_source("exlr8", source)


def test_source_checker_accepts_real_checkout_through_symlinked_parent(tmp_path, monkeypatch):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    source = _temporary_pinned_source(real_parent, monkeypatch)
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    assert not (alias / "source").is_symlink()
    assert check_source("exlr8", alias / "source")["status"] == "PASS"


def test_cli_dispatch_and_cuda_fail_unavailable(capsys):
    assert lab_main(["conformance", "cuda"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "UNAVAILABLE"
    assert report["simulated_gpu"] is False
    assert report["executed_tests"] == 0
    assert main(["manifest", "--runner", "gr8-dense"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"


def test_cli_cpu_and_invalid_source_exit_codes(capsys, tmp_path):
    assert main(["cpu", "--format", "exl3"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert main(["check-sources", "--upstream", "exlr8", "--source", str(tmp_path / "absent")]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_saved_examples_are_current_and_each_has_explicit_scope():
    directory = Path(__file__).resolve().parents[1] / "examples" / "conformance"
    expected = {
        "expected-cpu.json": cpu_report(), "fixtures.json": fixtures(),
        "tensorfold-cuda.json": runner_manifest("tensorfold-cuda"),
        "gr8-dense.json": runner_manifest("gr8-dense"),
    }
    for name, value in expected.items():
        assert json.loads((directory / name).read_text()) == value
        assert value["scope"] == SCOPE


def test_cpu_and_fixtures_have_no_network_or_subprocess_calls(capsys):
    with mock.patch("subprocess.run", side_effect=AssertionError("unexpected subprocess")), \
            mock.patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
        assert main(["cpu", "--format", "exlr8"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "PASS"
        assert main(["fixtures", "--format", "exl3"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "PASS"
