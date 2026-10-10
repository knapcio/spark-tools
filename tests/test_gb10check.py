# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest
from common.config import load
from gb10check.cli import container_command, inspect, main
from gb10check.resources import occupancy, parse_ptxas
from gb10check.ordering import barrier_cut, instruction_checks

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/gb10check"


def test_passing_and_deliberate_failures():
    passed = inspect(load(EXAMPLE / "pass.json"), EXAMPLE)
    assert passed["status"] == "PASS"
    assert passed["resources"][0]["occupancy"]["active_blocks_per_sm"] == 12
    failed = inspect(load(EXAMPLE / "fail.json"), EXAMPLE)
    assert failed["status"] == "FAIL"
    assert not failed["resources"][0]["checks"]["spills"]
    assert not failed["resources"][0]["checks"]["shared_memory"]


@pytest.mark.parametrize("case,expected_code", [("pass", 0), ("fail", 1)])
def test_parse_cli_reproduces_documented_outputs(case, expected_code, tmp_path):
    output = tmp_path / "report.json"
    assert main(["parse", "--config", str(EXAMPLE / (case + ".json")),
                 "--out", str(output)]) == expected_code
    assert load(output) == load(EXAMPLE / ("expected-" + case + ".json"))


@pytest.mark.parametrize("change", ["missing", "identity", "target", "duplicate", "compiler_error"])
def test_incomplete_logs_fail_closed(change):
    text = (EXAMPLE / "pass.ptxas").read_text()
    if change == "missing": text = text.replace("0 bytes spill stores", "unknown spills")
    if change == "identity": text = text.replace("properties for sample_copy", "properties for other")
    if change == "target": text = text.replace("sm_121", "sm_90")
    if change == "duplicate": text += text
    if change == "compiler_error": text += "ptxas fatal : invalid input"
    with pytest.raises(ValueError):
        parse_ptxas(text)


def test_explicit_launch_geometry_and_target():
    resource = parse_ptxas((EXAMPLE / "pass.ptxas").read_text())[0]
    assert occupancy(resource)["status"] == "unknown"
    with pytest.raises(ValueError): occupancy(resource, True, 0)
    config = load(EXAMPLE / "pass.json")
    config["target"] = "sm_121a"
    with pytest.raises(ValueError): inspect(config, EXAMPLE)


def ptx(body):
    return ".visible .entry example()\n{\nld.param.u64 %rd1, [example_param_3];\nld.param.u64 %rd2, [example_param_20];\nld.global.u32 %r1, [%rd1];\n" + body + "\nret;\n}"


def test_barrier_cut_and_pointer_provenance():
    good = ptx("bar.sync 0;\ndiscard.global.L2 [%rd1], 128;\nst.global.u32 [%rd2], %r1;")
    assert all(barrier_cut(good, 3, 20)["checks"].values())
    bad = good.replace("bar.sync 0;", "@%p1 bar.sync 0;")
    assert not barrier_cut(bad, 3, 20)["checks"]["scratch_read_barrier_cut"]
    with pytest.raises(ValueError): barrier_cut(good.replace("L2 [%rd1]", "L2 [%rd2]"), 3, 20)


def test_conditional_path_bypasses_barrier():
    text = ptx("@%p1 bra skip;\nbar.sync 0;\nskip:\ndiscard.global.L2 [%rd1], 128;")
    assert not barrier_cut(text, 3)["checks"]["scratch_read_barrier_cut"]


def test_loop_backedge_requires_barrier():
    text = ptx("again:\ndiscard.global.L2 [%rd1], 128;\nld.global.u32 %r2, [%rd1];\n@%p1 bra again;")
    assert not barrier_cut(text, 3)["checks"]["scratch_read_barrier_cut"]


def test_instruction_order_and_counts():
    assert instruction_checks("LDS BAR.SYNC STG", {"sequences": [["LDS", "BAR", "STG"]]})["sequence_0"]
    assert not instruction_checks("LDS STG BAR.SYNC", {"sequences": [["LDS", "BAR", "STG"]]})["sequence_0"]
    assert not instruction_checks("BAR BAR", {"unchanged_counts": ["BAR"]}, "BAR")["count_0"]
    with pytest.raises(ValueError): instruction_checks("BAR", {"unchanged_counts": ["BAR"]})


def test_container_mounts_are_local_read_only(tmp_path):
    command = container_command(EXAMPLE / "compile-pass.json", "example/toolkit:13", tmp_path)
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--platform") + 1] == "linux/arm64"
    assert not any("--gpus" == value for value in command)
    assert sum("readonly" in value for value in command) == 3
    assert any("target=/tool/gb10check,readonly" in value for value in command)
    assert any("target=/tool/common,readonly" in value for value in command)
    assert not any("target=/tool,readonly" in value for value in command)
    with pytest.raises(ValueError): container_command(EXAMPLE / "compile-pass.json", "image", EXAMPLE / "out")


def test_linux_image_platform(tmp_path):
    command = container_command(EXAMPLE / 'compile-pass.json', 'example/toolkit:13', tmp_path,
                                platform='linux/amd64')
    assert command[command.index('--platform')+1] == 'linux/amd64'
