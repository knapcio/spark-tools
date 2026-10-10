# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream-shaped fixtures, hostile sources and empirical split checks."""
import copy
import json
from pathlib import Path
import socket
import subprocess

import pytest

from common.lab_cli import main as lab_main
from lab.observations import InvalidObservation, calibrate, import_observations, lookup
from lab.observations.core import strict_json

FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "observations"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def vllm(data=None, meta=None):
    return import_observations("vllm-bench-serve-v1", json.dumps(data or load("vllm-bench-serve.json")).encode(),
                               meta or load("vllm-import.json"))


def prom(before=None, after=None, meta=None):
    return import_observations("tensorfold-prometheus-v1",
                               after if after is not None else (FIXTURES / "tensorfold-after.prom").read_bytes(),
                               meta or load("tensorfold-prometheus-import.json"),
                               before=before if before is not None else (FIXTURES / "tensorfold-before.prom").read_bytes())


def test_vllm_units_rates_and_source_identity():
    result = vllm()
    assert result["schema"] == "spark-observations" and result["version"] == 1
    assert result["counts"] == {"accepted": 4, "rejected": 0}
    assert result["aggregates"]["aggregate_output_rate"]["value"] == 8
    row = result["requests"][0]["metrics"]
    assert row["decode_rate"]["value"] == pytest.approx(10)
    assert row["decode_rate"]["unit"] == "token/second"
    assert row["inter_token_latency"]["unit"] == "second"
    assert result["aggregates"]["aggregate_output_rate"]["unit"] == "token/wall-second"
    assert result["aggregates"]["acceptance"]["value"] == .5
    assert result["scope"]["position_survival"] == "unavailable"
    assert "fixture only" not in json.dumps(result)


@pytest.mark.parametrize("field,value", [("output_lens", [4]), ("ttfts", [float("nan")]*4),
    ("completed", 3), ("failed", 1), ("duration", 0), ("duration", True),
    ("input_lens", [64.5]*4), ("total_output_tokens", 99), ("output_throughput", 99),
    ("spec_decode_accepted_tokens", 17), ("errors", ["error", "", "", ""]),
    ("latencies", [0.01]*4), ("itls", [[1]]*4)])
def test_vllm_invalid_measurements(field, value):
    data = load("vllm-bench-serve.json")
    data[field] = value
    with pytest.raises(InvalidObservation):
        vllm(data)


def test_failed_request_excluded_and_counted():
    data = load("vllm-bench-serve.json")
    data.update(completed=3, failed=1, total_input_tokens=256, total_output_tokens=12,
                output_throughput=6, request_throughput=1.5, total_token_throughput=134)
    data["errors"][0] = "private failure description"
    for field in ("output_lens", "ttfts", "latencies"):
        data[field][0] = 0
    data["itls"][0] = []
    result = vllm(data)
    assert result["counts"] == {"accepted": 3, "rejected": 1}
    assert "private failure" not in json.dumps(result)


def test_native_preserves_reported_rate_and_does_not_infer_round_cost():
    result = import_observations("tensorfold-native-v1", (FIXTURES / "tensorfold-native.json").read_bytes(),
                                load("tensorfold-native-import.json"))
    values = result["requests"][0]["metrics"]
    assert values["decode_rate"]["value"] == 10
    assert values["tokens_per_round"]["value"] == 2
    assert values["acceptance"]["basis"] == "accepted/drafted"
    assert "decode_seconds_per_round" not in values
    assert "inter_token_latencies" in result["missing_fields"]


@pytest.mark.parametrize("field,value", [("acceptance_rate", .9), ("tokens_per_round", 99), ("accepted", 5), ("rounds", 0)])
def test_native_inconsistent_speculative(field, value):
    data = load("tensorfold-native.json")
    data["speculative"][field] = value
    with pytest.raises(InvalidObservation):
        import_observations("tensorfold-native-v1", json.dumps(data).encode(), load("tensorfold-native-import.json"))


def test_native_mixed_workloads_rejected():
    data = load("tensorfold-native.json")
    second = copy.deepcopy(data)
    second["tensorfold"]["enable_thinking"] = True
    meta = load("tensorfold-native-import.json")
    meta["request_ids"] = ["n1", "n2"]
    with pytest.raises(InvalidObservation, match="mixed"):
        import_observations("tensorfold-native-v1", json.dumps([data, second]).encode(), meta)


def test_prometheus_verified_mirrors_and_matched_round_cost():
    result = prom()
    assert result["counts"] == {"accepted": 2, "rejected": 0}
    assert result["aliases"]["drafted"]["selected"] == "tensorfold:mtp_drafted_total"
    assert result["aliases"]["drafted"]["verified_mirrors"] == ["tensorfold:spec_decode_num_draft_tokens_total"]
    assert result["aggregates"]["accepted_tokens"]["value"] == 4
    assert result["aggregates"]["aggregate_output_rate"]["value"] == 8
    assert result["aggregates"]["decode_seconds_per_round"]["value"] == .15
    assert result["aggregates"]["pooled_decode_rate"]["unit"] == "token/request-second"
    assert "key=" not in json.dumps(result)


def no_http_snapshots():
    return ((FIXTURES / "tensorfold-before-no-http.prom").read_bytes(),
            (FIXTURES / "tensorfold-after-no-http.prom").read_bytes())


def test_native_prometheus_without_http_family_has_explicit_unknown_failure_visibility():
    before, after = no_http_snapshots()
    result = prom(before=before, after=after)
    assert result["counts"]["accepted"] == 2
    assert result["scope"]["failure_visibility"] == "unavailable"
    assert "requests" not in result["aliases"]


def success_companion():
    data = wrapper()
    data["responses"].append(copy.deepcopy(data["responses"][0]))
    return data


def test_native_success_companion_matches_all_interval_counters():
    before, after = no_http_snapshots()
    result = import_observations("tensorfold-prometheus-v1", after, load("tensorfold-prometheus-import.json"),
                                 before=before, client_capture=json.dumps(success_companion()).encode())
    assert result["scope"]["failure_visibility"] == "client-response-proof"
    assert len(result["client_capture_sha256"]) == 64


@pytest.mark.parametrize("change", ["failed", "mismatch", "unversioned"])
def test_native_success_companion_failed_mismatched_and_unversioned_rejected(change):
    before, after = no_http_snapshots()
    data = success_companion()
    if change == "failed":
        data["responses"][0] = {"response": {"error": {"message": "private failure"}}}
    elif change == "mismatch":
        data["responses"][0]["response"]["usage"].update(prompt_tokens=65, total_tokens=69)
    else:
        data = [row["response"] for row in data["responses"]]
    with pytest.raises(InvalidObservation):
        import_observations("tensorfold-prometheus-v1", after, load("tensorfold-prometheus-import.json"),
                            before=before, client_capture=json.dumps(data).encode())


def test_prometheus_strict_default_rejects_duplicate_aliases():
    meta = load("tensorfold-prometheus-import.json")
    del meta["alias_policy"]
    with pytest.raises(InvalidObservation, match="aliases"):
        prom(meta=meta)


@pytest.mark.parametrize("old,new", [
    (b"mtp_accepted_total 4", b"mtp_accepted_total 5"),
    (b"request_decode_time_seconds_sum 0.6", b"request_decode_time_seconds_sum 0.9"),
    (b'key="fixture"', b'key="changed"'),
    (b"generation_tokens_total 8", b"generation_tokens_total NaN"),
    (b"requests_running 0", b"requests_running 1"),
    (b"preemptions_total 0", b"preemptions_total 1"),
    (b"client_disconnections_total 0", b"client_disconnections_total 1"),
    (b'tensorfold:request_decode_seconds_bucket{le="0.01"} 0\n', b""),
    (b"tensorfold:request_decode_time_seconds_count 2\n", b""),
    (b'key="fixture",status="200"', b'key="fixture",status="500"'),
])
def test_prometheus_invalid_aliases_labels_incomplete_and_failed(old, new):
    after = (FIXTURES / "tensorfold-after.prom").read_bytes().replace(old, new)
    with pytest.raises(InvalidObservation):
        prom(after=after)


def test_counter_reset_rejected():
    before = (FIXTURES / "tensorfold-before.prom").read_bytes().replace(b"generation_tokens_total 0", b"generation_tokens_total 99")
    with pytest.raises(InvalidObservation, match="reset"):
        prom(before=before)


def test_histogram_interval_delta_must_remain_cumulative_and_count_bounded():
    def rewrite(raw, first, second, rest, total, count):
        lines = raw.decode().splitlines()
        for family in ("request_latency_seconds", "e2e_request_latency_seconds"):
            prefix = f"tensorfold:{family}_bucket"
            seen = 0
            for index, line in enumerate(lines):
                if line.startswith(prefix):
                    value = first if seen == 0 else second if seen == 1 else rest
                    lines[index] = line.rsplit(" ", 1)[0] + f" {value}"
                    seen += 1
                elif line.startswith(f"tensorfold:{family}_sum"):
                    lines[index] = f"tensorfold:{family}_sum {total}"
                elif line.startswith(f"tensorfold:{family}_count"):
                    lines[index] = f"tensorfold:{family}_count {count}"
        return ("\n".join(lines) + "\n").encode()

    before = rewrite((FIXTURES / "tensorfold-before.prom").read_bytes(), 0, 5, 10, 4, 10)
    after = rewrite((FIXTURES / "tensorfold-after.prom").read_bytes(), 5, 5, 12, 4.8, 12)
    with pytest.raises(InvalidObservation, match="histogram interval"):
        prom(before=before, after=after)


@pytest.mark.parametrize("field,value", [("isolated", False), ("drained", False), ("workload_id", "other"), ("duration_seconds", 0)])
def test_unmatched_interval_rejected(field, value):
    meta = load("tensorfold-prometheus-import.json")
    meta["interval"][field] = value
    with pytest.raises(InvalidObservation):
        prom(meta=meta)


def test_calibration_disjoint_lookup_holdout_and_unsupported_cells():
    result = calibrate([vllm()], load("split.json"))
    rate = next(x for x in result["cells"] if x["cell"]["metric"] == "decode_rate" and x["status"] == "supported")
    assert rate["estimate"] == pytest.approx(10)
    assert rate["holdout_errors"][0]["observed"] == pytest.approx(7.5)
    assert rate["holdout_errors"][0]["absolute_error_percent"] == pytest.approx(100/3)
    assert lookup(result, rate["cell"])["status"] == "supported"
    cell = copy.deepcopy(rate["cell"])
    cell["input_tokens"] = 65
    assert lookup(result, cell) == {"status": "unavailable", "estimate": None, "reason": "unsupported-exact-cell"}
    assert all(x["estimate"] is None for x in result["cells"] if x["status"] == "unavailable")
    assert set(result["train_request_ids"]).isdisjoint(result["holdout_request_ids"])


@pytest.mark.parametrize("change", ["overlap", "missing", "duplicates", "unknown-metric"])
def test_bad_explicit_splits(change):
    split = load("split.json")
    if change == "overlap":
        split["holdout_request_ids"].append("vllm-r1")
    elif change == "missing":
        split["holdout_request_ids"].pop()
    elif change == "duplicates":
        split["train_request_ids"].append("vllm-r1")
    else:
        split["metrics"] = ["kernel_cost"]
    with pytest.raises(InvalidObservation):
        calibrate([vllm()], split)


def test_request_ids_reused_across_sources_rejected():
    with pytest.raises(InvalidObservation, match="reused"):
        calibrate([vllm(), vllm()], load("split.json"))


def test_source_relabeling_cannot_create_an_independent_holdout():
    second = vllm()
    second["identity"]["source_id"] = "relabelled"
    for row in second["requests"]:
        row["request_id"] += "-copy"
    split = load("split.json")
    split["holdout_request_ids"] += [row["request_id"] for row in second["requests"]]
    with pytest.raises(InvalidObservation, match="source observation reused"):
        calibrate([vllm(), second], split)


@pytest.mark.parametrize("field,value", [("tokens_per_second", .001), ("prefill_seconds", 99), ("tokens_per_second", float("inf"))])
def test_native_rate_and_timing_inconsistent_with_counts(field, value):
    data = load("tensorfold-native.json")
    data["tensorfold"][field] = value
    with pytest.raises(InvalidObservation):
        import_observations("tensorfold-native-v1", json.dumps(data).encode(), load("tensorfold-native-import.json"))


def test_native_live_lanes_engine_and_patched_pin():
    data = load("tensorfold-native.json")
    data["tensorfold"]["engine"] = "lanes"
    meta = load("tensorfold-native-import.json")
    meta["runtime"]["source_revision"] += "+mia-364f21a1a46a366fa39c4da0d53b7b1f38624610"
    result = import_observations("tensorfold-native-v1", json.dumps(data).encode(), meta)
    assert result["identity"]["runtime"]["source_revision"] == meta["runtime"]["source_revision"]
    assert "observed_settings_sha256" in result["identity"]["runtime"]


def wrapper():
    return {"schema": "spark-tensorfold-native-capture", "version": 1,
            "responses": [{"response": load("tensorfold-native.json"),
                           "client": {"client_ttft_seconds": .1, "client_e2e_seconds": .5,
                                      "itls_seconds": [.1, .1, .1], "stream_token_events": 4}}]}


def test_native_versioned_client_streaming_wrapper():
    result = import_observations("tensorfold-native-v1", json.dumps(wrapper()).encode(), load("tensorfold-native-import.json"))
    values = result["requests"][0]["metrics"]
    assert values["client_decode_rate"]["value"] == pytest.approx(7.5)
    assert values["decode_rate"]["value"] == 10
    assert values["client_ttft"]["basis"] == "client-sse-monotonic"
    assert values["inter_token_latency"]["values"] == [.1, .1, .1]
    assert values["inter_token_latency"]["basis"] == "client-sse-monotonic-content-event"


@pytest.mark.parametrize("field,value", [("client_ttft_seconds", .6), ("client_e2e_seconds", float("nan")),
    ("itls_seconds", [.5, .5, .5]), ("stream_token_events", 2), ("stream_token_events", 5),
    ("itls_seconds", [float("inf")]*3), ("stream_token_events", True)])
def test_native_client_streaming_validation(field, value):
    data = wrapper()
    data["responses"][0]["client"][field] = value
    with pytest.raises(InvalidObservation):
        import_observations("tensorfold-native-v1", json.dumps(data).encode(), load("tensorfold-native-import.json"))


def test_unversioned_native_response_does_not_trust_client_metrics():
    data = load("tensorfold-native.json")
    data["client"] = wrapper()["responses"][0]["client"]
    result = import_observations("tensorfold-native-v1", json.dumps(data).encode(), load("tensorfold-native-import.json"))
    assert "client_ttft" not in result["requests"][0]["metrics"]


def test_unrecognized_capture_wrapper_rejected():
    data = wrapper()
    data["version"] = 2
    with pytest.raises(InvalidObservation):
        import_observations("tensorfold-native-v1", json.dumps(data).encode(), load("tensorfold-native-import.json"))


@pytest.mark.parametrize("source", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{'])
def test_strict_json_nonfinite_duplicates_and_malformed(source):
    with pytest.raises(InvalidObservation):
        strict_json(source)


def test_library_offline_and_allowlist_privacy(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("transport or process invoked")
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    data = load("vllm-bench-serve.json")
    data.update(endpoint="private endpoint", api_key="private credential", log="unrelated private log")
    data["generated_texts"] = ["private output"]*4
    result = vllm(data)
    assert "private" not in json.dumps(result)
    calibrate([result], load("split.json"))
    prom()


def test_cli_import_calibrate_and_refuses_overwrite(tmp_path, capsys):
    out = tmp_path / "observations.json"
    args = ["observations", "import", "--adapter", "vllm-bench-serve-v1", "--input", str(FIXTURES / "vllm-bench-serve.json"),
            "--metadata", str(FIXTURES / "vllm-import.json"), "--output", str(out)]
    assert lab_main(args) == 0
    assert lab_main(args) == 2
    assert "no observation exported" in capsys.readouterr().err
    assert lab_main(["observations", "calibrate", "--input", str(out), "--split", str(FIXTURES / "split.json")]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "spark-observation-calibration"


def test_cli_symlink_input_and_error_privacy(tmp_path, capsys):
    link = tmp_path / "private-path.json"
    link.symlink_to(FIXTURES / "vllm-bench-serve.json")
    assert lab_main(["observations", "import", "--adapter", "vllm-bench-serve-v1", "--input", str(link),
                     "--metadata", str(FIXTURES / "vllm-import.json")]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "private-path" not in captured.err
