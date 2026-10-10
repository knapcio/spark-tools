# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream formats; no network, process execution or free-text export."""
import math
import re

from .core import (MAX_REQUESTS, PINS, digest, identity, metric, number, report,
                   request_ids, require, strict_json)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _vllm(source, meta):
    data = strict_json(source)
    require(isinstance(data, dict), "benchmark object required")
    fields = ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors")
    require(all(isinstance(data.get(k), list) for k in fields), "detailed benchmark arrays required")
    n = len(data["input_lens"])
    ids = request_ids(meta, n)
    require(all(len(data[k]) == n for k in fields), "inconsistent benchmark array lengths")
    completed = number(data.get("completed"), integer=True)
    failed = number(data.get("failed"), integer=True)
    require(completed + failed == n, "benchmark completion counts mismatch")
    duration = number(data.get("duration"), positive=True)
    rows, rejected = [], 0
    input_total, output_total = 0, 0
    for i, rid in enumerate(ids):
        error = data["errors"][i]
        require(isinstance(error, str), "invalid benchmark error field")
        inp = number(data["input_lens"][i], integer=True)
        out = number(data["output_lens"][i], integer=True)
        ttft = number(data["ttfts"][i])
        latency = number(data["latencies"][i])
        intervals = data["itls"][i]
        require(isinstance(intervals, list) and len(intervals) <= MAX_REQUESTS, "invalid inter-token latency array")
        intervals = [number(x) for x in intervals]
        if error:
            # Failed vLLM outputs have zero actual output length and no timing.
            require(out == 0 and ttft == 0 and latency == 0 and not intervals,
                    "failed request carries successful measurements")
            rejected += 1
            continue
        require(inp > 0 and out > 0 and latency >= ttft, "invalid successful request")
        require(sum(intervals) <= latency - ttft + 1e-6, "inter-token sum exceeds request latency")
        values = {"input_tokens": metric(inp, "token", "client"),
                  "output_tokens": metric(out, "token", "client"),
                  "ttft": metric(ttft, "second", "client"),
                  "latency": metric(latency, "second", "client")}
        if out > 1 and latency > ttft:
            values["decode_rate"] = metric((out - 1) / (latency - ttft), "token/second", "client-after-first-token")
            values["time_per_output_token"] = metric((latency - ttft) / (out - 1), "second/token", "client-after-first-token")
        if intervals:
            # Chunked streaming may deliver more than one token per event.
            values["inter_token_latency"] = {"values": intervals, "unit": "second", "basis": "client-stream-event"}
        rows.append({"request_id": rid, "metrics": values})
        input_total += inp
        output_total += out
    require(len(rows) == completed and rejected == failed, "failed requests mixed with successful counts")
    require(number(data.get("total_input_tokens"), integer=True) == input_total
            and number(data.get("total_output_tokens"), integer=True) == output_total,
            "benchmark token totals mismatch")
    aggregates = {"aggregate_output_rate": metric(output_total / duration, "token/wall-second", "client-whole-run"),
                  "wall_duration": metric(duration, "second", "client-whole-run")}
    recognized = list(fields) + ["completed", "failed", "duration", "total_input_tokens", "total_output_tokens"]
    for field, expected in (("output_throughput", output_total / duration),
                            ("request_throughput", completed / duration),
                            ("total_token_throughput", (input_total + output_total) / duration)):
        if field in data:
            require(_close(number(data[field]), expected), "reported benchmark throughput mismatch")
            recognized.append(field)
    drafted = data.get("spec_decode_draft_tokens")
    accepted = data.get("spec_decode_accepted_tokens")
    if drafted is not None or accepted is not None:
        d, a = number(drafted, integer=True), number(accepted, integer=True)
        require(a <= d, "accepted tokens exceed drafted tokens")
        recognized += ["spec_decode_draft_tokens", "spec_decode_accepted_tokens"]
        aggregates["drafted_tokens"] = metric(d, "token", "server-whole-run")
        aggregates["accepted_tokens"] = metric(a, "token", "server-whole-run")
        if d:
            aggregates["acceptance"] = metric(a / d, "ratio", "accepted/drafted")
    missing = [k for k in ("spec_decode_draft_tokens", "spec_decode_accepted_tokens") if k not in data]
    return report("vllm-bench-serve-v1", meta, source, recognized, missing, rows, aggregates, rejected)


def _native(source, meta):
    data = strict_json(source)
    wrapped = isinstance(data, dict) and "schema" in data
    if wrapped:
        require(data.get("schema") == "spark-tensorfold-native-capture" and type(data.get("version")) is int
                and data["version"] == 1 and isinstance(data.get("responses"), list), "unsupported native capture wrapper")
        data = data["responses"]
    else:
        data = data if isinstance(data, list) else [data]
    ids = request_ids(meta, len(data))
    rows, rejected, recognized = [], 0, set()
    settings = None
    for rid, entry in zip(ids, data):
        require(isinstance(entry, dict), "native response object required")
        reply = entry.get("response") if wrapped else entry
        require(isinstance(reply, dict), "native response object required")
        if "error" in reply:
            require(not any(k in reply for k in ("usage", "tensorfold", "speculative")), "failed native response carries successful measurements")
            require(not wrapped or "client" not in entry, "failed native response carries client measurements")
            rejected += 1
            recognized.add("error")
            continue
        usage, runtime, spec = (reply.get(k) for k in ("usage", "tensorfold", "speculative"))
        require(all(isinstance(x, dict) for x in (usage, runtime, spec)), "complete native response telemetry required")
        inp = number(usage.get("prompt_tokens"), integer=True, positive=True)
        out = number(usage.get("completion_tokens"), integer=True, positive=True)
        require(number(usage.get("total_tokens"), integer=True) == inp + out, "native usage totals mismatch")
        require(isinstance(runtime.get("engine"), str) and bool(runtime["engine"]), "native engine identity required")
        current = tuple(runtime.get(k) for k in ("engine", "enable_thinking", "reasoning_effort", "sampling", "drafts"))
        require(type(runtime.get("enable_thinking")) is bool and type(runtime.get("drafts")) is bool
                and runtime.get("reasoning_effort") in (None, "none", "low", "medium", "high", "max")
                and runtime.get("sampling") in ("exact", "greedy"), "invalid native workload settings")
        require(settings is None or settings == current, "mixed native workload settings")
        settings = current
        seconds = number(runtime.get("seconds"))
        rate = number(runtime.get("tokens_per_second"))
        d, a, rounds = (number(spec.get(k), integer=True) for k in ("drafted", "accepted", "rounds"))
        require(a <= d and rounds > 0, "invalid native draft counters")
        require(_close(number(spec.get("acceptance_rate")), a / d if d else 0)
                and _close(number(spec.get("tokens_per_round")), out / rounds), "native speculative ratios mismatch")
        require((out > 1 and rate > 0) or (out == 1 and rate == 0), "native decode-rate token mismatch")
        require(out == 1 or (out - 1) / rate <= seconds + 1e-6,
                "native decode rate incompatible with token count and total duration")
        if runtime.get("prefill_seconds") is not None:
            require(number(runtime["prefill_seconds"]) <= seconds + 1e-6, "native prefill exceeds total duration")
        values = {"input_tokens": metric(inp, "token", "server-response"),
                  "output_tokens": metric(out, "token", "server-response"),
                  "latency": metric(seconds, "second", "server-submit-to-finish"),
                  "decode_rate": metric(rate, "token/second", "server-prefill-finished-to-finish; numerator=output-minus-one"),
                  "drafted_tokens": metric(d, "token", "server-response"),
                  "accepted_tokens": metric(a, "token", "server-response"),
                  "decode_rounds": metric(rounds, "round", "server-response"),
                  "tokens_per_round": metric(out / rounds, "token/round", "server-response")}
        if d:
            values["acceptance"] = metric(a / d, "ratio", "accepted/drafted")
        for field, name, basis in (("prefill_seconds", "prefill", "server-prompt-pass"),
                                   ("time_to_first_token", "ttft", "server-arrival-to-first-token")):
            if runtime.get(field) is not None:
                values[name] = metric(runtime[field], "second", basis)
                recognized.add("tensorfold." + field)
        if wrapped:
            client = entry.get("client")
            require(isinstance(client, dict), "native streaming client telemetry required")
            ttft = number(client.get("client_ttft_seconds"))
            e2e = number(client.get("client_e2e_seconds"))
            events = number(client.get("stream_token_events"), integer=True, positive=True)
            intervals = client.get("itls_seconds")
            require(isinstance(intervals, list) and len(intervals) == events - 1 and events <= out,
                    "native streaming event count mismatch")
            intervals = [number(value) for value in intervals]
            require(e2e >= ttft and sum(intervals) <= e2e - ttft + 1e-6,
                    "native streaming timing order mismatch")
            values["client_ttft"] = metric(ttft, "second", "client-sse-monotonic")
            values["client_latency"] = metric(e2e, "second", "client-sse-monotonic")
            values["inter_token_latency"] = {"values": intervals, "unit": "second", "basis": "client-sse-monotonic-content-event"}
            if out > 1 and e2e > ttft:
                values["client_decode_rate"] = metric((out - 1) / (e2e - ttft), "token/second", "client-sse-monotonic")
                values["client_time_per_output_token"] = metric((e2e - ttft) / (out - 1), "second/token", "client-sse-monotonic")
            recognized.update(("client.client_ttft_seconds", "client.client_e2e_seconds", "client.itls_seconds", "client.stream_token_events"))
        rows.append({"request_id": rid, "metrics": values})
        recognized.update(("usage.prompt_tokens", "usage.completion_tokens", "usage.total_tokens", "tensorfold.seconds",
                           "tensorfold.engine", "tensorfold.enable_thinking", "tensorfold.reasoning_effort", "tensorfold.sampling", "tensorfold.drafts",
                           "tensorfold.tokens_per_second", "speculative.drafted", "speculative.accepted", "speculative.rounds",
                           "speculative.acceptance_rate", "speculative.tokens_per_round"))
    missing = ["decode_seconds", "wall_duration"]
    if not wrapped:
        missing.append("inter_token_latencies")
    missing += [key for key in ("tensorfold.prefill_seconds", "tensorfold.time_to_first_token") if key not in recognized]
    result = report("tensorfold-native-v1", meta, source, recognized,
                    missing, rows, {}, rejected)
    if settings is not None:
        result["identity"]["runtime"]["observed_settings_sha256"] = digest(repr(settings).encode())
    return result


ALIASES = {
    "prompt": ("prompt_tokens_total",), "generation": ("generation_tokens_total",),
    "drafted": ("mtp_drafted_total", "spec_decode_num_draft_tokens_total"),
    "accepted": ("mtp_accepted_total", "spec_decode_num_accepted_tokens_total"),
    "rounds": ("decode_rounds_total",), "disconnects": ("client_disconnections_total",),
    "requests": ("requests_total",),
    "latency": ("request_latency_seconds", "e2e_request_latency_seconds"),
    "ttft": ("time_to_first_token_seconds",),
    "decode": ("request_decode_seconds", "request_decode_time_seconds"),
    "prefill": ("request_prefill_seconds", "request_prefill_time_seconds"),
    "tpot": ("request_time_per_output_token_seconds",),
    "running": ("requests_running", "num_requests_running"),
    "waiting": ("requests_waiting", "num_requests_waiting"),
    "generation_running": ("generation_tokens_running",),
    "preemptions": ("preemptions_total",),
}
HISTOGRAMS = {"latency", "ttft", "decode", "prefill", "tpot"}
TIME_EDGES = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, math.inf)
TPOT_EDGES = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, math.inf)
SAMPLE = re.compile(r'^(tensorfold:[A-Za-z_][A-Za-z_0-9]*)(\{[^\n]*\})?\s+([^\s]+)$')
LABEL = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)="((?:[^"\\]|\\[\\"n])*)"')


def _labels(text):
    if text is None:
        return ()
    inner = text[1:-1]
    parts, pos = {}, 0
    for match in LABEL.finditer(inner):
        require(match.start() == pos and match.group(1) not in parts, "malformed or duplicate metric label")
        parts[match.group(1)] = match.group(2)
        pos = match.end()
        if pos < len(inner) and inner[pos] == ',':
            pos += 1
            require(pos < len(inner), "malformed metric labels")
    require(pos == len(inner), "malformed metric labels")
    return tuple(sorted(parts.items()))


def _snapshot(raw, policy):
    require(policy in ("strict-v1", "tensorfold-pinned-mirrors-v1"), "unsupported alias policy")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError:
        require(False, "malformed metrics encoding")
    families, recognized = {}, set()
    names = {name: semantic for semantic, group in ALIASES.items() for name in group}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        require(match is not None, "malformed Prometheus sample")
        full, labels, token = match.groups()
        name = full[len("tensorfold:"):]
        suffix = ""
        base = name
        for end in ("_bucket", "_sum", "_count"):
            if name.endswith(end) and name[:-len(end)] in names:
                base, suffix = name[:-len(end)], end
                break
        try:
            value = number(float(token))
        except ValueError:
            require(False, "invalid Prometheus numeric sample")
        parsed_labels = _labels(labels)
        if base not in names:
            continue
        semantic = names[base]
        require((semantic in HISTOGRAMS) == bool(suffix), "metric family type mismatch")
        key = (suffix, parsed_labels)
        samples = families.setdefault(base, {})
        require(key not in samples, "duplicate metric sample")
        samples[key] = value
        recognized.add(full)
    selected, aliases = {}, {}
    for semantic, group in ALIASES.items():
        present = [name for name in group if name in families]
        if semantic in {"preemptions", "requests"} and not present:
            continue
        require(present, "incomplete Prometheus snapshot")
        require(len(present) == 1 or policy == "tensorfold-pinned-mirrors-v1", "ambiguous metric aliases")
        chosen = present[0]
        samples = families[chosen]
        require(all(families[name] == samples for name in present), "metric alias mismatch or partial mirror")
        if semantic in HISTOGRAMS:
            sums = [v for (s, ls), v in samples.items() if s == "_sum" and not ls]
            counts = [v for (s, ls), v in samples.items() if s == "_count" and not ls]
            buckets = []
            for (s, ls), v in samples.items():
                if s == "_bucket":
                    require(len(ls) == 1 and ls[0][0] == "le", "unsupported histogram labels")
                    try:
                        edge = float(ls[0][1])
                    except ValueError:
                        require(False, "invalid histogram edge")
                    require(not math.isnan(edge) and edge > 0, "invalid histogram edge")
                    buckets.append((edge, number(v, integer=True)))
                else:
                    require(not ls, "unsupported histogram labels")
            require(len(sums) == len(counts) == 1 and buckets, "incomplete histogram")
            buckets.sort()
            require(tuple(edge for edge, _ in buckets) == (TPOT_EDGES if semantic == "tpot" else TIME_EDGES), "incomplete pinned histogram buckets")
            count = number(counts[0], integer=True)
            require(buckets[-1] == (math.inf, count) and all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(buckets, buckets[1:])), "inconsistent histogram buckets")
            selected[semantic] = {"sum": sums[0], "count": count, "samples": samples}
        elif semantic == "requests":
            require(all(s == "" and set(dict(ls)) == {"key", "status"} for s, ls in samples), "unsupported request labels")
            for (_, ls), value in samples.items():
                status = dict(ls)["status"]
                require(status.isdigit() and 100 <= int(status) <= 599, "invalid HTTP status label")
                number(value, integer=True)
            selected[semantic] = {"samples": samples}
        else:
            require(set(samples) == {("", ())}, "unsupported counter or gauge labels")
            selected[semantic] = {"value": number(samples[("", ())], integer=True), "samples": samples}
        aliases[semantic] = {"selected": "tensorfold:" + chosen,
                             "verified_mirrors": ["tensorfold:" + name for name in present[1:]]}
    return selected, aliases, recognized


def _prometheus(source, meta, before, client_capture=None):
    require(isinstance(before, bytes), "two complete snapshots required")
    interval = meta.get("interval")
    require(isinstance(interval, dict) and interval.get("isolated") is True and interval.get("drained") is True
            and interval.get("workload_id") == meta["workload"]["id"], "matched isolated drained interval required")
    duration = number(interval.get("duration_seconds"), positive=True)
    policy = meta.get("alias_policy", "strict-v1")
    start, a1, f1 = _snapshot(before, policy)
    end, a2, f2 = _snapshot(source, policy)
    require(a1 == a2, "metric alias selection changed")
    delta = {}
    for name in start:
        left, right = start[name]["samples"], end[name]["samples"]
        require(set(left) == set(right), "metric label set changed")
        if name in {"running", "waiting", "generation_running"}:
            require(start[name]["value"] == end[name]["value"] == 0, "interval has unfinished workload")
            continue
        require(all(right[k] >= left[k] for k in left), "counter reset in observation interval")
        if name in HISTOGRAMS:
            delta[name] = {k: end[name][k] - start[name][k] for k in ("sum", "count")}
            buckets = []
            for (suffix, labels), value in right.items():
                if suffix == "_bucket":
                    edge = float(dict(labels)["le"])
                    buckets.append((edge, value - left[(suffix, labels)]))
            buckets.sort()
            require(buckets[-1] == (math.inf, delta[name]["count"])
                    and all(a[1] <= b[1] for a, b in zip(buckets, buckets[1:])),
                    "inconsistent histogram interval buckets")
        elif name == "requests":
            success = failed = 0
            for (_, labels), value in right.items():
                diff = value - left[("", labels)]
                if 200 <= int(dict(labels)["status"]) < 300:
                    success += diff
                else:
                    failed += diff
            delta[name] = {"success": int(success), "failed": int(failed)}
        else:
            delta[name] = end[name]["value"] - start[name]["value"]
    n = delta["latency"]["count"]
    ids = request_ids(meta, n)
    if "requests" in delta:
        require(delta["requests"]["success"] == n and delta["requests"]["failed"] == 0,
                "interval mixes failed or unmatched HTTP requests")
    require(delta["disconnects"] == 0, "interval mixes disconnected requests")
    require(delta.get("preemptions", 0) == 0, "interval mixes preempted workloads")
    require(all(delta[k]["count"] == n for k in HISTOGRAMS), "incomplete matched histogram interval")
    d, a = delta["drafted"], delta["accepted"]
    require(a <= d and delta["generation"] >= n and delta["prompt"] >= n
            and delta["rounds"] > 0, "invalid matched token or draft counters")
    aggregates = {"aggregate_output_rate": metric(delta["generation"] / duration, "token/wall-second", "matched-isolated-interval"),
                  "wall_duration": metric(duration, "second", "matched-isolated-interval"),
                  "output_tokens": metric(delta["generation"], "token", "matched-finished-request-counters"),
                  "input_tokens": metric(delta["prompt"], "token", "matched-finished-request-counters"),
                  "drafted_tokens": metric(d, "token", "matched-finished-request-counters"),
                  "accepted_tokens": metric(a, "token", "matched-finished-request-counters"),
                  "decode_rounds": metric(delta["rounds"], "round", "matched-finished-request-counters")}
    if d:
        aggregates["acceptance"] = metric(a / d, "ratio", "accepted/drafted")
    for name in HISTOGRAMS:
        aggregates[name + "_mean"] = metric(delta[name]["sum"] / n, "second/token" if name == "tpot" else "second/request", "matched-finished-request-histogram")
    if delta["rounds"]:
        aggregates["decode_seconds_per_round"] = metric(delta["decode"]["sum"] / delta["rounds"], "request-second/round", "matched-finished-request-counters; includes overlap")
        aggregates["tokens_per_round"] = metric(delta["generation"] / delta["rounds"], "token/round", "matched-finished-request-counters")
    if delta["decode"]["sum"]:
        aggregates["pooled_decode_rate"] = metric(delta["generation"] / delta["decode"]["sum"], "token/request-second", "finished-token-count/summed-request-decode-time")
    result = report("tensorfold-prometheus-v1", meta, source, f1 | f2,
                    ["per_request_decode_rate", "inter_token_latencies"],
                    [{"request_id": rid, "metrics": {}} for rid in ids], aggregates)
    result.update(before_source_sha256=digest(before), alias_policy=policy, aliases=a2,
                  interval={"duration_seconds": duration, "isolated": True, "drained": True,
                            "identity_basis": "operator-declared"},
                  labels_sha256=digest(repr(sorted(start.get("requests", {}).get("samples", {}))).encode()))
    result["scope"]["failure_visibility"] = "http-status-counters" if "requests" in delta else "unavailable"
    if client_capture is not None:
        # Only an explicit versioned client capture can bind all interval responses.
        capture = strict_json(client_capture)
        require(isinstance(capture, dict) and capture.get("schema") == "spark-tensorfold-native-capture",
                "versioned client success capture required")
        client_meta = dict(meta, adapter="tensorfold-native-v1")
        client = _native(client_capture, client_meta)
        require(client["counts"] == {"accepted": n, "rejected": 0}, "client interval includes failed responses")
        for name, field in (("prompt", "input_tokens"), ("generation", "output_tokens"),
                            ("drafted", "drafted_tokens"), ("accepted", "accepted_tokens"), ("rounds", "decode_rounds")):
            require(sum(row["metrics"][field]["value"] for row in client["requests"]) == delta[name],
                    "client response counters do not match interval")
        result["client_capture_sha256"] = digest(client_capture)
        result["scope"]["failure_visibility"] = "client-response-proof"
    return result


def import_observations(adapter, source, metadata, *, before=None, client_capture=None):
    require(adapter in PINS and isinstance(source, bytes), "unsupported observation adapter")
    require(len(source) <= 32 * 1024 * 1024
            and (before is None or isinstance(before, bytes) and len(before) <= 32 * 1024 * 1024)
            and (client_capture is None or isinstance(client_capture, bytes) and len(client_capture) <= 32 * 1024 * 1024),
            "source exceeds size limit or has invalid byte format")
    require(adapter == "tensorfold-prometheus-v1" or before is client_capture is None,
            "snapshot companions require Prometheus adapter")
    identity(metadata, adapter)
    if adapter == "vllm-bench-serve-v1":
        return _vllm(source, metadata)
    if adapter == "tensorfold-native-v1":
        return _native(source, metadata)
    return _prometheus(source, metadata, before, client_capture)
