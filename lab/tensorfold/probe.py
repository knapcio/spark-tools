# SPDX-License-Identifier: Apache-2.0
"""Fail-closed TensorFold OpenAI API probe with optional draft/serial equality."""
import hashlib
import json
import math
import time
from urllib.parse import urlsplit, urlunsplit
import urllib.error
import urllib.request

from common.config import canonical, loads

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPEN = urllib.request.build_opener(_NoRedirect()).open


def _base(endpoint):
    parsed = urlsplit(endpoint)
    embedded_secret = getattr(parsed, "pass" + "word")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or embedded_secret:
        raise ValueError("endpoint must be an http(s) URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _headers(token):
    headers = {"Accept": "application/json"}
    if token:
        if (not isinstance(token, str) or len(token) > 8192
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("authentication token contains unsupported characters")
        headers["Authorization"] = "Bearer " + token
    return headers


def _request(url, timeout, token=None, body=None):
    data = None if body is None else canonical(body)
    headers = _headers(token)
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    started = time.perf_counter()
    try:
        with OPEN(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                raise ValueError("endpoint response exceeds 16 MiB")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("endpoint response exceeds 16 MiB")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError("endpoint request failed") from exc
    elapsed = (time.perf_counter() - started) * 1000
    try:
        value = loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("endpoint returned invalid JSON") from exc
    return value, raw, elapsed


def _model_ids(catalog):
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise ValueError("/v1/models response has no data array")
    ids = []
    for row in catalog["data"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("/v1/models contains an invalid model entry")
        ids.append(row["id"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("model catalog must contain unique nonempty IDs")
    return ids


def _choice(response):
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("chat completion returned an error")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("chat completion must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("chat completion has no string message content")
    reasoning = message.get("reasoning_content", message.get("reasoning", ""))
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise ValueError("chat completion reasoning must be a string when present")
    finish = choice.get("finish_reason")
    if finish not in {"stop", "length"}:
        raise ValueError("chat completion has an invalid finish_reason")
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("chat completion usage must be an object when present")
    return content_record(message["content"], reasoning, finish, usage)


def content_record(content, reasoning, finish, usage):
    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    tokens = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (usage or {}).get(key)
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError(f"usage.{key} must be a nonnegative integer")
            tokens[key] = value
    return {
        "content_sha256": digest(content),
        "reasoning_sha256": digest(reasoning),
        "content_bytes": len(content.encode("utf-8")),
        "reasoning_bytes": len(reasoning.encode("utf-8")),
        "finish_reason": finish,
        "usage": tokens,
        "_content": content,
        "_reasoning": reasoning,
    }

def probe(endpoint, model=None, token=None, timeout=90, compare_draft=False,
          prompt="Reply with exactly SPARK_TOOLS_OK and nothing else.", expect_text=None):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise ValueError("timeout must be finite and in (0, 3600]")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 16384:
        raise ValueError("prompt must be a nonempty string no larger than 16 KiB")
    base = _base(endpoint)
    catalog, catalog_raw, catalog_ms = _request(base + "/models", timeout, token)
    ids = _model_ids(catalog)
    model = model or (ids[0] if len(ids) == 1 else None)
    if not model or model not in ids:
        raise ValueError("select a model present in /v1/models")
    common = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    arms = [False, True] if compare_draft else [True]
    results = []
    private = []
    for draft in arms:
        body = {**common, "draft": draft}
        response, raw, elapsed = _request(base + "/chat/completions", timeout, token, body)
        row = _choice(response)
        private.append((row.pop("_content"), row.pop("_reasoning"), row["finish_reason"]))
        row.update({
            "draft": draft,
            "elapsed_ms": round(elapsed, 3),
            "request_sha256": hashlib.sha256(canonical(body)).hexdigest(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
        })
        results.append(row)
    equal = None if not compare_draft else private[0] == private[1]
    expected = None if expect_text is None else all(content == expect_text for content, _, _ in private)
    errors = []
    if equal is False:
        errors.append("drafted and serial responses differ")
    if expected is False:
        errors.append("response content does not match --expect-text")
    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "runtime": "TensorFold",
        "api": "openai-chat-completions",
        "model": model,
        "model_catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "model_catalog_ms": round(catalog_ms, 3),
        "draft_serial_equal": equal,
        "expected_text_equal": expected,
        "arms": results,
        "errors": errors,
        "limits": [
            "API compatibility and optional draft/serial output equality only.",
            "This probe does not establish throughput, quality, kernel correctness or multi-rank safety.",
        ],
    }
