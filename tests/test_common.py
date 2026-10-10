# SPDX-License-Identifier: Apache-2.0
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import pytest
from common.config import digest, finite, integer, loads, validate_survival
from common.endpoint import HTTP
from common.stats import interval_union, paired_bootstrap, quantile


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        loads(text)


@pytest.mark.parametrize("value", [True, "2", math.inf, math.nan, -1])
def test_finite_rejects_invalid(value):
    with pytest.raises(ValueError):
        finite(value, low=0)


def test_integer_and_digest():
    with pytest.raises(ValueError):
        integer(1.0)
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


@pytest.mark.parametrize("values", [[], [.2, .3], [1.1], [math.nan]])
def test_survival_invalid(values):
    with pytest.raises(ValueError):
        validate_survival(values)


def test_statistics():
    assert quantile([1, 3], .25) == 1.5
    assert interval_union([(0, 2), (1, 3), (4, 5)]) == 4
    assert paired_bootstrap([2, 2, 2])["ci95"] == [2, 2]
    assert paired_bootstrap([2, 3], clusters=["a", "a"])["ci95"] is None


def test_http_refuses_redirect_without_forwarding_bearer_token():
    received = []

    class Collector(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            received.append(self.headers.get('Authorization'))
            self.send_response(204)
            self.end_headers()

    collector = ThreadingHTTPServer(('127.0.0.1', 0), Collector)

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.send_response(302)
            self.send_header('Location', f'http://localhost:{collector.server_port}/collect')
            self.end_headers()

    redirect = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (collector, redirect)]
    for thread in threads:
        thread.start()
    try:
        transport = HTTP(f'http://127.0.0.1:{redirect.server_port}', 'synthetic_review_token')
        with pytest.raises(OSError, match='endpoint request failed'):
            transport.ask({'model': 'fixture'}, 2)
        assert received == []
    finally:
        redirect.shutdown()
        collector.shutdown()
        redirect.server_close()
        collector.server_close()


def test_http_rejects_malformed_token_without_disclosure():
    token = 'synthetic_review_secret\ntrailing'
    with pytest.raises(ValueError, match='unsupported characters') as caught:
        HTTP('http://127.0.0.1:8000', token)
    assert 'synthetic_review_secret' not in str(caught.value)
