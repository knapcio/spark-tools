# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Portable HTTP transport extracted from the existing quality panel."""
import json
import socket
import threading
from urllib.parse import urlsplit
import urllib.error
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPEN = urllib.request.build_opener(_NoRedirect()).open

class HTTP:
    def __init__(self, base, token=None):
        parsed = urlsplit(base)
        embedded_secret = getattr(parsed, 'pass' + 'word')
        if (parsed.scheme not in {'http', 'https'} or not parsed.netloc
                or parsed.username or embedded_secret or parsed.query or parsed.fragment):
            raise ValueError('endpoint must be an http(s) URL without embedded credentials, query or fragment')
        if token:
            if (not isinstance(token, str) or len(token) > 8192
                    or any(not 33 <= ord(character) <= 126 for character in token)):
                raise ValueError('authentication token contains unsupported characters')
        self.token = token
        self.base = base.rstrip('/');self.lock = threading.Lock();self.responses = set()
    def open(self, body, timeout):
        endpoint = self.base + ('/chat/completions' if self.base.endswith('/v1') else '/v1/chat/completions')
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + self.token} if self.token else {})})
        try:
            r = OPEN(req, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OSError('endpoint request failed') from exc
        with self.lock:self.responses.add(r)
        return r
    def release(self, response):
        with self.lock:self.responses.discard(response)
        response.close()
    def cancel(self):
        with self.lock:responses = list(self.responses)
        for r in responses:
            try:r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):pass
            try:r.close()
            except OSError:pass
    def metrics(self):
        try:
            with OPEN(self.base + '/metrics', timeout=3) as r:
                return r.read().decode()
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OSError('endpoint metrics request failed') from exc
    def ask(self, body, timeout):
        r = self.open(body, timeout)
        try:return r.read()
        finally:self.release(r)

