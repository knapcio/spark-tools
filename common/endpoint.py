# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Portable HTTP transport extracted from the existing quality panel."""
import json
import socket
import threading
import urllib.request

OPEN = urllib.request.urlopen

class HTTP:
    def __init__(self, base, token=None):
        self.token = token
        self.base = base.rstrip('/');self.lock = threading.Lock();self.responses = set()
    def open(self, body, timeout):
        endpoint = self.base + ('/chat/completions' if self.base.endswith('/v1') else '/v1/chat/completions')
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + self.token} if self.token else {})})
        r = OPEN(req, timeout=timeout)
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
        with OPEN(self.base + '/metrics', timeout=3) as r:
            return r.read().decode()
    def ask(self, body, timeout):
        r = self.open(body, timeout)
        try:return r.read()
        finally:self.release(r)

