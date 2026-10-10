# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
import urllib.error
import urllib.request
import unittest
from unittest import mock

from lab.exlr8.cli import main as exlr8_main
from lab.exlr8.remote import EXLR8_CODEBOOK, HfClient, _SafeRedirect, inspect_hf


class FakeClient:
    def __init__(self, tamper=False, missing_sidecar=False, incomplete_ranks=False,
                 bad_index_header=False, invalid_tensor=False, excessive_tp=False,
                 object_codebook=False):
        self.downloaded_bytes = 1234
        self.commit = "a" * 40
        self.missing_sidecar = missing_sidecar
        self.incomplete_ranks = incomplete_ranks
        self.bad_index_header = bad_index_header
        self.invalid_tensor = invalid_tensor
        self.excessive_tp = excessive_tp
        self.object_codebook = object_codebook
        self.weight_rows = {
            "dense/exlr8-dense-003.safetensors": (280, "1" * 64),
            "experts/exlr8-layer-003.index.safetensors": (280, "2" * 64),
            "experts/exlr8-layer-003.k2.safetensors": (288, "3" * 64),
        }
        identity = hashlib.sha256()
        for path, (size, digest) in sorted(self.weight_rows.items()):
            identity.update(f"{path} {size} {digest}\n".encode())
        self.set_id = identity.hexdigest()
        if tamper:
            self.set_id = "f" * 64
        self.manifest = {
            "format": "exlr8-v2-manifest", "version": "fixture", "name": "small",
            "set_id": self.set_id, "mtp_layer": "none",
            "layers": {"3": {"gate": [[2]], "up": [[2]], "down": [2]}},
        }

    def repo_info(self, repo, revision):
        siblings = []
        for path in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "manifests/exlr8-manifest-small.json"):
            siblings.append({"rfilename": path, "size": 10})
        for path, (size, digest) in self.weight_rows.items():
            siblings.append({"rfilename": path, "size": size, "lfs": {"sha256": digest}})
            if path.startswith("dense/"):
                sidecar = "hashes/" + path + ".sha256"
            else:
                sidecar = "hashes/" + path.split("/", 1)[1] + ".sha256"
            if not self.missing_sidecar or not path.endswith(".k2.safetensors"):
                siblings.append({"rfilename": sidecar, "size": 65})
        tp = 1_000_000_000 if self.excessive_tp else 2
        for rank in range(1 if self.incomplete_ranks or self.excessive_tp else 2):
            path = f"index-slices/tp{tp}/k2/rank{rank}.safetensors"
            siblings.append({"rfilename": path, "size": 272,
                             "lfs": {"sha256": str(4 + rank) * 64}})
            siblings.append({"rfilename": "hashes/" + path + ".sha256", "size": 65})
        return {"sha": self.commit, "siblings": siblings}

    def json_file(self, repo, commit, path):
        return self.manifest

    def header(self, repo, commit, path, file_size):
        def entry(dtype, shape, start, end):
            if self.invalid_tensor:
                return {"dtype": "BOGUS", "shape": [-1], "data_offsets": [-100, 10**30]}
            return {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}

        if path.endswith(".k2.safetensors"):
            metadata = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "contents": "trellis",
                        "layer": "3", "k": "2"}
            tensors = {name: entry("U8", [8], index * 8, (index + 1) * 8)
                       for index, name in enumerate(("trellis.gate", "trellis.up", "trellis.down"))}
        elif ".index." in path:
            codebook = EXLR8_CODEBOOK
            if self.bad_index_header:
                codebook = "mcg 0xCBAC1FED unrounded FP32; incompatible with E4M3"
            if self.object_codebook:
                codebook = {"E4M3": False, "round": False, "0xCBAC1FED": False}
            metadata = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "contents": "index",
                        "layer": "3", "codebook": codebook}
            tensors = {"k2.index": entry("I64", [2], 0, 16)}
        elif path.startswith("dense/"):
            metadata = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "contents": "dense"}
            tensors = {"x.record": entry("I64", [2], 0, 16)}
        else:
            rank = path.rsplit("rank", 1)[1].split(".", 1)[0]
            tp = "1000000000" if self.excessive_tp else "2"
            metadata = {"format": "exlr8-v2-index-slice", "tp": tp, "rank": rank, "k": "2",
                        "set_id": self.set_id}
            tensors = {"L003.offset": entry("I64", [1], 0, 8)}
        return metadata, tensors, 256


class Exlr8RemoteTests(unittest.TestCase):
    def test_metadata_only_release_passes(self):
        client = FakeClient()
        report = inspect_hf("owner/model", client=client)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["set_id"], report["computed_set_id"])
        self.assertEqual(report["weight_files"], 3)
        self.assertEqual(report["downloaded_metadata_bytes"], 1234)
        self.assertTrue(report["header_samples"])

    def test_set_id_tampering_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(tamper=True), header_samples=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("manifest set_id does not match LFS weight metadata", report["errors"])

    def test_missing_hash_sidecar_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(missing_sidecar=True), header_samples=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("missing hash sidecar" in error for error in report["errors"]))

    def test_incomplete_tp_rank_set_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(incomplete_ranks=True), header_samples=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("do not cover ranks" in error for error in report["errors"]))

    def test_unrounded_index_header_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(bad_index_header=True))
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("rounded E4M3" in error for error in report["errors"]))

    def test_non_string_codebook_declaration_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(object_codebook=True), header_samples=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("metadata must map strings to strings" in error
                            for error in report["errors"]))

    def test_invalid_safetensors_metadata_fails_closed(self):
        client = FakeClient()
        original = client.header

        def invalid_header(*args, **kwargs):
            metadata, tensors, size = original(*args, **kwargs)
            metadata["extra"] = {"not": "a string"}
            return metadata, tensors, size

        client.header = invalid_header
        report = inspect_hf("owner/model", client=client)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("metadata must map strings to strings" in error
                            for error in report["errors"]))

    def test_invalid_tensor_layout_fails_closed(self):
        report = inspect_hf("owner/model", client=FakeClient(invalid_tensor=True))
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("unsupported dtype" in error for error in report["errors"]))

    def test_valid_zero_sized_tensor_is_accepted(self):
        client = FakeClient()
        original = client.header

        def header_with_empty_tensor(*args, **kwargs):
            metadata, tensors, size = original(*args, **kwargs)
            payload_bytes = args[3] - 8 - size
            tensors["valid.empty"] = {
                "dtype": "F16", "shape": [128, 0],
                "data_offsets": [payload_bytes, payload_bytes],
            }
            return metadata, tensors, size

        client.header = header_with_empty_tensor
        report = inspect_hf("owner/model", client=client)
        self.assertEqual(report["status"], "PASS")

    def test_excessive_tp_is_rejected_before_rank_set_allocation(self):
        report = inspect_hf("owner/model", client=FakeClient(excessive_tp=True))
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("TP/rank bounds" in error for error in report["errors"]))

    def test_zero_requested_samples_still_checks_every_index_header(self):
        report = inspect_hf("owner/model", client=FakeClient(bad_index_header=True), header_samples=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("rounded E4M3" in error for error in report["errors"]))

    def test_repo_revision_and_sample_bounds(self):
        with self.assertRaisesRegex(ValueError, "owner/name"):
            inspect_hf("https://example.invalid/model", client=FakeClient())
        with self.assertRaisesRegex(ValueError, "revision"):
            inspect_hf("owner/model", revision="../private", client=FakeClient())
        with self.assertRaisesRegex(ValueError, "header_samples"):
            inspect_hf("owner/model", header_samples=33, client=FakeClient())

    def test_bounded_range_rejects_truncated_body_and_wrong_total(self):
        class Response:
            status = 206

            def __init__(self, body, content_range):
                self.body = body
                self.headers = {"Content-Range": content_range}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return self.body[:limit]

        class Opener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        client = HfClient()
        client.opener = Opener(Response(b"x" * 40, "bytes 8-1031/9999"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            client._read("https://huggingface.co/example", 1024, require_range=(8, 1031, 9999))
        client.opener = Opener(Response(b"x" * 1024, "bytes 8-1031/10000"))
        with self.assertRaisesRegex(ValueError, "honor"):
            client._read("https://huggingface.co/example", 1024, require_range=(8, 1031, 9999))

    def test_cli_rejects_control_characters_without_disclosing_token(self):
        token = "synthetic_secret\ntrailing"
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"SPARK_REVIEW_TOKEN": token}):
            with redirect_stdout(output):
                result = exlr8_main([
                    "inspect-hf", "owner/model", "--token-env", "SPARK_REVIEW_TOKEN",
                ])
        self.assertEqual(result, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["errors"], ["Hugging Face token contains unsupported characters"])
        self.assertNotIn("synthetic_secret", output.getvalue())

    def test_redirect_rejects_same_host_on_non_https_port(self):
        request = urllib.request.Request(
            "https://huggingface.co/owner/model",
            headers={"Authorization": "Bearer synthetic_review_token"},
        )
        with self.assertRaises(urllib.error.URLError):
            _SafeRedirect().redirect_request(
                request, None, 302, "Found", {},
                "https://huggingface.co:8443/collect",
            )


if __name__ == "__main__":
    unittest.main()
