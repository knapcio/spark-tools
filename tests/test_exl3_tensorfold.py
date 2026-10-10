# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock

from lab.exl3.checkpoint import MARKERS, inspect_checkpoint
from lab.exl3.cli import main as exl3_main
from lab.tensorfold.probe import probe
from lab.tensorfold.source import _read_beneath, source_manifest


def write_checkpoint(root, *, bits=2, codebook="mcg", marker=None, config=True,
                     half_without_mul1=False, duplicate=False, truncate_trellis=False,
                     overlap=False, metadata=None, bad_bias=False, bad_dtype=False,
                     empty_tensor=False):
    root = Path(root)
    if config:
        (root / "config.json").write_text(json.dumps({
            "model_type": "fixture",
            "quantization_config": {"quant_method": "exl3", "bits": bits, "codebook": codebook},
        }))
    width = int(16 * bits)
    parts = [
        ("model.layers.0.mlp.down_proj.trellis", "I16", [8, 8, width], bytes(8 * 8 * width * 2)),
        ("model.layers.0.mlp.down_proj.suh", "F16", [128], bytes(256)),
        ("model.layers.0.mlp.down_proj.svh", "F16", [128], bytes(256)),
    ]
    chosen = None if half_without_mul1 else codebook
    if chosen in MARKERS:
        value = MARKERS[chosen] if marker is None else marker
        parts.append((f"model.layers.0.mlp.down_proj.{chosen}", "I32", [], struct.pack("<I", value)))
    if bad_bias:
        parts.append(("model.layers.0.mlp.down_proj.bias", "U8", [1], b"x"))
    if empty_tensor:
        parts.append(("model.empty", "F16", [128, 0], b""))
    offset = 0
    header, payload = {}, bytearray()
    for name, dtype, shape, data in parts:
        if truncate_trellis and name.endswith(".trellis"):
            data = data[:-2]
        if bad_dtype and name.endswith(".trellis"):
            dtype = []
        start = 0 if overlap and name.endswith(".suh") else offset
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, start + len(data)]}
        offset += len(data)
        payload.extend(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    if duplicate:
        (root / "model-00002-of-00002.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class Exl3CheckpointTests(unittest.TestCase):
    def test_valid_header_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory)
            report = inspect_checkpoint(directory)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["format"], "EXL3")
        self.assertEqual(report["exl3_tensors"], 1)
        self.assertEqual(report["codebooks"], {"mcg": 1})
        self.assertEqual(report["weight_share_by_bits"], {"2": 1.0})
        self.assertEqual(report["average_trellis_bits"], 2.0)

    def test_half_width_requires_mul1(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, bits=2.5, codebook="3inst", half_without_mul1=True)
            with self.assertRaisesRegex(ValueError, "no valid EXL3"):
                inspect_checkpoint(directory)

    def test_bad_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, marker=0)
            with self.assertRaisesRegex(ValueError, "no valid EXL3"):
                inspect_checkpoint(directory)

    def test_config_and_duplicates_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, config=False)
            with self.assertRaisesRegex(ValueError, "quant_method=exl3"):
                inspect_checkpoint(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate tensor"):
                inspect_checkpoint(directory)

    def test_truncated_or_excessive_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            (root / "bad.safetensors").write_bytes(struct.pack("<Q", 999999) + b"{}")
            with self.assertRaisesRegex(ValueError, "invalid or excessive"):
                inspect_checkpoint(root)

    def test_truncated_trellis_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, truncate_trellis=True)
            with self.assertRaisesRegex(ValueError, "payload length does not match"):
                inspect_checkpoint(directory)

    def test_overlapping_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, overlap=True)
            with self.assertRaisesRegex(ValueError, "overlapping tensor payload"):
                inspect_checkpoint(directory)

    def test_cli_does_not_disclose_path_from_os_error(self):
        output = io.StringIO()
        with mock.patch("lab.exl3.cli.inspect_checkpoint",
                        side_effect=PermissionError("/private/example/model/config.json")):
            with redirect_stdout(output):
                self.assertEqual(exl3_main(["inspect", "/private/example/model"]), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["errors"], ["checkpoint could not be read"])
        self.assertNotIn("/private/example", output.getvalue())

    def test_invalid_metadata_and_bias_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, metadata=[])
            with self.assertRaisesRegex(ValueError, "metadata must map strings to strings"):
                inspect_checkpoint(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, metadata={"format": False})
            with self.assertRaisesRegex(ValueError, "metadata must map strings to strings"):
                inspect_checkpoint(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, bad_bias=True)
            with self.assertRaisesRegex(ValueError, "no valid EXL3"):
                inspect_checkpoint(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, bad_dtype=True)
            with self.assertRaisesRegex(ValueError, "invalid dtype/shape"):
                inspect_checkpoint(directory)

    def test_valid_zero_sized_tensor_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, empty_tensor=True)
            report = inspect_checkpoint(directory)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["plain_tensors"], 1)

    def test_shard_replacement_with_symlink_or_fifo_fails_closed(self):
        from lab.exl3 import checkpoint as checkpoint_module

        for replacement in ("symlink", "fifo"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "model"
                root.mkdir()
                write_checkpoint(root)
                shard = root / "model-00001-of-00001.safetensors"
                original_shards = checkpoint_module._shards

                def swap_after_inventory(descriptor):
                    rows = original_shards(descriptor)
                    if replacement == "symlink":
                        outside = Path(directory) / "outside.safetensors"
                        shard.replace(outside)
                        shard.symlink_to(outside)
                    else:
                        shard.unlink()
                        os.mkfifo(shard)
                    return rows

                with mock.patch("lab.exl3.checkpoint._shards", swap_after_inventory):
                    with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                        inspect_checkpoint(root)

    def test_symlinked_model_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real"
            real.mkdir()
            write_checkpoint(real)
            linked = Path(directory) / "linked"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "non-symlink directory"):
                inspect_checkpoint(linked)


class TensorFoldSourceTests(unittest.TestCase):
    def tree(self, root):
        package = Path(root) / "python/tensorfold"
        (package / "cuda/exl3").mkdir(parents=True)
        (package / "server").mkdir()
        (package / "__init__.py").write_text("")
        (package / "cuda/build.py").write_text("def build(): pass\n")
        (package / "cuda/exl3/format.py").write_text("FORMAT = 'EXL3'\n")
        (package / "cuda/exl3/linear.cu").write_text("__global__ void linear() {}\n")
        (package / "server/scheduler.py").write_text("class Scheduler: pass\n")
        return package

    def test_manifest_finds_exl3_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tree(directory)
            first = source_manifest(directory, require_exl3=True)
            second = source_manifest(directory, require_exl3=True)
        self.assertEqual(first["status"], "PASS")
        self.assertTrue(first["components"]["exl3"])
        self.assertTrue(first["components"]["scheduler"])
        self.assertEqual(first["tree_sha256"], second["tree_sha256"])
        self.assertEqual(first["exl3_files"], 2)

    def test_manifest_rejects_missing_exl3_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.tree(directory)
            for path in (package / "cuda/exl3").iterdir():
                path.unlink()
            (package / "cuda/exl3").rmdir()
            report = source_manifest(directory, require_exl3=True)
            self.assertEqual(report["status"], "FAIL")
        with tempfile.TemporaryDirectory() as directory:
            package = self.tree(directory)
            (package / "linked.py").symlink_to(package / "__init__.py")
            with self.assertRaisesRegex(ValueError, "symlinks are not allowed"):
                source_manifest(directory)

    def test_manifest_rejects_symlinked_package_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            self.tree(outside)
            requested = root / "requested"
            (requested / "python").mkdir(parents=True)
            (requested / "python/tensorfold").symlink_to(outside / "python/tensorfold")
            with self.assertRaisesRegex(ValueError, "package root not found"):
                source_manifest(requested)

    def test_manifest_rejects_symlinked_intermediate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            self.tree(outside)
            requested = root / "requested"
            requested.mkdir()
            (requested / "python").symlink_to(outside / "python")
            with self.assertRaisesRegex(ValueError, "package root not found"):
                source_manifest(requested)

    def test_manifest_rejects_symlinked_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            self.tree(real)
            linked = root / "linked"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "source root"):
                source_manifest(linked)

    def test_manifest_keeps_open_root_when_path_is_swapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = root / "requested"
            self.tree(requested)
            expected = source_manifest(requested, require_exl3=True)["tree_sha256"]
            outside = root / "outside"
            self.tree(outside)
            (outside / "python/tensorfold/server/scheduler.py").unlink()
            (outside / "python/tensorfold/cuda/exl3/linear.cu").write_text(
                "__global__ void outside_secret() {}\n")
            parked = root / "parked"
            from lab.tensorfold import source as source_module
            original = source_module._package_prefix

            def swap_after_discovery(descriptor):
                result = original(descriptor)
                requested.rename(parked)
                requested.symlink_to(outside)
                return result

            with mock.patch("lab.tensorfold.source._package_prefix", swap_after_discovery):
                report = source_manifest(requested, require_exl3=True)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["tree_sha256"], expected)

    def test_descriptor_read_rejects_file_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.tree(root)
            outside = root / "outside.cu"
            outside.write_text("__global__ void outside() {}\n")
            linked = package / "cuda/linked.cu"
            linked.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "read safely"):
                _read_beneath(package, linked.relative_to(package))

    def test_regular_file_to_fifo_race_fails_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.tree(directory)
            target = package / "kernel.cu"
            target.write_text("__global__ void kernel() {}\n")
            original_open = os.open
            swapped = False

            def swap_before_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "kernel.cu" and kwargs.get("dir_fd") is not None and not swapped:
                    target.unlink()
                    os.mkfifo(target)
                    swapped = True
                return original_open(path, flags, *args, **kwargs)

            with mock.patch("lab.tensorfold.source.os.open", side_effect=swap_before_open):
                with self.assertRaisesRegex(ValueError, "regular"):
                    _read_beneath(package, Path("kernel.cu"))

    def test_walk_error_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tree(directory)

            with mock.patch("lab.tensorfold.source.os.scandir", side_effect=PermissionError("fixture")):
                with self.assertRaisesRegex(ValueError, "traversed completely"):
                    source_manifest(directory)


class Handler(BaseHTTPRequestHandler):
    mismatch = False

    def log_message(self, *args):
        pass

    def _send(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "fixture", "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        content = "DIFFERENT" if self.mismatch and body.get("draft") else "SPARK_TOOLS_OK"
        self._send({
            "choices": [{"message": {"content": content, "reasoning_content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        })


class TensorFoldProbeTests(unittest.TestCase):
    def serve(self, mismatch=False):
        handler = type("ProbeHandler", (Handler,), {"mismatch": mismatch})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_probe_and_draft_serial_equality(self):
        report = probe(self.serve(), compare_draft=True, expect_text="SPARK_TOOLS_OK")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["draft_serial_equal"])
        self.assertTrue(report["expected_text_equal"])
        self.assertEqual([row["draft"] for row in report["arms"]], [False, True])
        self.assertNotIn("SPARK_TOOLS_OK", json.dumps(report))

    def test_probe_fails_on_draft_mismatch(self):
        report = probe(self.serve(mismatch=True), compare_draft=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["draft_serial_equal"])

    def test_probe_rejects_credentials_and_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "embedded credentials"):
            probe("http://user:pass@example.invalid")
        with self.assertRaisesRegex(ValueError, "select a model"):
            probe(self.serve(), model="missing")

    def test_transport_error_does_not_disclose_endpoint(self):
        endpoint = "http://127.0.0.1:1/private-name"
        with self.assertRaises(ValueError) as caught:
            probe(endpoint, timeout=0.1)
        self.assertEqual(str(caught.exception), "endpoint request failed")
        self.assertNotIn("private-name", str(caught.exception))

    def test_redirect_is_not_followed(self):
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "http://example.invalid/collect")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with self.assertRaisesRegex(ValueError, "endpoint request failed"):
            probe(f"http://127.0.0.1:{server.server_port}")

    def test_token_control_characters_are_rejected_without_disclosure(self):
        token = "synthetic_secret\ntrailing"
        with self.assertRaisesRegex(ValueError, "unsupported characters") as caught:
            probe(self.serve(), token=token)
        self.assertNotIn("synthetic_secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
