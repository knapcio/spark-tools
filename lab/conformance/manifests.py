# SPDX-License-Identifier: Apache-2.0
"""Public upstream identities and external runner recipes; never auto-executed."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import subprocess

from . import SCOPE

UPSTREAMS = {
    "tensorfold": {
        "repository": "https://github.com/ashhart/TensorFold",
        "revision": "db281878ddb836fd0df510d8771ecb7e0fe47d26",
        "license": "Apache-2.0; EXL3 format credit Turboderp MIT",
        "files": {
            "tests/cuda/test_exl3_linear.py": "70871553d8fcf3f10e40cf1e015f50b3070f8b668dcc37c8108ffeff93df2dc5",
            "src/tensorfold/cuda/exl3/format.py": "25b617d65976ce9bec016f88086ed0565bd2ed56f223f9f490ecab73048145f7",
            "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        },
    },
    "gr8": {
        "repository": "https://github.com/kindlingai/gr8",
        "revision": "d40ab1d8e010e125890b9220f3cf1cd80a4a8e06",
        "license": "MIT, Copyright (c) 2026 g6 engine contributors",
        "files": {
            "engine/loader-rs/tests/make_exlr8_fixture.py": "a3185f22547b49c125691ed4620b7e6c39a1268346d1a0b61d025b1772194155",
            "engine/loader-rs/tests/exlr8_fixture.rs": "7b1b371383c3c674b47380bc8a426dd7b8887e40feb7155755f71a0546ec5bd6",
            "engine/loader-rs/Cargo.toml": "9f55ad8b90cf08e245df45dad1e664c0b5370f7ee24ffab012b2dc7123037038",
            "LICENSE": "04e824320814e8b48da067bdcf816490b64d66556a9a1c681d0c01ba51203cbf",
        },
    },
    "exlr8": {
        "repository": "https://github.com/kindlingai/exlr8",
        "revision": "d56e3ecf7aa3104be0b77b59b076406562fd20c5",
        "license": "MIT, Copyright (c) 2025 Turboderp (loader/LICENSE)",
        "files": {
            "loader/exlr8.py": "44f51ad076cb6b986b24b2861072fb7529f850c212a00bd8b977dd6965022a98",
            "loader/LICENSE": "93ccc2f4c97c03bc767c611c75f3bb16b97e49606f5cd3ca05d7f6201ff8f8de",
        },
    },
}
RUNNERS = ("tensorfold-cuda", "gr8-dense")


def runner_manifest(runner):
    if runner not in RUNNERS:
        raise ValueError("unknown external runner")
    result = {"schema": 1, "scope": SCOPE, "status": "PLAN_ONLY", "runner": runner,
              "executes": False, "source_validation": "check-sources before execution",
              "claims_excluded": ["model", "quality", "collective", "whole-engine", "performance"],
              "record": ["host", "UTC and Europe/Warsaw time", "image digest or environment identity",
                         "source revisions and file hashes", "commands and return codes", "stdout/stderr",
                         "fixture SHA256 before/after", "test counts and all skips"]}
    if runner == "tensorfold-cuda":
        result.update({
            "upstreams": {"tensorfold": deepcopy(UPSTREAMS["tensorfold"])},
            "requires": ["isolated pinned checkout", "Python >=3.11", "NumPy", "pytest", "CUDA PyTorch",
                         "working GPU", "TensorFold CUDA toolchain/dependencies"],
            "max_seconds": 600,
            "commands": [
                {"cwd": "{tensorfold}", "argv": ["{python}", "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"], "max_seconds": 30},
                {"cwd": "{tensorfold}", "argv": ["{python}", "-m", "pytest", "tests/cuda/test_exl3_linear.py", "-ra", "--junitxml={results}/tensorfold.xml"], "max_seconds": 570},
            ],
            "exact_decoder": {"tests": ["test_decode_is_bit_exact"], "comparison": "fp16 bit patterns; zero tolerance",
                              "codebooks": ["3inst", "mcg", "mul1"]},
            "toleranced_matmul": {"tests": ["test_matches_the_float64_reference"],
                                  "metric": "relative L2 error", "threshold": 0.002},
            "other_checks": ["row invariance", "stored/strips layout", "dtype", "bias", "split-K"],
            "acceptance": "Require CUDA preflight success, nonzero executed tests and zero skips/failures/errors; report exact decoder and toleranced matmul separately",
        })
    else:
        result.update({
            "upstreams": {name: deepcopy(UPSTREAMS[name]) for name in ("gr8", "exlr8")},
            "requires": ["disposable pinned gr8 and exlr8 checkouts", "PyTorch CPU with float8_e4m3fn", "Rust supporting slice::as_chunks (>=1.88)", "Cargo"],
            "max_seconds": 600,
            "commands": [
                {"cwd": "{gr8}", "argv": ["{python}", "engine/loader-rs/tests/make_exlr8_fixture.py", "{exlr8}", "{gr8}/engine/loader-rs/tests/data/exlr8-fixture.safetensors"], "max_seconds": 300},
                {"cwd": "{gr8}", "argv": ["cargo", "test", "--manifest-path", "engine/loader-rs/Cargo.toml", "--target-dir", "{results}/target", "--test", "exlr8_fixture", "--", "--nocapture"], "max_seconds": 300},
            ],
            "exact_decoder": {"comparison": "FP32 LE bytes / 64-bit FNV-1a; full-byte equality for small tensors",
                              "matrices": 11, "tests": 4,
                              "formats": ["TRELLIS", "INT4", "MXFP8", "FP8_BLOCK", "FP8_ROW", "BF16", "FP32"],
                              "trellis_cases": ["K4 out 256x256", "K4 in 256x256", "K3 out 128x128", "K2 in 128x128"]},
            "toleranced_matmul": None,
            "acceptance": "Require a fixture hash record, 4 passed Rust tests, zero failures/ignored tests; this covers dense loader functions only",
            "writes": ["synthetic fixture inside disposable gr8 checkout", "Cargo output under {results}/target"],
        })
    return result


def check_source(name, source):
    """Read-only HEAD/clean tracked-tree/file pin check; no code is imported."""
    if name not in UPSTREAMS:
        raise ValueError("unknown upstream")
    supplied = Path(source)
    if supplied.is_symlink():
        raise ValueError("source must be a non-symlink directory")
    try:
        root = supplied.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("source must be a non-symlink directory") from error
    if not root.is_dir():
        raise ValueError("source must be a non-symlink directory")
    pin = UPSTREAMS[name]
    errors = []
    for args in (("rev-parse", "HEAD"), ("diff", "HEAD", "--name-only")):
        process = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                 text=True, timeout=15, check=False,
                                 env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        if process.returncode:
            raise ValueError("git source identity could not be checked")
        if args[0] == "rev-parse":
            revision = process.stdout.strip()
            if revision != pin["revision"]:
                errors.append("HEAD differs from pinned revision")
        elif process.stdout.strip():
            errors.append("tracked source differs from HEAD")
    files = []
    for relative, expected in pin["files"].items():
        path = root
        for component in Path(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("symlinks are forbidden in pinned source paths")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ValueError("pinned source file must be regular and <=1 MiB")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": actual, "matches": actual == expected})
        if actual != expected:
            errors.append(f"file pin mismatch: {relative}")
    return {"schema": 1, "scope": SCOPE, "status": "PASS" if not errors else "FAIL",
            "upstream": name, "revision": revision, "files": files, "errors": errors,
            "execution_verified": False, "untracked_files_checked": False}
