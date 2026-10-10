# SPDX-License-Identifier: Apache-2.0
"""Reproducible inventory of a TensorFold source or installed package tree."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat

from common.config import canonical

SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py", ".zig"}
MAX_FILES = 20000
MAX_DIRECTORIES = 20000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024


@contextmanager
def _source_root(source):
    """Anchor the requested root before any path discovery can redirect it."""
    source = Path(os.path.abspath(os.fspath(source)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError("TensorFold source root must be a regular, non-symlink directory") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("TensorFold source root must be a regular, non-symlink directory")
        yield source, descriptor
    finally:
        os.close(descriptor)


def _directory_flags():
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory(root_descriptor, parts):
    current = os.dup(root_descriptor)
    try:
        for part in parts:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = next_descriptor
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise ValueError("source path component is not a directory")
        return current
    except (OSError, ValueError) as exc:
        os.close(current)
        raise ValueError("source directory could not be opened safely") from exc


def _mode_at(directory_descriptor, name):
    try:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return None


def _package_prefix(root_descriptor):
    candidates = (("src", "tensorfold"), ("python", "tensorfold"), ("tensorfold",), ())
    for parts in candidates:
        try:
            descriptor = _open_directory(root_descriptor, parts)
        except ValueError:
            continue
        try:
            init_mode = _mode_at(descriptor, "__init__.py")
            cuda_mode = _mode_at(descriptor, "cuda")
            server_mode = _mode_at(descriptor, "server")
            if (init_mode is not None and stat.S_ISREG(init_mode)
                    and any(mode is not None and stat.S_ISDIR(mode)
                            for mode in (cuda_mode, server_mode))):
                return parts
        finally:
            os.close(descriptor)
    raise ValueError("TensorFold package root not found (expected src/tensorfold, python/tensorfold or tensorfold)")


def _read_file_at(directory_descriptor, name, relative):
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, os.O_RDONLY | nofollow | nonblock,
                             dir_fd=directory_descriptor)
    except OSError as exc:
        raise ValueError(f"source path could not be read safely: {relative.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source files must be regular and non-symlink")
        if before.st_size > MAX_FILE_BYTES:
            raise ValueError(f"source file exceeds {MAX_FILE_BYTES} bytes: {relative.name}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or len(data) != before.st_size:
            raise ValueError(f"source file changed while being inventoried: {relative.name}")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"source file exceeds {MAX_FILE_BYTES} bytes: {relative.name}")
        return data
    finally:
        os.close(descriptor)


def _read_beneath(package, relative, root_descriptor=None):
    """Open a source through non-symlink directory descriptors."""
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe source path")
    owned_root = None
    try:
        if root_descriptor is None:
            owned_root = os.open(package, _directory_flags())
            root_descriptor = owned_root
        directory = _open_directory(root_descriptor, parts[:-1])
        try:
            return _read_file_at(directory, parts[-1], relative)
        finally:
            os.close(directory)
    except OSError as exc:
        raise ValueError(f"source path could not be read safely: {relative.name}") from exc
    finally:
        if owned_root is not None:
            os.close(owned_root)


def _kind(relative, data):
    suffix = relative.suffix.lower()
    if suffix in {".cu", ".cuh"}:
        return "cuda"
    if suffix in {".cpp", ".cc", ".c", ".h", ".hpp"}:
        return "native"
    if suffix == ".zig":
        return "zig"
    if b"import triton" in data or b"from triton" in data or b"@triton.jit" in data:
        return "triton"
    return "python"


def _source_manifest_open(source, root_descriptor, require_exl3):
    package_parts = _package_prefix(root_descriptor)
    rows = []
    total = 0
    directories = 0
    pending = [()]
    while pending:
        relative_directory = pending.pop()
        directories += 1
        if directories > MAX_DIRECTORIES:
            raise ValueError(f"too many source directories (limit {MAX_DIRECTORIES})")
        descriptor = _open_directory(root_descriptor, package_parts + relative_directory)
        try:
            try:
                with os.scandir(descriptor) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError as exc:
                raise ValueError("source tree could not be traversed completely") from exc
            for entry in entries:
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as exc:
                    raise ValueError("source tree could not be traversed completely") from exc
                relative_parts = relative_directory + (entry.name,)
                if stat.S_ISLNK(mode):
                    raise ValueError("symlinks are not allowed in a source manifest")
                if stat.S_ISDIR(mode):
                    pending.append(relative_parts)
                    continue
                if not stat.S_ISREG(mode):
                    raise ValueError("special files are not allowed in a source manifest")
                relative = Path(*relative_parts)
                if relative.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                data = _read_file_at(descriptor, entry.name, relative)
                total += len(data)
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("TensorFold source inventory exceeds one GiB")
                parts = {part.lower() for part in relative.parts}
                exl3 = "exl3" in parts or "exl3" in relative.stem.lower()
                rows.append({
                    "path": relative.as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "kind": _kind(relative, data),
                    "exl3": exl3,
                })
                if len(rows) > MAX_FILES:
                    raise ValueError(f"too many source files (limit {MAX_FILES})")
        finally:
            os.close(descriptor)
    rows.sort(key=lambda row: row["path"])
    kinds = Counter(row["kind"] for row in rows)
    exl3_rows = [row for row in rows if row["exl3"]]
    components = {
        "cuda": kinds["cuda"] > 0,
        "triton": kinds["triton"] > 0,
        "zig": kinds["zig"] > 0,
        "server": any(row["path"].startswith("server/") for row in rows),
        "scheduler": any("scheduler" in row["path"].lower() for row in rows),
        "exl3": bool(exl3_rows),
    }
    errors = []
    if not rows:
        errors.append("no TensorFold source files found")
    if not components["cuda"]:
        errors.append("no CUDA sources found")
    if require_exl3:
        if not exl3_rows:
            errors.append("no EXL3 sources found")
        if not any(row["kind"] == "cuda" for row in exl3_rows):
            errors.append("no EXL3 CUDA source found")
        if not any(row["kind"] in {"python", "triton"} for row in exl3_rows):
            errors.append("no EXL3 loader/format source found")
    identity = hashlib.sha256(canonical(rows)).hexdigest()
    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "runtime": "TensorFold",
        "source_files": len(rows),
        "source_bytes": total,
        "tree_sha256": identity,
        "kinds": dict(sorted(kinds.items())),
        "components": components,
        "exl3_files": len(exl3_rows),
        "files": rows,
        "errors": errors,
        "limits": [
            "Source inventory does not compile or execute kernels.",
            "Runtime compatibility still requires a matching TensorFold build, CUDA toolchain and GPU test.",
        ],
    }


def source_manifest(source, require_exl3=False):
    with _source_root(source) as (anchored_source, descriptor):
        return _source_manifest_open(anchored_source, descriptor, require_exl3)
