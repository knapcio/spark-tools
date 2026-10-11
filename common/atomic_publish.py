# SPDX-License-Identifier: Apache-2.0
"""Crash-safe, create-only publication of small immutable byte records."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import os
import secrets
import stat
import sys
from typing import Callable


class AtomicPublishError(OSError):
    """Base class for publication safety failures."""


class PublishConflictError(FileExistsError, AtomicPublishError):
    """The canonical path already contains different bytes."""


class UnsafePathError(AtomicPublishError):
    """A path is not a stable regular file or directory."""


class PathReplacedError(AtomicPublishError):
    """A checked path was replaced while it was in use."""


@dataclass(frozen=True)
class PublishResult:
    created: bool
    size: int
    sha256: str


_Fault = Callable[[str], None]
_TEMP_ATTEMPTS = 128
_READ_SIZE = 1024 * 1024


def _flags(*names: str) -> int:
    value = 0
    for name in names:
        flag = getattr(os, name, None)
        if flag is None:
            raise NotImplementedError(f"platform lacks {name}")
        value |= flag
    return value


def _fire(fault: _Fault | None, stage: str) -> None:
    if fault is not None:
        fault(stage)


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _payload_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _open_directory(path: str) -> tuple[int, os.stat_result]:
    flags = _flags("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnsafePathError(error.errno, "destination directory is unavailable or unsafe") from error
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode):
        os.close(descriptor)
        raise UnsafePathError(errno.ENOTDIR, "destination parent is not a directory")
    return descriptor, current


def _assert_directory_stable(path: str, expected: os.stat_result) -> None:
    descriptor = None
    try:
        descriptor, current = _open_directory(path)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise PathReplacedError(errno.ESTALE, "destination directory was replaced")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_open(
    directory: int,
    name: str,
    expected: bytes,
    fault: _Fault | None = None,
) -> PublishResult:
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise UnsafePathError(errno.EINVAL, "canonical path is not a regular file")
    _fire(fault, "validation_after_lstat")

    flags = _flags("O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW")
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        raise UnsafePathError(error.errno, "canonical path cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise UnsafePathError(errno.EINVAL, "canonical path is not a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PathReplacedError(errno.ESTALE, "canonical path was replaced before open")
        _fire(fault, "validation_after_open")

        offset = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _READ_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            end = offset + len(chunk)
            if end > len(expected) or chunk != expected[offset:end]:
                raise PublishConflictError(errno.EEXIST, "canonical file has different contents")
            offset = end
        if offset != len(expected):
            raise PublishConflictError(errno.EEXIST, "canonical file has different contents")

        after = os.fstat(descriptor)
        _fire(fault, "validation_before_relstat")
        try:
            linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError as error:
            raise PathReplacedError(errno.ESTALE, "canonical path disappeared during validation") from error
        if _fingerprint(before) != _fingerprint(opened):
            raise PathReplacedError(errno.ESTALE, "canonical file changed before validation")
        if _fingerprint(opened) != _fingerprint(after):
            raise PathReplacedError(errno.ESTALE, "canonical file changed during validation")
        if _fingerprint(after) != _fingerprint(linked):
            raise PathReplacedError(errno.ESTALE, "canonical path was replaced during validation")
        return PublishResult(False, offset, digest.hexdigest())
    finally:
        os.close(descriptor)


def _read_descriptor_exact(descriptor: int, expected: bytes) -> str:
    offset = 0
    digest = hashlib.sha256()
    while True:
        chunk = os.pread(descriptor, _READ_SIZE, offset)
        if not chunk:
            break
        digest.update(chunk)
        end = offset + len(chunk)
        if end > len(expected) or chunk != expected[offset:end]:
            raise PathReplacedError(errno.ESTALE, "publication source has different contents")
        offset = end
    if offset != len(expected):
        raise PathReplacedError(errno.ESTALE, "publication source has different contents")
    return digest.hexdigest()


def _verify_source(
    directory: int,
    name: str,
    descriptor: int,
    expected: bytes,
    baseline: os.stat_result,
) -> os.stat_result:
    try:
        named_before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        raise PathReplacedError(errno.ESTALE, "publication source disappeared") from error
    held_before = os.fstat(descriptor)
    for value in (baseline, named_before, held_before):
        if not stat.S_ISREG(value.st_mode):
            raise UnsafePathError(errno.EINVAL, "publication source is not a regular file")
    identity = (baseline.st_dev, baseline.st_ino)
    if (named_before.st_dev, named_before.st_ino) != identity:
        raise PathReplacedError(errno.ESTALE, "publication source path was replaced")
    if (held_before.st_dev, held_before.st_ino) != identity:
        raise PathReplacedError(errno.ESTALE, "publication source descriptor was replaced")
    if baseline.st_nlink != 1 or named_before.st_nlink != 1 or held_before.st_nlink != 1:
        raise PathReplacedError(errno.ESTALE, "publication source has an unexpected link count")
    if _fingerprint(named_before) != _fingerprint(baseline):
        raise PathReplacedError(errno.ESTALE, "publication source metadata changed")
    if _fingerprint(held_before) != _fingerprint(baseline):
        raise PathReplacedError(errno.ESTALE, "publication source metadata changed")
    _read_descriptor_exact(descriptor, expected)
    held_after = os.fstat(descriptor)
    try:
        named_after = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        raise PathReplacedError(errno.ESTALE, "publication source disappeared") from error
    if _fingerprint(held_after) != _fingerprint(baseline):
        raise PathReplacedError(errno.ESTALE, "publication source changed during validation")
    if _fingerprint(named_after) != _fingerprint(baseline):
        raise PathReplacedError(errno.ESTALE, "publication source path changed during validation")
    return held_after


def _validate_published(
    directory: int,
    target: str,
    descriptor: int,
    expected: bytes,
    baseline: os.stat_result,
    expected_nlink: int,
    expected_ctime_ns: int,
) -> tuple[os.stat_result, PublishResult]:
    held_before = os.fstat(descriptor)
    try:
        named_before = os.stat(target, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        raise PathReplacedError(errno.ESTALE, "canonical path disappeared after publication") from error
    for value in (held_before, named_before):
        if not stat.S_ISREG(value.st_mode):
            raise UnsafePathError(errno.EINVAL, "published path is not a regular file")
        if (value.st_dev, value.st_ino) != (baseline.st_dev, baseline.st_ino):
            raise PathReplacedError(errno.ESTALE, "published path is not the pinned source")
        if _payload_fingerprint(value) != _payload_fingerprint(baseline):
            raise PathReplacedError(errno.ESTALE, "published contents or metadata changed")
        if value.st_nlink != expected_nlink:
            raise PathReplacedError(errno.ESTALE, "published file has an unexpected link count")
        if value.st_ctime_ns != expected_ctime_ns:
            raise PathReplacedError(errno.ESTALE, "published file ctime changed unexpectedly")

    _read_descriptor_exact(descriptor, expected)
    validated = _validate_open(directory, target, expected)
    held_after = os.fstat(descriptor)
    try:
        named_after = os.stat(target, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        raise PathReplacedError(errno.ESTALE, "canonical path disappeared during validation") from error
    for value in (held_after, named_after):
        if _payload_fingerprint(value) != _payload_fingerprint(baseline):
            raise PathReplacedError(errno.ESTALE, "published file changed during validation")
        if value.st_nlink != expected_nlink:
            raise PathReplacedError(errno.ESTALE, "published link count changed during validation")
        if value.st_ctime_ns != expected_ctime_ns:
            raise PathReplacedError(errno.ESTALE, "published file ctime changed during validation")
    if (held_after.st_dev, held_after.st_ino) != (named_after.st_dev, named_after.st_ino):
        raise PathReplacedError(errno.ESTALE, "canonical path was replaced during validation")
    return held_after, validated


def _validate_source_link(
    directory: int,
    name: str,
    descriptor: int,
    baseline: os.stat_result,
    expected_nlink: int,
    expected_ctime_ns: int,
) -> os.stat_result:
    try:
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        raise PathReplacedError(errno.ESTALE, "publication source link disappeared") from error
    held = os.fstat(descriptor)
    for value in (named, held):
        if not stat.S_ISREG(value.st_mode):
            raise UnsafePathError(errno.EINVAL, "publication source link is not a regular file")
        if _payload_fingerprint(value) != _payload_fingerprint(baseline):
            raise PathReplacedError(errno.ESTALE, "publication source link was replaced or changed")
        if value.st_nlink != expected_nlink:
            raise PathReplacedError(errno.ESTALE, "publication source has an unexpected link count")
        if value.st_ctime_ns != expected_ctime_ns:
            raise PathReplacedError(errno.ESTALE, "publication source ctime changed unexpectedly")
    if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
        raise PathReplacedError(errno.ESTALE, "publication source link was replaced")
    return held


def _reserve(directory: int, target: str, mode: int) -> tuple[int, str]:
    flags = _flags("O_RDWR", "O_CREAT", "O_EXCL", "O_CLOEXEC", "O_NOFOLLOW")
    for _ in range(_TEMP_ATTEMPTS):
        name = f".{target}.atomic-{secrets.token_hex(16)}"
        try:
            return os.open(name, flags, mode, dir_fd=directory), name
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "could not reserve a unique temporary file")


def _write_all(descriptor: int, data: bytes, fault: _Fault | None) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "write made no progress")
        if written > len(view) - offset:
            raise OSError(errno.EIO, "write reported an invalid byte count")
        offset += written
        if offset < len(view):
            _fire(fault, "partial_write")


def _native_no_replace(directory: int, source: str, target: str) -> str | None:
    encoded_source = os.fsencode(source)
    encoded_target = os.fsencode(target)
    library = ctypes.CDLL(None, use_errno=True)
    function = None
    flag = 0
    method = ""
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flag = 1  # RENAME_NOREPLACE
        method = "renameat2"
    elif sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = 4  # RENAME_EXCL
        method = "renameatx_np"
    if function is None:
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(directory, encoded_source, directory, encoded_target, flag) == 0:
        return method
    code = ctypes.get_errno()
    unsupported = {errno.ENOSYS, errno.EINVAL, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}
    if code in unsupported:
        return None
    raise OSError(code, os.strerror(code))


def _publish_no_replace(directory: int, source: str, target: str) -> tuple[str, bool]:
    method = _native_no_replace(directory, source, target)
    if method is not None:
        return method, False
    os.link(
        source,
        target,
        src_dir_fd=directory,
        dst_dir_fd=directory,
        follow_symlinks=False,
    )
    return "link", True


def _unlink_if_present(directory: int, name: str) -> bool:
    try:
        os.unlink(name, dir_fd=directory)
        return True
    except FileNotFoundError:
        return False


def validate_bytes(path: str | os.PathLike[str], expected: bytes, *, _fault: _Fault | None = None) -> PublishResult:
    """Validate a canonical regular file without following its final symlink."""
    if not isinstance(expected, bytes):
        raise TypeError("expected must be bytes")
    absolute = os.path.abspath(os.fspath(path))
    parent, name = os.path.split(absolute)
    if not name or name in (".", ".."):
        raise ValueError("path must name a file")
    directory, directory_stat = _open_directory(parent)
    try:
        _fire(_fault, "directory_opened")
        result = _validate_open(directory, name, expected, _fault)
        _assert_directory_stable(parent, directory_stat)
        return result
    finally:
        os.close(directory)


def publish_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    mode: int = 0o600,
    _fault: _Fault | None = None,
) -> PublishResult:
    """Publish immutable bytes once, or validate an identical prior result.

    Temporary files are deliberately unique and old temporary files are
    ignored. A process interruption may leave one behind; it can be retained
    as crash evidence or removed by a separate age-based cleanup policy.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if type(mode) is not int or mode < 0 or mode > 0o777:
        raise ValueError("mode must be an integer between 0 and 0o777")
    absolute = os.path.abspath(os.fspath(path))
    parent, name = os.path.split(absolute)
    if not name or name in (".", ".."):
        raise ValueError("path must name a file")

    directory, directory_stat = _open_directory(parent)
    temporary = None
    descriptor = None
    published = False
    preserve_temporary = False
    try:
        _fire(_fault, "directory_opened")
        try:
            result = _validate_open(directory, name, data, _fault)
        except FileNotFoundError:
            pass
        else:
            _assert_directory_stable(parent, directory_stat)
            return result

        descriptor, temporary = _reserve(directory, name, mode)
        try:
            _fire(_fault, "reserved")
        except BaseException:
            preserve_temporary = True
            raise
        try:
            _write_all(descriptor, data, _fault)
        except BaseException as error:
            if _fault is not None and not isinstance(error, OSError):
                preserve_temporary = True
            raise
        os.fsync(descriptor)
        baseline = os.fstat(descriptor)
        try:
            _fire(_fault, "full_before_publish")
        except BaseException:
            preserve_temporary = True
            raise
        _verify_source(directory, temporary, descriptor, data, baseline)
        try:
            _fire(_fault, "prepublish_verified")
        except BaseException:
            preserve_temporary = True
            raise
        _verify_source(directory, temporary, descriptor, data, baseline)

        try:
            _, temporary_is_linked = _publish_no_replace(directory, temporary, name)
            published = True
        except FileExistsError:
            _verify_source(directory, temporary, descriptor, data, baseline)
            result = _validate_open(directory, name, data, _fault)
            if not _unlink_if_present(directory, temporary):
                raise PathReplacedError(errno.ESTALE, "publication source disappeared during cleanup")
            temporary = None
            unlinked = os.fstat(descriptor)
            if unlinked.st_nlink != 0 or unlinked.st_ctime_ns < baseline.st_ctime_ns:
                raise PathReplacedError(errno.ESTALE, "publication source cleanup was not isolated")
            os.fsync(directory)
            _assert_directory_stable(parent, directory_stat)
            return result

        publication_pin = os.fstat(descriptor)
        publication_nlink = 2 if temporary_is_linked else 1
        if _payload_fingerprint(publication_pin) != _payload_fingerprint(baseline):
            raise PathReplacedError(errno.ESTALE, "pinned source changed during publication")
        if publication_pin.st_nlink != publication_nlink:
            raise PathReplacedError(errno.ESTALE, "pinned source link count changed during publication")
        try:
            _fire(_fault, "publish_before_dir_fsync")
        except BaseException:
            preserve_temporary = True
            raise
        if temporary_is_linked:
            linked_stat, _ = _validate_published(
                directory, name, descriptor, data, baseline, 2, publication_pin.st_ctime_ns
            )
            _validate_source_link(
                directory, temporary, descriptor, baseline, 2, publication_pin.st_ctime_ns
            )
            os.fsync(directory)
            if not _unlink_if_present(directory, temporary):
                raise PathReplacedError(errno.ESTALE, "publication source link disappeared")
            temporary = None
            cleanup_pin = os.fstat(descriptor)
            if cleanup_pin.st_nlink != 1 or cleanup_pin.st_ctime_ns < linked_stat.st_ctime_ns:
                raise PathReplacedError(errno.ESTALE, "publication source cleanup was not isolated")
            os.fsync(directory)
            _, validated = _validate_published(
                directory, name, descriptor, data, baseline, 1, cleanup_pin.st_ctime_ns
            )
        else:
            try:
                os.stat(temporary, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise PathReplacedError(errno.ESTALE, "native publication left a source path")
            temporary = None
            published_stat, _ = _validate_published(
                directory, name, descriptor, data, baseline, 1, publication_pin.st_ctime_ns
            )
            os.fsync(directory)
            _, validated = _validate_published(
                directory, name, descriptor, data, baseline, 1, published_stat.st_ctime_ns
            )
        _assert_directory_stable(parent, directory_stat)
        return PublishResult(True, validated.size, validated.sha256)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None and not published and not preserve_temporary:
            if _unlink_if_present(directory, temporary):
                os.fsync(directory)
        os.close(directory)
