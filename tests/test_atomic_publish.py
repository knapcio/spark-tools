# SPDX-License-Identifier: Apache-2.0
import os
import stat
import threading

import pytest

from common import atomic_publish
from common.atomic_publish import (
    PathReplacedError,
    PublishConflictError,
    UnsafePathError,
    publish_bytes,
    validate_bytes,
)


class InjectedCrash(BaseException):
    pass


def crash_at(stage):
    def fault(current):
        if current == stage:
            raise InjectedCrash(stage)
    return fault


def temporary_files(target):
    return sorted(target.parent.glob(f".{target.name}.atomic-*"))


def select_backend(monkeypatch, backend):
    if backend == "link":
        monkeypatch.setattr(atomic_publish, "_native_no_replace", lambda *args: None)
    else:
        real_native = atomic_publish._native_no_replace

        def require_native(*args):
            method = real_native(*args)
            assert method in {"renameat2", "renameatx_np"}
            return method

        monkeypatch.setattr(atomic_publish, "_native_no_replace", require_native)


def test_publish_and_idempotent_validation(tmp_path):
    target = tmp_path / "result.bin"
    first = publish_bytes(target, b"complete")
    second = publish_bytes(target, b"complete")
    checked = validate_bytes(target, b"complete")
    assert (first.created, second.created, checked.created) == (True, False, False)
    assert first.size == second.size == checked.size == 8
    assert first.sha256 == second.sha256 == checked.sha256
    assert target.read_bytes() == b"complete"
    assert temporary_files(target) == []


@pytest.mark.parametrize("stage", ["reserved", "full_before_publish"])
def test_prepublication_crash_preserves_orphan_and_retry_ignores_it(tmp_path, stage):
    target = tmp_path / "result.bin"
    with pytest.raises(InjectedCrash):
        publish_bytes(target, b"complete", _fault=crash_at(stage))
    assert not target.exists()
    assert len(temporary_files(target)) == 1
    assert publish_bytes(target, b"complete").created
    assert target.read_bytes() == b"complete"
    assert len(temporary_files(target)) == 1


def test_partial_write_crash_preserves_partial_orphan(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    real_write = atomic_publish.os.write

    def short_write(descriptor, data):
        return real_write(descriptor, data[:3])

    monkeypatch.setattr(atomic_publish.os, "write", short_write)
    with pytest.raises(InjectedCrash):
        publish_bytes(target, b"complete", _fault=crash_at("partial_write"))
    orphan = temporary_files(target)
    assert len(orphan) == 1
    assert orphan[0].read_bytes() == b"com"
    assert not target.exists()


def test_short_writes_are_completed(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    real_write = atomic_publish.os.write

    def short_write(descriptor, data):
        return real_write(descriptor, data[:2])

    monkeypatch.setattr(atomic_publish.os, "write", short_write)
    assert publish_bytes(target, b"complete").created
    assert target.read_bytes() == b"complete"


def test_reservation_flags_and_fsync_order(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    real_open = atomic_publish.os.open
    real_fsync = atomic_publish.os.fsync
    reservations = []
    synced_kinds = []

    def record_open(path, flags, *args, **kwargs):
        if isinstance(path, str) and path.startswith(f".{target.name}.atomic-"):
            reservations.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def record_fsync(descriptor):
        synced_kinds.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        return real_fsync(descriptor)

    monkeypatch.setattr(atomic_publish.os, "open", record_open)
    monkeypatch.setattr(atomic_publish.os, "fsync", record_fsync)
    publish_bytes(target, b"complete")
    assert len(reservations) == 1
    assert reservations[0] & os.O_EXCL
    assert reservations[0] & os.O_NOFOLLOW
    assert synced_kinds[0] == "file"
    assert "directory" in synced_kinds[1:]


def test_zero_progress_write_fails_without_canonical_file(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    monkeypatch.setattr(atomic_publish.os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError, match="no progress"):
        publish_bytes(target, b"complete")
    assert not target.exists()
    assert temporary_files(target) == []


def test_crash_after_publish_is_valid_on_retry(tmp_path):
    target = tmp_path / "result.bin"
    with pytest.raises(InjectedCrash):
        publish_bytes(target, b"complete", _fault=crash_at("publish_before_dir_fsync"))
    assert target.read_bytes() == b"complete"
    assert not publish_bytes(target, b"complete").created


@pytest.mark.parametrize("backend", ["native", "link"])
@pytest.mark.parametrize("attack", ["symlink", "regular", "in_place", "disappear"])
def test_source_path_attack_between_fsync_and_publish_fails_closed(
    tmp_path, monkeypatch, backend, attack
):
    target = tmp_path / "result.bin"
    other = tmp_path / "other.bin"
    other.write_bytes(b"tampered")
    select_backend(monkeypatch, backend)

    def attack_source(stage):
        if stage != "prepublish_verified":
            return
        source = temporary_files(target)[0]
        if attack == "symlink":
            source.unlink()
            source.symlink_to(other.name)
        elif attack == "regular":
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(b"tampered")
            replacement.replace(source)
        elif attack == "in_place":
            source.write_bytes(b"tampered")
        else:
            source.unlink()

    with pytest.raises(OSError):
        publish_bytes(target, b"complete", _fault=attack_source)


@pytest.mark.parametrize("backend", ["native", "link"])
def test_canonical_mutation_before_postpublication_validation_fails_closed(
    tmp_path, monkeypatch, backend
):
    target = tmp_path / "result.bin"
    select_backend(monkeypatch, backend)

    def mutate_target(stage):
        if stage == "publish_before_dir_fsync":
            target.write_bytes(b"tampered")

    with pytest.raises(PathReplacedError):
        publish_bytes(target, b"complete", _fault=mutate_target)
    assert target.read_bytes() == b"tampered"


@pytest.mark.parametrize("backend", ["native", "link"])
def test_restored_canonical_contents_still_reject_metadata_race(
    tmp_path, monkeypatch, backend
):
    target = tmp_path / "result.bin"
    select_backend(monkeypatch, backend)

    def mutate_and_restore(stage):
        if stage == "publish_before_dir_fsync":
            target.write_bytes(b"tampered")
            target.write_bytes(b"complete")

    with pytest.raises(PathReplacedError):
        publish_bytes(target, b"complete", _fault=mutate_and_restore)
    assert target.read_bytes() == b"complete"


def test_link_fallback_crash_may_retain_linked_orphan(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    monkeypatch.setattr(atomic_publish, "_native_no_replace", lambda *args: None)
    with pytest.raises(InjectedCrash):
        publish_bytes(target, b"complete", _fault=crash_at("publish_before_dir_fsync"))
    orphan = temporary_files(target)
    assert len(orphan) == 1
    assert os.stat(target).st_ino == os.stat(orphan[0]).st_ino
    assert not publish_bytes(target, b"complete").created


def test_link_fallback_cleans_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    monkeypatch.setattr(atomic_publish, "_native_no_replace", lambda *args: None)
    assert publish_bytes(target, b"complete").created
    assert target.read_bytes() == b"complete"
    assert temporary_files(target) == []


def test_no_overwrite_on_conflict(tmp_path):
    target = tmp_path / "result.bin"
    target.write_bytes(b"first")
    with pytest.raises(PublishConflictError):
        publish_bytes(target, b"second")
    assert target.read_bytes() == b"first"
    assert temporary_files(target) == []


def test_concurrent_identical_publishers_have_one_creator(tmp_path):
    target = tmp_path / "result.bin"
    barrier = threading.Barrier(12)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            results.append(publish_bytes(target, b"shared"))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum(result.created for result in results) == 1
    assert target.read_bytes() == b"shared"
    assert temporary_files(target) == []


def test_concurrent_different_publishers_do_not_overwrite(tmp_path):
    target = tmp_path / "result.bin"
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def worker(data):
        try:
            barrier.wait()
            results.append((data, publish_bytes(target, data)))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(value,)) for value in (b"first", b"second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert len(errors) == 1 and isinstance(errors[0], PublishConflictError)
    assert target.read_bytes() == results[0][0]
    assert temporary_files(target) == []


def test_symlink_canonical_is_rejected(tmp_path):
    target = tmp_path / "result.bin"
    other = tmp_path / "other.bin"
    other.write_bytes(b"complete")
    target.symlink_to(other.name)
    with pytest.raises(UnsafePathError):
        publish_bytes(target, b"complete")
    assert other.read_bytes() == b"complete"


def test_canonical_replacement_during_validation_is_rejected(tmp_path):
    target = tmp_path / "result.bin"
    target.write_bytes(b"complete")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"complete")

    def replace(stage):
        if stage == "validation_after_lstat":
            replacement.replace(target)

    with pytest.raises(PathReplacedError):
        validate_bytes(target, b"complete", _fault=replace)


def test_parent_replacement_is_detected(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    target = parent / "result.bin"

    def replace(stage):
        if stage == "directory_opened":
            parent.rename(moved)
            parent.mkdir()

    with pytest.raises(PathReplacedError):
        publish_bytes(target, b"complete", _fault=replace)
    assert (moved / "result.bin").read_bytes() == b"complete"
    assert not target.exists()


def test_existing_non_regular_path_is_rejected(tmp_path):
    target = tmp_path / "result.bin"
    target.mkdir()
    with pytest.raises(UnsafePathError):
        publish_bytes(target, b"complete")


def test_input_and_mode_validation(tmp_path):
    with pytest.raises(TypeError, match="bytes"):
        publish_bytes(tmp_path / "result.bin", "text")
    with pytest.raises(ValueError, match="mode"):
        publish_bytes(tmp_path / "result.bin", b"data", mode=0o1000)


def test_current_platform_uses_atomic_backend(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    methods = []
    real = atomic_publish._publish_no_replace

    def record(*args):
        result = real(*args)
        methods.append(result[0])
        return result

    monkeypatch.setattr(atomic_publish, "_publish_no_replace", record)
    publish_bytes(target, b"complete")
    allowed = {"renameatx_np", "link"} if os.sys.platform == "darwin" else {"renameat2", "link"}
    assert methods[0] in allowed


