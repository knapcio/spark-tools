# SPDX-License-Identifier: Apache-2.0
"""Synthetic markers stay joined so the full export has no policy exemption."""
from pathlib import Path
import os
import subprocess
import pytest
from common.audit import RULES, audit_tree, main, scan_bytes, scan_text


NEGATIVES = {
    "personal-path": "/Us" + "ers/example/code",
    "temporary-path": "/var/fol" + "ders/example",
    "private-host": "Spa" + "rk_01",
    "ssh-alias": "Host example-private",
    "private-address": "100" + ".64.1.2",
    "private-port": "180" + "95",
    "private-service": "glm-serving" + "-watch",
    "privilege-handling": "su" + "do command",
    "credential": "hf" + "_synthetic0123456789",
    "credential-path": "hf" + "-key.txt",
    "coordination-file": "fleet" + "_busy",
    "private-repository": "example-de" + "v",
    "private-run": "202610" + "10-123456",
    "traffic-data": "ZC" + "ode",
    "attribution": "Co" + "-Authored-By: example",
    "session-link": "https://chatgpt.com/" + "c/example",
    "internal-vocabulary": "cand" + "3win9",
    "non-english": "pros" + "ze",
}


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_every_class_is_a_hard_failure(name, tmp_path):
    assert set(NEGATIVES) == set(RULES)
    value = NEGATIVES[name]
    assert name in {row.rule for row in scan_text(value, "synthetic")}
    (tmp_path / "data.txt").write_text(value)
    report = audit_tree(tmp_path, history=False)
    assert report["status"] == "FAIL"
    assert name in {row["rule"] for row in report["findings"]}


@pytest.mark.parametrize("value", [
    "/ho" + "me/example", "/private/t" + "mp/example",
    "spark" + "-03", "nuc" + "-agent", "192.16" + "8.1.1",
    "example." + "ts.net", "180" + "93", "180" + "99", ":88" + "88",
    "be" + "e", "be" + "e-service", "fleet" + ".lock", "dis" + "pramd", "ha" + "slo",
    "sk" + "-synthetic0123456789", "/ke" + "ys/example",
    "Clau" + "de", "Anth" + "ropic", "AI" + "-authored",
    "/diagnos" + "tics/sample", "win" + "dow", "B " + "boot",
    "g" + "en", "rece" + "ipt", "as" + "tra", "s" + "ol",
    "fa" + "ble", "stack" + "-2", "w" + "4", chr(261),
])
def test_additional_markers(value):
    assert scan_text(value, "synthetic")


@pytest.mark.parametrize("name,value", [
    ("ssh-alias", "ss" + "h custom-alias"),
    ("ssh-alias", "ss" + "h node0-private"),
    ("private-host", "Spa" + "rk_0x"),
    ("private-host", "spark" + "-0x"),
    ("ssh-alias", '"ssh_' + 'alias": "custom-alias"'),
    ("ssh-alias", '"ssh_' + 'alias": "node0-private"'),
    ("credential", '"api_' + 'key": "synthetic0123456789"'),
    ("credential", 'zai_' + 'key = "synthetic0123456789"'),
    ("credential", "eyJsynthetic012345." + "eyJsynthetic012345.synthetic012345"),
    ("credential-path", '"api_key_' + 'path": "secret.txt"'),
    ("credential-path", ".cache/huggingface/" + "token"),
    ("credential-path", ".zai-api-" + "key"),
    ("coordination-file", "deployment" + ".lock"),
    ("coordination-file", "admission" + ".lock"),
    ("coordination-file", "release" + ".lock"),
    ("private-run", "202610" + "10T123456789Z"),
    ("internal-vocabulary", "/rece" + "ipts/example.json"),
    ("internal-vocabulary", "g" + "en-12"),
    ("traffic-data", "zc" + "ode_requests"),
    ("attribution", "Clau" + "de_session"),
])
def test_marker_variants_fail_the_tree(name, value, tmp_path):
    (tmp_path / "data.txt").write_text(value)
    report = audit_tree(tmp_path, history=False)
    assert report["status"] == "FAIL"
    assert name in {item["rule"] for item in report["findings"]}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_each_class_in_deleted_history_is_a_hard_failure(name, tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.org")
    (tmp_path / "data.txt").write_text(NEGATIVES[name])
    git(tmp_path, "add", "data.txt")
    git(tmp_path, "commit", "-m", "Add synthetic data")
    git(tmp_path, "rm", "data.txt")
    git(tmp_path, "commit", "-m", "Remove synthetic data")
    report = audit_tree(tmp_path)
    assert report["status"] == "FAIL"
    assert any(item["rule"] == name and item["location"].startswith("git:blob:")
               for item in report["findings"])


def test_deleted_history_and_commit_metadata(tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.org")
    path = tmp_path / "data.txt"
    path.write_text(NEGATIVES["private-address"])
    git(tmp_path, "add", "data.txt")
    git(tmp_path, "commit", "-m", "Add synthetic data")
    git(tmp_path, "rm", "data.txt")
    git(tmp_path, "commit", "-m", "Remove synthetic data")
    report = audit_tree(tmp_path)
    assert report["status"] == "FAIL"
    assert any(item["location"].startswith("git:blob:") for item in report["findings"])
    git(tmp_path, "commit", "--allow-empty", "-m", NEGATIVES["attribution"])
    assert any(item["rule"] == "attribution" and item["location"].startswith("git:commit:")
               for item in audit_tree(tmp_path)["findings"])


def test_names_ignored_files_and_symlinks(tmp_path):
    bad = tmp_path / NEGATIVES["private-host"]
    bad.write_text("clean")
    (tmp_path / "ignored.txt").write_text(NEGATIVES["credential"])
    (tmp_path / "link").symlink_to("ignored.txt")
    nested = tmp_path / "nested" / ".git"
    nested.mkdir(parents=True)
    (nested / "data.txt").write_text(NEGATIVES["private-port"])
    report = audit_tree(tmp_path, history=False)
    assert {item["rule"] for item in report["findings"]} >= {
        "private-host", "credential", "symbolic-link", "private-port"}


def test_clean_placeholders_and_authorship():
    assert scan_text("node0 127.0.0.1:8000 knapcio <knapcio@gmail.com> OpenAI-compatible", "clean") == []
    assert scan_text("Host node0\nssh node0\nssh localhost\nssh 127.0.0.1", "clean") == []


def test_binary_fails_closed(tmp_path):
    (tmp_path / "opaque.bin").write_bytes(b"\xff\xfe")
    assert audit_tree(tmp_path, history=False)["findings"][0]["rule"] == "unreviewable-binary"


def test_unreachable_object_and_cli_failure(tmp_path, capsys):
    git(tmp_path, "init", "--template=", "-b", "main")
    (tmp_path / "data.txt").write_text(NEGATIVES["credential"])
    git(tmp_path, "hash-object", "-w", "data.txt")
    (tmp_path / "data.txt").unlink()
    assert main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert '"credential"' in output
    assert NEGATIVES["credential"] not in output


def test_git_ref_names_and_remote_are_checked(tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.org")
    git(tmp_path, "commit", "--allow-empty", "-m", "Clean content")
    git(tmp_path, "branch", NEGATIVES["private-host"])
    git(tmp_path, "branch", "objects/" + NEGATIVES["private-host"])
    # A local path is sufficient; no network operation is made.
    git(tmp_path, "remote", "add", "example", ".")
    report = audit_tree(tmp_path)
    rules = {item["rule"] for item in report["findings"]}
    assert "private-host" in rules
    assert "configured-remote" in rules
    assert any(item["rule"] == "private-host" and "refs/heads/objects/" in item["location"]
               for item in report["findings"])


def test_linked_worktree_checks_shared_history_and_its_own_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--template=", "-b", "main")
    git(source, "config", "user.name", "Example Author")
    git(source, "config", "user.email", "author@example.org")
    (source / "data.txt").write_text(NEGATIVES["credential"])
    git(source, "add", "data.txt")
    git(source, "commit", "-m", "Add synthetic data")
    git(source, "rm", "data.txt")
    git(source, "commit", "-m", "Remove synthetic data")
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-b", "parallel", str(linked))
    (linked / "local.txt").write_text(NEGATIVES["private-host"])
    report = audit_tree(linked)
    assert report["status"] == "FAIL"
    assert any(item["rule"] == "credential" and item["location"].startswith("git:blob:")
               for item in report["findings"])
    assert any(item["rule"] == "private-host" and item["location"] == "local.txt"
               for item in report["findings"])


def test_deleted_non_ascii_filename_fails_history(tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.org")
    name = chr(261) + ".txt"
    (tmp_path / name).write_text("clean")
    git(tmp_path, "add", name)
    git(tmp_path, "commit", "-m", "Add synthetic file")
    git(tmp_path, "rm", name)
    git(tmp_path, "commit", "-m", "Remove synthetic file")
    assert any(item["rule"] == "non-english" and item["location"].startswith("git:tree:")
               for item in audit_tree(tmp_path)["findings"])


def test_staged_name_is_checked_after_working_file_is_deleted(tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    name = NEGATIVES["private-host"]
    (tmp_path / name).write_text("clean")
    git(tmp_path, "add", name)
    (tmp_path / name).unlink()
    assert any(item["rule"] == "private-host" and item["location"] == "git-index:index"
               for item in audit_tree(tmp_path)["findings"])


# Exact synthetic values from the review, assembled without source exemptions.
REVIEW_MARKERS = [
    pytest.param("private-host", "bench_Spa" + "rk_01.json", id="host"),
    pytest.param("private-port", "metrics_180" + "95.json", id="port"),
    pytest.param("internal-vocabulary", "prefix_rece" + "ipts_run.json", id="label"),
    pytest.param("credential-path", "hf_" + "token_path.txt", id="key-path"),
    pytest.param("credential", "API_" + "KEY=0123456789abcdef.0123456789abcdef", id="assignment"),
    pytest.param("attribution", "Author: Co" + "dex <co" + "dex@open" + "ai.com>", id="author"),
    pytest.param("non-english", "To je" + "st pry" + "watny pli" + "k z wyn" + "ikami testu", id="prose"),
]


@pytest.mark.parametrize("rule,value", REVIEW_MARKERS)
def test_review_marker_content(rule, value, tmp_path):
    assert rule in {item.rule for item in scan_text(value, "synthetic")}
    (tmp_path / "data.txt").write_text(value)
    report = audit_tree(tmp_path, history=False)
    assert report["status"] == "FAIL"
    assert any(item["rule"] == rule and item["location"] == "data.txt"
               for item in report["findings"])


@pytest.mark.parametrize("rule,value", REVIEW_MARKERS)
def test_review_marker_filename(rule, value, tmp_path):
    (tmp_path / value).write_text("clean")
    report = audit_tree(tmp_path, history=False)
    assert report["status"] == "FAIL"
    assert any(item["rule"] == rule and item["location"] == "name:" + value
               for item in report["findings"])


@pytest.mark.parametrize("rule,value", REVIEW_MARKERS)
def test_review_marker_commit_metadata(rule, value, tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.org")
    git(tmp_path, "commit", "--allow-empty", "-m", value)
    assert any(item["rule"] == rule and item["location"].startswith("git:commit:")
               for item in audit_tree(tmp_path)["findings"])


@pytest.mark.parametrize("author", ["Co" + "dex", "Open" + "AI"])
def test_review_attribution_git_author(author, tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    git(tmp_path, "config", "user.name", author)
    git(tmp_path, "config", "user.email", "author@example.org")
    git(tmp_path, "commit", "--allow-empty", "-m", "Clean content")
    assert any(item["rule"] == "attribution" and item["location"].startswith("git:commit:")
               for item in audit_tree(tmp_path)["findings"])


def test_review_utf16le_file(tmp_path):
    data = ("Spa" + "rk_01").encode("utf-16le")
    assert scan_bytes(data, "synthetic")[0].rule == "unreviewable-binary"
    (tmp_path / "data.txt").write_bytes(data)
    report = audit_tree(tmp_path, history=False)
    assert report["status"] == "FAIL"
    assert any(item["rule"] == "unreviewable-binary" and item["location"] == "data.txt"
               for item in report["findings"])


@pytest.mark.parametrize("data", [b"clean\x00text", b"clean\x01text", b"clean\x1btext", b"clean\x7ftext"])
def test_review_control_bytes_fail_closed(data, tmp_path):
    (tmp_path / "data.txt").write_bytes(data)
    assert audit_tree(tmp_path, history=False)["status"] == "FAIL"


@pytest.mark.parametrize("denied", ["root", "nested", "git-root", "git-nested"])
def test_review_unreadable_directory(denied, tmp_path, monkeypatch):
    root = tmp_path / "configs"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested/data.txt").write_text("clean")
    history = denied.startswith("git-")
    if history:
        git(root, "init", "--template=", "-b", "main")
        (root / ".git/custom").mkdir()
        (root / ".git/custom/data.txt").write_text("clean")
    target = {"root": root, "nested": root / "nested",
              "git-root": root / ".git", "git-nested": root / ".git/custom"}[denied]
    scandir = os.scandir
    calls = []

    def deny(path):
        if Path(path) == target:
            calls.append(path)
            raise PermissionError("synthetic denied traversal")
        return scandir(path)

    monkeypatch.setattr(os, "scandir", deny)
    with pytest.raises(PermissionError, match="synthetic denied traversal"):
        audit_tree(root, history=history)
    assert calls


def test_review_clean_staged_index_and_text(tmp_path):
    git(tmp_path, "init", "--template=", "-b", "main")
    (tmp_path / "data.txt").write_bytes(b"clean\ttext\r\n")
    git(tmp_path, "add", "data.txt")
    assert audit_tree(tmp_path)["status"] == "PASS"


def test_review_bearer_credential(tmp_path):
    value = "Authorization: Bear" + "er 0123456789abcdef.0123456789abcdef"
    (tmp_path / "data.txt").write_text(value)
    report = audit_tree(tmp_path, history=False)
    assert any(item["rule"] == "credential" for item in report["findings"])


@pytest.mark.parametrize("metadata", [False, True], ids=["tree", "git-metadata"])
def test_review_special_file_fails_closed(metadata, tmp_path):
    parent = tmp_path
    if metadata:
        git(tmp_path, "init", "--template=", "-b", "main")
        parent = tmp_path / ".git"
    os.mkfifo(parent / "opaque")
    report = audit_tree(tmp_path, history=metadata)
    assert any(item["rule"] == "unreviewable-file" for item in report["findings"])
