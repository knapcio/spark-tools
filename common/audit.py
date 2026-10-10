# SPDX-License-Identifier: Apache-2.0
"""Fail closed on private export markers in files, names and every Git object.

Rules use joined fragments so the policy can inspect its own source and tests
without exemptions. Findings report locations and rule names, never matched
secret content. Git inspection includes unreachable objects and local logs.
"""
import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import stat
import subprocess
from common.config import dump


def _join(*parts):
    return "".join(parts)


# No path or content allowlist. Synthetic negative tests use the same joining.
RULES = {
    "personal-path": _join(r"/(?:Us", r"ers|ho", r"me)/|~/(?:Projects|\.ssh)|\bzbi", "zbi\\b"),
    "temporary-path": _join(r"/(?:var/fol", r"ders|private/t", r"mp)(?:/|\b)|/tmp/[A-Za-z0-9_-]{6,}"),
    "private-host": _join(r"\b(?:Spa", r"rk[_-]?0[\dx]+|nuc", r"-agent)(?:\b|[_-])"),
    "ssh-alias": r"(?im)^\s*Host[ \t]+(?!node\d+(?=\s|$)|localhost(?=\s|$)|\*(?=\s|$))\S+|(?-i:\bssh)[ \t]+(?!node\d+(?=\s|$)|localhost(?=\s|$)|127\.0\.0\.1(?=\s|$))\S+|[\"']?ssh_alias[\"']?\s*[:=]\s*[\"']?(?!node\d+(?=[\"'\s,}]|$)|localhost(?=[\"'\s,}]|$))\w+",
    "private-address": _join(r"\b(?:192\.16", r"8\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|100\.\d{1,3}\.\d{1,3}\.\d{1,3})\b|\.", "ts", r"\.net\b"),
    "private-port": _join(r"\b(?:180", r"93|180", r"95|180", r"99|88", r"88)\b"),
    "private-service": _join(r"\b(?:glm-serving", r"-watch|be", r"e(?:[-_][\w-]*)?|dis", r"pram\w*)\b"),
    "privilege-handling": _join(r"\b(?:su", r"do|ha", r"slo)\b|(?:password|passphrase)\s*[:=]\s*[^\s<]"),
    "credential": _join(r"\b(?:hf", r"_[A-Za-z0-9]{8,}|sk", r"-[A-Za-z0-9_-]{8,})\b|[\"']?(?:api[_-]?key|access[_-]?token|secret|hf[_-]?token|zai[_-]?key)[\"']?\s*[:=]\s*[\"']?(?![\"'<])[\w.-]{8,}|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|\bAuthorization\s*:\s*Bearer\s+[^\s<]{8,}"),
    "credential-path": _join(r"(?:key|token|secret)[-_](?:file|path)[\"']?\s*[:=]|/(?:\.ssh|\.secrets|keys)/|(?:hf|huggingface|zai|z\.ai)[-_.](?:api[-_.])?(?:key|token)(?:\.(?:txt|json|env)|\b)|\.(?:cache/huggingface|huggingface)/token\b"),
    "coordination-file": _join(r"\b(?:fleet", r"_busy|fleet[-_]\w+|fleet\.(?:json|lock|yaml|txt)|(?:deployment|admission|release)[-_.]?lock|owner\.txt|lock[-_]fleet|glm[-_]control\.lock)\b"),
    "private-repository": _join(r"-de", r"v\b|\bglm53-full-mac-[\w-]+|\b(?:nextrel|relgate)[-_][\w-]+"),
    "private-run": _join(r"\b20\d{6}(?:[-_]\d{4,6}|T\d{6,}Z)\b|\bglm53(?:full|[-_]full)[-_][\w-]+|\bdiagnostics/|\brun[-_](?:20\d{6}|[a-f0-9]{12,})\b|spark-tools-10", "10"),
    "traffic-data": _join(r"\b(?:zc", r"ode(?:[-_]\w+)?|be", r"e[-_]traffic)\b|(?:traffic|chat)[-_](?:capture|dump)"),
    "attribution": _join(r"\b(?:Clau", r"de|Anth", r"ropic|Co", r"dex)[\w-]*\b|Co", r"[ -]?Authored|(?:AI|assistant|model)[ -](?:generated|authored)|(?:generated|written|authored) (?:by|with) (?:chatgpt|co", r"dex|open", r"ai)|\b(?:author|committer|copyright)\b[^\r\n]*\b(?:open", r"ai|chat", r"gpt)\b"),
    "session-link": _join(r"https?://(?:chatgpt|chat\.openai|clau", r"de)\.(?:com|ai)/(?:c|share|session)/|plug", "in://"),
    "internal-vocabulary": _join(r"\b(?:win", r"dow|B[-_\s]+boot|g", r"en(?:[-_]\d+)?|rece", r"ipt\w*|as", r"tra|s", r"ol|fa", r"ble|stack", r"-2|w", r"4|cand", r"3win\w*)\b"),
    "non-english": _join(r"[^\x00-\x7f]|\b(?:pros", r"ze|dzie", r"kuje|potwier", r"dzam|got", r"owe|uzytk", r"ownik|nie wol", r"no|pry", r"watn[a-z]*|pli", r"k[a-z]*|wyn", r"ik[a-z]*|(?:to|nie|tak) je", r"st)\b"),
}
# Underscores delimit markers in artifact names and configuration keys too.
# Python's word boundary treats them as letters and misses embedded markers.
BOUNDARY = r"(?:(?<![a-z0-9])(?=[a-z0-9])|(?<=[a-z0-9])(?![a-z0-9]))"
PATTERNS = {name: re.compile(pattern.replace(r"\b", BOUNDARY), re.IGNORECASE)
            for name, pattern in RULES.items()}


@dataclass(frozen=True)
class Finding:
    location: str
    line: int
    rule: str


def scan_text(text, location):
    return [Finding(location, text.count("\n", 0, match.start()) + 1, name)
            for name, pattern in PATTERNS.items() for match in pattern.finditer(text)]


def scan_bytes(data, location):
    # UTF-16 without a BOM can decode as UTF-8 while hiding every marker.
    # Only tab, CR and LF are supported text controls.
    if any(byte < 32 and byte not in (9, 10, 13) or byte == 127 for byte in data):
        return [Finding(location, 0, "unreviewable-binary")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(location, 0, "unreviewable-binary")]
    return scan_text(text, location)


def _git(root, *args):
    proc = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(root), *args],
                          capture_output=True, check=False)
    if proc.returncode:
        raise ValueError("Git audit command failed: " + args[0])
    return proc.stdout


def _walk(root, skip=()):
    """Yield paths and types without suppressing traversal or stat errors.

    Skip only explicitly handled top-level entries; never follow symlinks.
    lstat also distinguishes unsupported special files from regular text.
    """
    with os.scandir(root) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        if entry.name in skip:
            continue
        path = Path(entry.path)
        mode = entry.stat(follow_symlinks=False).st_mode
        yield path, mode
        if stat.S_ISDIR(mode):
            yield from _walk(path)


def audit_tree(root, history=True):
    root = Path(root).resolve()
    if not stat.S_ISDIR(root.stat().st_mode):
        raise ValueError("audit root must be a directory")
    findings, files, objects = [], 0, 0
    # The top-level Git database and metadata are inspected separately below.
    for path, mode in _walk(root, skip=(".git",)):
        rel = path.relative_to(root)
        findings.extend(scan_text(rel.as_posix(), "name:" + rel.as_posix()))
        if stat.S_ISLNK(mode):
            findings.append(Finding(rel.as_posix(), 0, "symbolic-link"))
        elif stat.S_ISREG(mode):
            files += 1
            findings.extend(scan_bytes(path.read_bytes(), rel.as_posix()))
        elif not stat.S_ISDIR(mode):
            findings.append(Finding(rel.as_posix(), 0, "unreviewable-file"))
    if history:
        if _git(root, "rev-parse", "--is-inside-work-tree").strip() != b"true":
            raise ValueError("full audit requires a local Git repository")
        git_dir = Path(_git(root, "rev-parse", "--git-common-dir").decode().strip())
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
        # Packed and unreachable objects are included, not just the current branch.
        inventory = _git(root, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")
        for row in inventory.decode().splitlines():
            oid, kind = row.split()
            data = _git(root, "cat-file", "-p", oid)
            # Pretty-printed tree names avoid scanning Git's binary tree framing.
            findings.extend(scan_bytes(data, f"git:{kind}:{oid}"))
            objects += 1
        for path, mode in _walk(git_dir, skip=("objects",)):
            rel = path.relative_to(git_dir)
            location = "git-meta:" + rel.as_posix()
            findings.extend(scan_text(rel.as_posix(), "name:" + location))
            if stat.S_ISLNK(mode):
                findings.append(Finding(location, 0, "symbolic-link"))
            elif stat.S_ISREG(mode):
                is_index = rel.parts == ("index",) or (
                    len(rel.parts) == 3 and rel.parts[0] == "worktrees" and rel.parts[2] == "index")
                if is_index:
                    # Git's binary index framing is not export content. Inspect
                    # staged names via Git; their blobs are in the object scan.
                    entries = _git(root, "--git-dir=" + str(path.parent), "ls-files", "--stage", "-z")
                    # NUL is Git's record separator, never part of a pathname.
                    for entry in entries.split(b"\0"):
                        findings.extend(scan_bytes(entry, "git-index:" + rel.as_posix()))
                else:
                    findings.extend(scan_bytes(path.read_bytes(), location))
            elif not stat.S_ISDIR(mode):
                findings.append(Finding(location, 0, "unreviewable-file"))
        if _git(root, "remote").strip():
            findings.append(Finding("git-meta:config", 0, "configured-remote"))
    return {"status": "FAIL" if findings else "PASS", "files": files, "git_objects": objects,
            "findings": [asdict(item) for item in findings]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_tree(args.root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    dump(report, args.out)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
