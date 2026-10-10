# Export policy

Run `spark-export-audit .` before committing and after the final commit. Every
finding fails the command. It scans all files, including ignored files, every
Git object (including deleted and unreachable content), filenames, commit
metadata and local Git metadata. No content exemptions exist. Symbolic links
and files that cannot be reviewed as UTF-8 text also fail. Embedded control
bytes other than tab, CR and LF are rejected, including NUL-bearing UTF-16.
Unreadable directories, files and metadata abort the audit with an error;
partial traversal never produces a passing report. No remote is allowed.
Binary Git indexes are read through Git's staged-entry listing; the object
scan covers their staged blobs, including content not yet committed.
Linked worktrees share the same object and metadata scan. Their administrative
links contain local paths and therefore fail this strict check. Do not remove
an active worktree to obtain a passing report; complete local integration first
and audit a standalone export repository.

The policy rejects personal paths, private addresses, machine names, SSH
aliases, service identifiers, reserved service ports, privilege handling,
credentials and credential paths, coordination files, private repository and
run identifiers, captured traffic, attribution markers, session links,
private vocabulary and non-English markers. Each category has a synthetic
negative tree test and deleted-history test. Rule fragments keep the audit and
its negative tests subject to exactly the same policy as every other file.

Use anonymised numeric calibration data and explicit configuration. Example
network values use `node0` or `127.0.0.1:8000`. Use environment variables for
endpoint authentication; credentials must never be embedded in configuration
or outputs. Store all experiment outputs outside the checkout.

Pattern matching cannot discover an arbitrary newly invented machine alias
or secret format. Non-English detection includes common ASCII Polish markers,
but is not a language classifier. Manually review all prose, filenames and
commit metadata for non-English text as well as running the audit. Expand
the rule set when a new private class is identified; never add an exemption.
