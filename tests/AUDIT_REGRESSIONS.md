# Audit regression evidence

Verified offline on Python 3.14.6 with pytest 9.1.1 from the local package cache.
The baseline was `4878d74`, with the new tests added before changing
`common/audit.py`. The final tests were also rerun in an isolated directory
against `common` loaded directly from that revision. No network or GPU was used.

Command in the test environment:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_audit.py -q
```

Before: **35 failed, 96 passed**. After: **131 passed**.
Every failing case below became a passing test; none was skipped or marked
as an expected failure. Exact review inputs are constructed from joined
fragments in `REVIEW_MARKERS` so the test source obeys the export policy too.

| Finding | Regression tests | Before | After |
| --- | --- | --- | --- |
| P1 marker variants | `test_review_marker_content`, `test_review_marker_filename`, `test_review_marker_commit_metadata`, each with `host`, `port`, `label`, `key-path`, `assignment`, `author`, `prose` | All 21 failed: no required finding | All 21 pass |
| P1 author identities | `test_review_attribution_git_author`, both identities | Both failed: no attribution finding in commit object | Both pass |
| P1 credential header | `test_review_bearer_credential` | Failed: no credential finding | Passes |
| P1 UTF-16LE input | `test_review_utf16le_file` | Failed: empty findings for the exact encoded host marker | Passes byte scan and file audit |
| P1 NUL/control input | `test_review_control_bytes_fail_closed`, bytes 0, 1, 27, 127 | All 4 failed: tree reported PASS | All 4 pass |
| P1 unreadable directories | `test_review_unreadable_directory`, `root`, `nested`, `git-root`, `git-nested` | All 4 failed: traversal suppressed the injected PermissionError | All 4 pass: error propagates |
| Unreviewable special files | `test_review_special_file_fails_closed`, `tree`, `git-metadata` | Both failed: FIFO silently ignored | Both pass without opening the FIFO |

The marker cases reproduce the four reviewed basenames, the unquoted API
assignment with the two synthetic hexadecimal components, the reviewed
author line and the exact ASCII Polish sentence. Each is checked as file
content, a filename and commit metadata. The author-identity test additionally
checks real Git author and committer fields. The UTF-16LE case uses the exact
machine marker without a BOM. The directory tests inject `PermissionError`
from `os.scandir`, including the `configs` root and both the Git metadata root
and a nested metadata directory.

`test_review_clean_staged_index_and_text` passes before and after. It protects
ordinary tab/CR/LF text and Git's NUL-delimited staged-entry framing from false
binary findings. The existing clean-authorship test still passes.

Related validation:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_audit.py tests/test_gb10check.py tests/test_common.py -q
```

**160 passed**, including exact GB10 example output comparisons after updating
the occupancy method and limitation wording. The MIT notice now states that
the inherited unresolved-licence caveat has no applicable bundled material.

Owned-file audit scope: root project documentation and metadata, `LICENSES`,
`common`, `configs`, `docs`, `gb10check`, `examples/gb10check`, the three test
modules above and this evidence file. Worker packages and the integration
smoke script are outside this change.

The 50 owned files were copied to an isolated local repository with a fresh
root commit using the project author identity. `audit_tree` with history
enabled returned **PASS: 50 files, 59 Git objects, zero findings**, including
filenames, staged entries, commit metadata and local Git metadata. This is an
owned-scope export check, not certification of unrelated integration history.
