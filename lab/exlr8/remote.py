# SPDX-License-Identifier: Apache-2.0
"""Inspect an EXLR8 Hugging Face release without downloading weight payloads."""
from collections import Counter, defaultdict
import hashlib
import json
import re
import struct
from urllib.parse import quote, urlsplit
import urllib.error
import urllib.request

from common.config import loads

API_LIMIT = 16 * 1024 * 1024
FILE_LIMIT = 16 * 1024 * 1024
HEADER_LIMIT = 8 * 1024 * 1024
MAX_FILES = 5000
MAX_HEADER_SAMPLES = 32
MAX_TP = 1024
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPERT = re.compile(r"experts/exlr8-layer-(\d{3})\.(index|k([234]))\.safetensors\Z")
SLICE = re.compile(r"index-slices/tp(\d+)/k([234])/rank(\d+)\.safetensors\Z")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
EXLR8_CODEBOOK = (
    "mcg 0xCBAC1FED, every value rounded to E4M3 inside the search "
    "(round to nearest even, saturating finite)"
)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        redirect_host = (parsed.hostname or "").lower()
        try:
            redirect_port = parsed.port
        except ValueError as exc:
            raise urllib.error.URLError("refused redirect port") from exc
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or redirect_port not in (None, 443)
                or not (redirect_host == "huggingface.co"
                        or redirect_host.endswith(".huggingface.co")
                        or redirect_host.endswith(".hf.co"))):
            raise urllib.error.URLError("refused redirect host")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        origin = urlsplit(request.full_url)
        origin_port = origin.port or 443
        target_port = redirect_port or 443
        if (redirected is not None
                and (origin.scheme, origin.hostname, origin_port)
                != (parsed.scheme, parsed.hostname, target_port)):
            redirected.remove_header("Authorization")
        return redirected


class HfClient:
    """Small stdlib client pinned to Hugging Face and bounded by byte counts."""

    def __init__(self, token=None, timeout=30):
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise ValueError("timeout must be in (0, 300]")
        if token:
            if (not isinstance(token, str) or len(token) > 8192
                    or any(not 33 <= ord(character) <= 126 for character in token)):
                raise ValueError("Hugging Face token contains unsupported characters")
        self.token = token
        self.timeout = timeout
        self.downloaded_bytes = 0
        self.opener = urllib.request.build_opener(_SafeRedirect())

    def _read(self, url, limit, headers=None, require_range=None):
        request_headers = {"Accept": "application/json", **(headers or {})}
        if self.token and urlsplit(url).hostname == "huggingface.co":
            request_headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if require_range is not None:
                    start, end, total = require_range
                    if (response.status != 206
                            or response.headers.get("Content-Range", "") != f"bytes {start}-{end}/{total}"):
                        raise ValueError("server did not honor the bounded byte range")
                raw = response.read(limit + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError("Hugging Face request failed") from exc
        if len(raw) > limit:
            raise ValueError("Hugging Face response exceeds the configured limit")
        if require_range is not None and len(raw) != require_range[1] - require_range[0] + 1:
            raise ValueError("Hugging Face returned a truncated byte range")
        self.downloaded_bytes += len(raw)
        return raw

    def _json(self, url, limit):
        raw = self._read(url, limit)
        try:
            value = loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Hugging Face returned invalid JSON") from exc
        return value

    def repo_info(self, repo, revision):
        rev = quote(revision, safe="")
        return self._json(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", API_LIMIT)

    def json_file(self, repo, commit, path):
        path = quote(path, safe="/")
        return self._json(f"https://huggingface.co/{repo}/resolve/{commit}/{path}", FILE_LIMIT)

    def header(self, repo, commit, path, file_size):
        url = f"https://huggingface.co/{repo}/resolve/{commit}/{quote(path, safe='/')}"
        prefix = self._read(url, 8, {"Range": "bytes=0-7"}, (0, 7, file_size))
        size = struct.unpack("<Q", prefix)[0]
        if size <= 1 or size > HEADER_LIMIT or 8 + size > file_size:
            raise ValueError(f"{path}: invalid or excessive safetensors header")
        raw = self._read(url, size, {"Range": f"bytes=8-{7 + size}"},
                         (8, 7 + size, file_size))
        try:
            value = loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid safetensors header JSON") from exc
        if not isinstance(value, dict) or not isinstance(value.get("__metadata__"), dict):
            raise ValueError(f"{path}: missing safetensors metadata")
        metadata = value.pop("__metadata__")
        if any(not isinstance(key, str) or not isinstance(item, str)
               for key, item in metadata.items()):
            raise ValueError(f"{path}: safetensors metadata must map strings to strings")
        return metadata, value, size


def _files(info):
    if not isinstance(info, dict) or not isinstance(info.get("sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", info["sha"]):
        raise ValueError("repository metadata has no immutable commit SHA")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or not siblings or len(siblings) > MAX_FILES:
        raise ValueError("repository metadata has an invalid file inventory")
    files = {}
    for row in siblings:
        if not isinstance(row, dict) or not isinstance(row.get("rfilename"), str):
            raise ValueError("repository metadata has an invalid file entry")
        path = row["rfilename"]
        if path in files or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("repository metadata has an unsafe or duplicate path")
        size = row.get("size")
        lfs = row.get("lfs")
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        if path.endswith(".safetensors"):
            if type(size) is not int or size <= 8 or not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError(f"{path}: safetensors file lacks bounded size/LFS SHA-256 metadata")
        files[path] = {"size": size, "sha256": digest}
    return files


def _manifest(path, value):
    if not isinstance(value, dict) or value.get("format") != "exlr8-v2-manifest":
        raise ValueError(f"{path}: not an exlr8-v2-manifest")
    if not isinstance(value.get("version"), str) or not value["version"]:
        raise ValueError(f"{path}: missing format version")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError(f"{path}: missing variant name")
    if not isinstance(value.get("set_id"), str) or not SHA256.fullmatch(value["set_id"]):
        raise ValueError(f"{path}: invalid set_id")
    layers = value.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path}: no layer width table")
    required = set()
    experts = halves = None
    counts = Counter()
    for layer, row in layers.items():
        if not isinstance(layer, str) or not layer.isdigit() or not isinstance(row, dict):
            raise ValueError(f"{path}: invalid layer entry")
        gate, up, down, shared = (row.get(name) for name in ("gate", "up", "down", "shared"))
        if not isinstance(gate, list) or not gate or not isinstance(up, list) or not isinstance(down, list):
            raise ValueError(f"{path}: layer {layer} has invalid routed widths")
        if len(gate) != len(up) or len(gate) != len(down):
            raise ValueError(f"{path}: layer {layer} routed width counts disagree")
        current_experts = len(gate)
        current_halves = len(gate[0]) if isinstance(gate[0], list) else 0
        if (not current_halves or len(down) != current_experts
                or any(not isinstance(row2, list) or len(row2) != current_halves for row2 in gate + up)):
            raise ValueError(f"{path}: layer {layer} has inconsistent half widths")
        if experts is None:
            experts, halves = current_experts, current_halves
        if (current_experts, current_halves) != (experts, halves):
            raise ValueError(f"{path}: layer {layer} geometry differs from other layers")
        values = [item for row2 in gate + up for item in row2] + down
        if shared is not None:
            if not isinstance(shared, dict) or not isinstance(shared.get("gate"), list) or not isinstance(shared.get("up"), list):
                raise ValueError(f"{path}: layer {layer} has an invalid shared expert")
            if len(shared["gate"]) != halves or len(shared["up"]) != halves:
                raise ValueError(f"{path}: layer {layer} shared half widths disagree")
            values += shared["gate"] + shared["up"] + [shared.get("down")]
        if any(type(width) is not int or width not in {2, 3, 4} for width in values):
            raise ValueError(f"{path}: layer {layer} uses an unsupported width")
        counts.update(values)
        required.update((int(layer), width) for width in values)
    return {
        "name": value["name"], "version": value["version"], "set_id": value["set_id"],
        "layers": len(layers), "experts": experts, "halves": halves,
        "width_counts": {str(k): v for k, v in sorted(counts.items())},
    }, required, set(layers)


def _sidecar(path):
    if path.startswith("dense/"):
        return "hashes/" + path + ".sha256"
    if path.startswith("experts/"):
        return "hashes/" + path.split("/", 1)[1] + ".sha256"
    if path.startswith("index-slices/"):
        return "hashes/" + path + ".sha256"
    raise ValueError("unexpected weight path")


def _header_samples(files, count):
    mandatory, candidates = [], []
    experts = sorted(path for path in files if EXPERT.fullmatch(path))
    for width in ("k2", "k3", "k4"):
        match = next((path for path in experts if f".{width}." in path), None)
        if match:
            mandatory.append(match)
    indexes = [path for path in experts if ".index." in path]
    if indexes:
        mandatory += indexes
    dense = sorted(path for path in files if path.startswith("dense/") and path.endswith(".safetensors"))
    if dense:
        mandatory.append(dense[0])
        candidates.append(dense[-1])
        vocab = next((path for path in dense if "vocab" in path), None)
        if vocab:
            candidates.append(vocab)
    slices = sorted(path for path in files if SLICE.fullmatch(path))
    if slices:
        mandatory.append(slices[0])
        candidates.append(slices[-1])
    selected = list(dict.fromkeys(mandatory))
    target = max(count, len(selected))
    for path in candidates:
        if path not in selected and len(selected) < target:
            selected.append(path)
    return selected


def _tensor_layout_errors(tensors, payload_bytes):
    errors = []
    if not isinstance(tensors, dict) or not tensors:
        return ["header has no tensor entries"]
    if type(payload_bytes) is not int or payload_bytes < 0:
        return ["header exceeds declared file size"]
    ranges = []
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            errors.append("invalid tensor entry")
            continue
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            errors.append(f"{name}: unsupported dtype")
            continue
        if (not isinstance(shape, list) or len(shape) > 16
                or any(type(value) is not int or not 0 <= value <= 2**63 - 1 for value in shape)):
            errors.append(f"{name}: invalid shape")
            continue
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_bytes):
            errors.append(f"{name}: invalid data offsets")
            continue
        elements = 1
        for value in shape:
            elements *= value
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[dtype]:
            errors.append(f"{name}: payload length does not match dtype and shape")
            continue
        ranges.append((offsets[0], offsets[1], name))
    previous_end = 0
    for start, end, name in sorted(ranges):
        if start != previous_end:
            reason = "overlapping" if start < previous_end else "unindexed gap before"
            errors.append(f"{name}: {reason} tensor payload")
        previous_end = max(previous_end, end)
    if ranges and previous_end != payload_bytes:
        errors.append("trailing payload bytes are not indexed")
    return errors


def _validate_header(path, metadata, tensors, set_id, file_size, header_size):
    errors = []
    if (not isinstance(metadata, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in metadata.items())):
        return ["safetensors metadata must map strings to strings"]
    errors += _tensor_layout_errors(tensors, file_size - 8 - header_size)
    if path.startswith("experts/"):
        match = EXPERT.fullmatch(path)
        layer, kind, width = match.groups()
        if metadata.get("format") != "exlr8-v2" or metadata.get("scheme") != "exlr8-e4m3-v2":
            errors.append("format/scheme mismatch")
        if metadata.get("layer") != str(int(layer)):
            errors.append("layer mismatch")
        if kind == "index":
            if metadata.get("contents") != "index" or metadata.get("codebook") != EXLR8_CODEBOOK:
                errors.append("index does not declare rounded E4M3 MCG")
        elif (metadata.get("contents") != "trellis" or metadata.get("k") != width
              or not {"trellis.gate", "trellis.up", "trellis.down"} <= set(tensors)):
            errors.append("trellis metadata/tensors mismatch")
    elif path.startswith("dense/"):
        if metadata.get("format") != "exlr8-v2" or metadata.get("scheme") != "exlr8-e4m3-v2":
            errors.append("format/scheme mismatch")
        expected = "vocab" if "vocab" in path else "dense"
        if metadata.get("contents") != expected:
            errors.append("dense/vocab content mismatch")
    else:
        match = SLICE.fullmatch(path)
        tp, width, rank = match.groups()
        if (metadata.get("format") != "exlr8-v2-index-slice" or metadata.get("tp") != tp
                or metadata.get("k") != width or metadata.get("rank") != rank
                or metadata.get("set_id") != set_id):
            errors.append("index-slice metadata mismatch")
    return errors


def inspect_hf(repo, revision="main", token=None, timeout=30, header_samples=10, client=None):
    if not isinstance(repo, str) or not REPO.fullmatch(repo):
        raise ValueError("repo must be an owner/name Hugging Face repository ID")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision) or ".." in revision.split("/"):
        raise ValueError("revision contains unsupported characters")
    if type(header_samples) is not int or not 0 <= header_samples <= MAX_HEADER_SAMPLES:
        raise ValueError(f"header_samples must be in [0, {MAX_HEADER_SAMPLES}]")
    client = client or HfClient(token, timeout)
    info = client.repo_info(repo, revision)
    files = _files(info)
    commit = info["sha"]
    manifest_paths = sorted(path for path in files if path.startswith("manifests/") and path.endswith(".json"))
    if not manifest_paths:
        raise ValueError("release has no manifests")
    variants, required, layer_sets = [], set(), []
    for path in manifest_paths:
        summary, needs, layer_set = _manifest(path, client.json_file(repo, commit, path))
        variants.append(summary)
        required |= needs
        layer_sets.append(layer_set)
    if len({row["name"] for row in variants}) != len(variants):
        raise ValueError("variant names are not unique")
    set_ids = {row["set_id"] for row in variants}
    if len(set_ids) != 1:
        raise ValueError("manifests disagree on set_id")
    if any(layer_set != layer_sets[0] for layer_set in layer_sets[1:]):
        raise ValueError("manifests disagree on covered layers")
    set_id = set_ids.pop()

    errors = []
    for layer in layer_sets[0]:
        path = f"experts/exlr8-layer-{int(layer):03d}.index.safetensors"
        if path not in files:
            errors.append(f"missing {path}")
    for layer, width in sorted(required):
        path = f"experts/exlr8-layer-{layer:03d}.k{width}.safetensors"
        if path not in files:
            errors.append(f"missing {path}")

    weights = sorted(path for path in files if path.endswith(".safetensors")
                     and (path.startswith("dense/") or path.startswith("experts/")))
    identity = hashlib.sha256()
    weight_bytes = 0
    for path in weights:
        row = files[path]
        if path.startswith("experts/") and not EXPERT.fullmatch(path):
            errors.append(f"unrecognized expert weight path {path}")
        identity.update(f"{path} {row['size']} {row['sha256']}\n".encode())
        weight_bytes += row["size"]
        if _sidecar(path) not in files:
            errors.append(f"missing hash sidecar for {path}")
    computed_set_id = identity.hexdigest()
    if computed_set_id != set_id:
        errors.append("manifest set_id does not match LFS weight metadata")

    slices = defaultdict(set)
    for path in files:
        match = SLICE.fullmatch(path)
        if match:
            tp, width, rank = map(int, match.groups())
            if not 1 <= tp <= MAX_TP or not 0 <= rank < tp:
                errors.append(f"invalid TP/rank bounds in {path}")
                continue
            slices[(tp, width)].add(rank)
            if _sidecar(path) not in files:
                errors.append(f"missing hash sidecar for {path}")
    for (tp, width), ranks in sorted(slices.items()):
        if ranks != set(range(tp)):
            errors.append(f"TP{tp} K{width} index slices do not cover ranks 0..{tp - 1}")
    required_widths = {width for _, width in required}
    slice_tps = {tp for tp, _ in slices}
    if not slices:
        errors.append("release has no TP index slices")
    for tp in sorted(slice_tps):
        present_widths = {width for current_tp, width in slices if current_tp == tp}
        if present_widths != required_widths:
            errors.append(f"TP{tp} index slices do not cover manifest widths {sorted(required_widths)}")

    headers = []
    for path in _header_samples(files, header_samples):
        metadata, tensors, size = client.header(repo, commit, path, files[path]["size"])
        row_errors = _validate_header(path, metadata, tensors, set_id, files[path]["size"], size)
        headers.append({"path": path, "header_bytes": size, "tensor_entries": len(tensors), "errors": row_errors})
        errors += [f"{path}: {error}" for error in row_errors]

    required_root = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for path in required_root:
        if path not in files:
            errors.append(f"missing {path}")
    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "format": "EXLR8",
        "repository": repo,
        "requested_revision": revision,
        "commit": commit,
        "set_id": set_id,
        "computed_set_id": computed_set_id,
        "variants": variants,
        "weight_files": len(weights),
        "weight_bytes": weight_bytes,
        "index_slice_sets": [{"tp": tp, "k": width, "ranks": len(ranks)} for (tp, width), ranks in sorted(slices.items())],
        "header_samples": headers,
        "downloaded_metadata_bytes": client.downloaded_bytes,
        "errors": errors,
        "limits": [
            "Uses repository metadata, manifests and bounded safetensors header ranges; weight payloads are not downloaded.",
            "LFS SHA-256 identities and sidecar presence are checked, but payload hashes and sidecar contents are not recomputed.",
            "Header compatibility does not execute rounded-E4M3 decoding, kernels, collectives or the serving engine.",
        ],
    }
