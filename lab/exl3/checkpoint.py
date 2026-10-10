# SPDX-License-Identifier: Apache-2.0
"""Bounded, header-only validation of EXL3 safetensors checkpoints.

The tensor naming and trellis geometry follow ExLlamaV3's public EXL3 format.
No tensor payload is read except the four-byte codebook marker.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
import re
import stat
import struct

from common.config import loads

BITS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)
MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
PARTS = {"trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias"}
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_HEADER_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SHARDS = 4096
_NUMBER = re.compile(r"\.\d+(?=\.|$)")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _root_directory(model_dir):
    root = os.path.abspath(os.fspath(model_dir))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ValueError("model directory must be a regular, non-symlink directory") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("model directory must be a regular, non-symlink directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _entry_info(root_descriptor, name):
    try:
        return os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{name} could not be inspected safely") from exc


def _open_regular(root_descriptor, name, expected=None):
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("unsafe checkpoint filename")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as exc:
        raise ValueError(f"{name} must be a regular, non-symlink file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (expected is not None and _identity(info) != expected):
            raise ValueError(f"{name} must be an unchanged regular, non-symlink file")
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def _unchanged(descriptor, before, name):
    if _identity(os.fstat(descriptor)) != _identity(before):
        raise ValueError(f"{name} changed while being inspected")


def _read_json(root_descriptor, name, expected):
    descriptor, before = _open_regular(root_descriptor, name, expected)
    try:
        if before.st_size > MAX_JSON_BYTES:
            raise ValueError(f"{name} exceeds 16 MiB")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        _unchanged(descriptor, before, name)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{name} exceeds 16 MiB")
    try:
        value = loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _quantization_fields(root_descriptor, config):
    blocks = []
    for parent in (config, config.get("text_config") or {}):
        if not isinstance(parent, dict):
            continue
        for key in ("quantization_config", "quantization"):
            blocks.append(parent.get(key) or {})
    sidecar = _entry_info(root_descriptor, "quantization_config.json")
    if sidecar is not None:
        if not stat.S_ISREG(sidecar.st_mode):
            raise ValueError("quantization_config.json must be a regular, non-symlink file")
        blocks.append(_read_json(root_descriptor, "quantization_config.json", _identity(sidecar)))
    for block in blocks:
        if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "exl3":
            return dict(block)
    return {}


def _shards(root_descriptor):
    before = os.fstat(root_descriptor)
    try:
        with os.scandir(root_descriptor) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise ValueError("model directory could not be traversed safely") from exc
    after = os.fstat(root_descriptor)
    if ((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("model directory changed while being inspected")
    shards = []
    for entry in entries:
        if not entry.name.endswith(".safetensors"):
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError("safetensors shard could not be inspected safely") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("safetensors shards must be regular, non-symlink files")
        shards.append((entry.name, _identity(info)))
    if not shards:
        raise ValueError("no root-level .safetensors shards found")
    if len(shards) > MAX_SHARDS:
        raise ValueError(f"too many safetensors shards (limit {MAX_SHARDS})")
    return shards


def _header(root_descriptor, filename, expected):
    descriptor, before = _open_regular(root_descriptor, filename, expected)
    size = before.st_size
    try:
        handle = os.fdopen(os.dup(descriptor), "rb")
        with handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{filename}: truncated safetensors prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 1 or header_size > MAX_HEADER_BYTES or 8 + header_size > size:
                raise ValueError(f"{filename}: invalid or excessive safetensors header")
            try:
                header = loads(handle.read(header_size).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{filename}: invalid safetensors header JSON") from exc
        _unchanged(descriptor, before, filename)
    finally:
        os.close(descriptor)
    if not isinstance(header, dict):
        raise ValueError(f"{filename}: safetensors header must be an object")
    metadata = header.pop("__metadata__", None)
    if (metadata is not None
            and (not isinstance(metadata, dict)
                 or any(not isinstance(key, str) or not isinstance(value, str)
                        for key, value in metadata.items()))):
        raise ValueError(f"{filename}: safetensors metadata must map strings to strings")
    payload = size - 8 - header_size
    ranges = []
    for tensor_name, entry in header.items():
        if not isinstance(tensor_name, str) or not tensor_name or not isinstance(entry, dict):
            raise ValueError(f"{filename}: invalid tensor entry")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if (not isinstance(dtype, str) or dtype not in DTYPE_BYTES
                or not isinstance(shape, list) or len(shape) > 16
                or any(type(v) is not int or not 0 <= v <= 2**63 - 1 for v in shape)):
            raise ValueError(f"{filename}: invalid dtype/shape for {tensor_name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload):
            raise ValueError(f"{filename}: invalid data offsets for {tensor_name}")
        elements = 1
        for value in shape:
            elements *= value
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[dtype]:
            raise ValueError(f"{filename}: payload length does not match dtype and shape for {tensor_name}")
        ranges.append((offsets[0], offsets[1], tensor_name))
    previous_end = 0
    for start, end, tensor_name in sorted(ranges):
        if start != previous_end:
            reason = "overlapping" if start < previous_end else "unindexed gap before"
            raise ValueError(f"{filename}: {reason} tensor payload for {tensor_name}")
        previous_end = end
    if previous_end != payload:
        raise ValueError(f"{filename}: trailing payload bytes are not indexed")
    return header, header_size


def _marker(root_descriptor, name, expected, header_size, entry):
    if entry["dtype"] != "I32" or entry["shape"] not in ([], [1]):
        raise ValueError("codebook marker must be one int32 scalar")
    start, end = entry["data_offsets"]
    if end - start != 4:
        raise ValueError("codebook marker must occupy four bytes")
    descriptor, before = _open_regular(root_descriptor, name, expected)
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            handle.seek(8 + header_size + start)
            raw = handle.read(4)
        _unchanged(descriptor, before, name)
    finally:
        os.close(descriptor)
    if len(raw) != 4:
        raise ValueError("truncated codebook marker")
    return struct.unpack("<I", raw)[0]


def _scale(parts, full, packed, length, label):
    present = [name for name in (full, packed) if name in parts]
    if len(present) != 1:
        raise ValueError(f"exactly one {label} scale tensor ({full} or {packed}) is required")
    name = present[0]
    expected = ("F16", [length]) if name == full else ("I16", [length // 16])
    actual = (parts[name]["entry"]["dtype"], parts[name]["entry"]["shape"])
    if actual != expected:
        raise ValueError(f".{name} must be {expected[0]} {expected[1]}, got {actual[0]} {actual[1]}")
    start, end = parts[name]["entry"]["data_offsets"]
    if end - start != 2 * expected[1][0]:
        raise ValueError(f".{name} payload length does not match its dtype and shape")
    return name


def _group(prefix, parts, root_descriptor):
    trellis = parts["trellis"]["entry"]
    shape = trellis["shape"]
    if trellis["dtype"] != "I16" or len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("trellis must be int16 [K/16, N/16, 16 * bits]")
    start, end = trellis["data_offsets"]
    if end - start != 2 * shape[0] * shape[1] * shape[2]:
        raise ValueError("trellis payload length does not match its dtype and shape")
    bits = shape[2] / 16
    bits = int(bits) if float(bits).is_integer() else bits
    if bits not in BITS:
        raise ValueError(f"unsupported EXL3 width {bits:g}")
    markers = [name for name in MARKERS if name in parts]
    if len(markers) > 1:
        raise ValueError("both mcg and mul1 codebook markers are present")
    codebook = markers[0] if markers else "3inst"
    if not float(bits).is_integer() and codebook != "mul1":
        raise ValueError(f"{bits:g}-bit tiles require the mul1 codebook")
    k, n = 16 * shape[0], 16 * shape[1]
    if k % 128 or n % 128:
        raise ValueError(f"K={k} and N={n} must be multiples of 128")
    in_scales = _scale(parts, "suh", "su", k, "input")
    out_scales = _scale(parts, "svh", "sv", n, "output")
    marker_value = None
    if markers:
        part = parts[codebook]
        marker_value = _marker(root_descriptor, part["file"], part["identity"],
                               part["header_size"], part["entry"])
        if marker_value != MARKERS[codebook]:
            raise ValueError(f"unexpected {codebook} marker 0x{marker_value:08x}")
    if "bias" in parts:
        bias = parts["bias"]["entry"]
        if bias["dtype"] not in {"F16", "BF16"} or bias["shape"] != [n]:
            raise ValueError(f"bias must be F16 or BF16 with shape [{n}]")
    return {
        "bits": bits,
        "codebook": codebook,
        "k": k,
        "n": n,
        "input_scales": in_scales,
        "output_scales": out_scales,
        "bias": "bias" in parts,
        "trellis_bytes": shape[0] * shape[1] * shape[2] * 2,
        "files": sorted({part["file"] for part in parts.values()}),
    }


def _inspect_checkpoint_open(root_descriptor):
    config_info = _entry_info(root_descriptor, "config.json")
    if config_info is not None and not stat.S_ISREG(config_info.st_mode):
        raise ValueError("config.json must be a regular, non-symlink file")
    config = (_read_json(root_descriptor, "config.json", _identity(config_info))
              if config_info is not None else {})
    quant = _quantization_fields(root_descriptor, config)
    if not quant:
        raise ValueError("config does not declare quant_method=exl3")

    entries = {}
    total_headers = 0
    shards = _shards(root_descriptor)
    for filename, identity in shards:
        header, header_size = _header(root_descriptor, filename, identity)
        total_headers += header_size
        if total_headers > MAX_TOTAL_HEADER_BYTES:
            raise ValueError("aggregate safetensors headers exceed 512 MiB")
        for name, entry in header.items():
            if name in entries:
                raise ValueError(f"duplicate tensor name across shards: {name}")
            entries[name] = {
                "file": filename, "identity": identity,
                "header_size": header_size, "entry": entry,
            }

    grouped = defaultdict(dict)
    for name, value in entries.items():
        prefix, dot, part = name.rpartition(".")
        if dot and part in PARTS:
            grouped[prefix][part] = value

    groups, invalid, taken = {}, {}, set()
    for prefix, parts in sorted(grouped.items()):
        if "trellis" not in parts:
            continue
        taken.update(f"{prefix}.{part}" for part in parts)
        try:
            groups[prefix] = _group(prefix, parts, root_descriptor)
        except ValueError as exc:
            invalid[prefix] = str(exc)
    if not groups:
        raise ValueError("no valid EXL3 tensor groups found")

    weights = sum(row["k"] * row["n"] for row in groups.values())
    by_bits = Counter()
    by_codebook = Counter()
    categories = defaultdict(Counter)
    for prefix, row in groups.items():
        by_bits[row["bits"]] += row["k"] * row["n"]
        by_codebook[row["codebook"]] += 1
        categories[_NUMBER.sub(".*", prefix)][row["bits"]] += 1
    trellis = sum(row["trellis_bytes"] for row in groups.values())
    errors = [f"{prefix}: {reason}" for prefix, reason in sorted(invalid.items())]
    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "format": "EXL3",
        "model_type": config.get("model_type") or (config.get("text_config") or {}).get("model_type"),
        "quantization": {key: quant[key] for key in ("quant_method", "version", "bits", "head_bits", "mtp_bits", "codebook") if key in quant},
        "shards": len(shards),
        "tensor_entries": len(entries),
        "exl3_tensors": len(groups),
        "plain_tensors": len(entries) - len(taken),
        "codebooks": dict(sorted(by_codebook.items())),
        "weight_share_by_bits": {f"{bits:g}": round(count / weights, 8) for bits, count in sorted(by_bits.items())},
        "average_trellis_bits": round(8 * trellis / weights, 8),
        "trellis_bytes": trellis,
        "categories": {name: {f"{bits:g}": count for bits, count in sorted(rows.items())} for name, rows in sorted(categories.items())},
        "errors": errors,
    }


def inspect_checkpoint(model_dir):
    with _root_directory(model_dir) as root_descriptor:
        return _inspect_checkpoint_open(root_descriptor)
