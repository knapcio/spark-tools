# SPDX-License-Identifier: Apache-2.0
"""Independent scalar arithmetic for EXL3/EXLR8 synthetic trellis fixtures.

Format credit: Turboderp / ExLlamaV3 (MIT, 2025), TensorFold contributors
(Apache-2.0), Kindling AI / gr8 contributors (MIT, 2026). No upstream
implementation is bundled; pinned external oracles are in manifests.py.
"""
from bisect import bisect_left
from functools import lru_cache
import hashlib
import math
import struct

WIDTHS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)
CODEBOOKS = ("3inst", "mcg", "mul1")
MAX_DIM = 256
GOLDEN = {
    "mcg": "95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9",
    "3inst": "ad88de2dcff5bb05fff2faec37cceceb43aa1c2e8203f9d81a6fbd6dd3393eea",
    "mul1": "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8",
    "exlr8-fp32": "44e78f7a0260c97e54c7f0122e7df4bcdd362601baa36f45d8038973877c624d",
    "exlr8-e4m3": "852259a151b9d7c4d8cc6b7537ba5f548e78681488e926953c04e94b275591b9",
}


def half_from_bits(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half_round(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def exl3_value(state, codebook="mcg"):
    if type(state) is not int or not 0 <= state < 65536:
        raise ValueError("state must be an integer in 0..65535")
    if codebook == "mul1":
        word = (state * 0x83DCD12D) & 0xFFFFFFFF
        total = 1024 + sum((word >> shift) & 255 for shift in (0, 8, 16, 24))
        return half_round(total * half_from_bits(0x1EEE) + half_from_bits(0xC931))
    if codebook not in ("mcg", "3inst"):
        raise ValueError("unknown EXL3 codebook")
    word = ((state * 0xCBAC1FED) if codebook == "mcg"
            else (state * 89226354 + 64248484)) & 0xFFFFFFFF
    word = (word & 0x8FFF8FFF) ^ 0x3B603B60
    return half_round(half_from_bits(word & 65535) + half_from_bits(word >> 16))


def e4m3_value(code):
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("E4M3 code must be a byte")
    magnitude = code & 127
    if magnitude == 127:
        return math.copysign(float("nan"), -1 if code & 128 else 1)
    exponent, fraction = magnitude >> 3, magnitude & 7
    value = (fraction / 512 if exponent == 0
             else math.ldexp(1 + fraction / 8, exponent - 7))
    return math.copysign(value, -1 if code & 128 else 1)


E4M3_POSITIVE = tuple(e4m3_value(code) for code in range(127))


def e4m3_round_bits(value):
    """Clamp finite input to +/-448, then RN-even; reject nonfinite input.

    Signed zero is retained. Infinity/NaN are outside the synthetic fixture
    policy (MCG states are finite); rejection does not assert upstream policy.
    """
    if not math.isfinite(value):
        raise ValueError("nonfinite input is outside the conformance fixture policy")
    sign = 128 if math.copysign(1, value) < 0 else 0
    magnitude = min(abs(value), 448)
    high = bisect_left(E4M3_POSITIVE, magnitude)
    if high == 0:
        return sign
    low = high - 1
    down, up = magnitude - E4M3_POSITIVE[low], E4M3_POSITIVE[high] - magnitude
    selected = low if down < up or (down == up and low % 2 == 0) else high
    return sign | selected


def e4m3_round_bits_arithmetic(value):
    """Independent exponent/significand oracle, without the value table."""
    if not math.isfinite(value):
        raise ValueError("nonfinite input is outside the conformance fixture policy")
    sign = 128 if math.copysign(1, value) < 0 else 0
    magnitude = min(abs(value), 448)
    if magnitude < 1 / 64:
        return sign | round(magnitude * 512)
    exponent = math.frexp(magnitude)[1] - 1
    step = math.ldexp(1.0, exponent - 3)
    significand = round(magnitude / step)
    if significand == 16:
        exponent += 1
        significand = 8
    return sign | ((exponent + 7) * 8 + significand - 8)


@lru_cache(maxsize=4)
def codebook_values(name):
    if name == "exlr8":
        return tuple(e4m3_value(e4m3_round_bits(exl3_value(state))) for state in range(65536))
    if name not in CODEBOOKS:
        raise ValueError("unknown codebook")
    return tuple(exl3_value(state, name) for state in range(65536))


def value_bytes(values, dtype="fp16"):
    if dtype not in {"fp16", "fp32"}:
        raise ValueError("dtype must be fp16 or fp32")
    pack = "<e" if dtype == "fp16" else "<f"
    return b"".join(struct.pack(pack, value) for value in values)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _width(bits):
    if isinstance(bits, bool) or bits not in WIDTHS:
        raise ValueError("unsupported EXL3 width")
    return int(bits * 16)


def tile_states(raw, bits):
    """Read tail-biting 16-bit windows from LE32/MSB-first packed stream."""
    words = _width(bits)
    if not isinstance(raw, bytes) or len(raw) != words * 2:
        raise ValueError("tile payload length/dtype does not match width")
    stream = []
    for (word,) in struct.iter_unpack("<I", raw):
        stream.extend((word >> bit) & 1 for bit in range(31, -1, -1))
    states = []
    for position in range(1, 257):
        end = (position * int(bits) if float(bits).is_integer()
               else (position * int(2 * bits) - position % 2) // 2)
        state = 0
        for index in range(end - 16, end):
            state = (state << 1) | stream[index % len(stream)]
        states.append(state)
    return states


def exlr8_symbol_states(raw, bits):
    """Second decoder: swapped LE16 words, K-bit symbols, circular history."""
    if bits not in (2, 3, 4) or isinstance(bits, bool):
        raise ValueError("EXLR8 fixture widths are K2/K3/K4")
    if not isinstance(raw, bytes) or len(raw) != bits * 32:
        raise ValueError("tile payload length/dtype does not match width")
    shorts = struct.unpack("<" + "H" * (16 * bits), raw)
    swapped = [shorts[index ^ 1] for index in range(len(shorts))]
    stream = [(word >> bit) & 1 for word in swapped for bit in range(15, -1, -1)]
    symbols = [sum(stream[index * bits + bit] << (bits - 1 - bit) for bit in range(bits))
               for index in range(256)]
    return [sum(symbols[(index - history) % 256] << (history * bits)
                for history in range((16 + bits - 1) // bits)) & 65535 for index in range(256)]


def tile_coordinates(position):
    lane, item = divmod(position, 8)
    return (2 * (lane % 4) + (item & 1) + 8 * ((item >> 1) & 1),
            lane // 4 + 8 * (item >> 2))


def decode_tile(raw, bits, codebook="mcg", exlr8=False):
    if exlr8 and (bits not in (2, 3, 4) or codebook != "mcg"):
        raise ValueError("EXLR8 fixture requires MCG at K2/K3/K4")
    if bits in (1.5, 2.5, 3.5) and codebook != "mul1":
        raise ValueError("half widths require mul1")
    table = codebook_values("exlr8" if exlr8 else codebook)
    tile = [0.0] * 256
    states = exlr8_symbol_states(raw, bits) if exlr8 else tile_states(raw, bits)
    for position, state in enumerate(states):
        row, col = tile_coordinates(position)
        tile[row * 16 + col] = table[state]
    return tile


def synthetic_raw(bits, tile_index=0):
    """Versioned integer pattern, independent of random-library versions."""
    words = _width(bits)
    return bytes((index * 73 + (index // 7) * 19 + tile_index * 31 + 0xA5) & 255
                 for index in range(words * 2))


def dense_fixture(bits, side, inp=128, out=128):
    """EXLR8 entry order in the rotated domain, returned HF (out,in).

    Scale vectors and Hadamard reconstruction are intentionally external:
    fp32 GEMM accumulation is not an exact packed-decoder byte assertion.
    """
    if bits not in (2, 3, 4) or side not in ("in", "out"):
        raise ValueError("expected K2/K3/K4 and side in/out")
    if any(type(dim) is not int or dim < 128 or dim > MAX_DIM or dim % 128 for dim in (inp, out)):
        raise ValueError("fixture dimensions must be 128 or 256")
    matrix = [0.0] * (inp * out)
    raw = bytearray()
    order = []
    if side == "out":
        order = [(r, c) for entry in range(out // 128)
                 for r in range(inp // 16) for c in range(entry * 8, entry * 8 + 8)]
    else:
        order = [(r, c) for entry in range(inp // 128) for block in range(out // 128)
                 for r in range(entry * 8, entry * 8 + 8) for c in range(block * 8, block * 8 + 8)]
    for index, (row_tile, col_tile) in enumerate(order):
        packed = synthetic_raw(bits, index)
        raw.extend(packed)
        tile = decode_tile(packed, bits, exlr8=True)
        for row in range(16):
            for col in range(16):
                matrix[(col_tile * 16 + col) * inp + row_tile * 16 + row] = tile[row * 16 + col]
    return bytes(raw), matrix
