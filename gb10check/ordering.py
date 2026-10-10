# SPDX-License-Identifier: MIT
# Copyright 2026 knapcio
"""Configurable static instruction checks adapted from the GB10 kernel labs.

Linear SASS checks are diagnostics. The PTX barrier cut checks all represented
control-flow paths, conservatively treating predicate barriers as non-cuts.
Neither establishes scratch ownership, global synchronization or GPU safety.
"""
import re


def sequence(text, patterns):
    offset = 0
    for pattern in patterns:
        match = re.search(pattern, text[offset:])
        if match is None:
            return False
        offset += match.end()
    return True


def instruction_checks(text, specification, baseline=None):
    checks = {}
    for index, pattern in enumerate(specification.get("required", [])):
        checks[f"required_{index}"] = re.search(pattern, text) is not None
    for index, pattern in enumerate(specification.get("forbidden", [])):
        checks[f"forbidden_{index}"] = re.search(pattern, text) is None
    for index, patterns in enumerate(specification.get("sequences", [])):
        checks[f"sequence_{index}"] = bool(patterns) and sequence(text, patterns)
    for index, pattern in enumerate(specification.get("unchanged_counts", [])):
        if baseline is None:
            raise ValueError("instruction count comparison needs a baseline")
        checks[f"count_{index}"] = len(re.findall(pattern, text)) == len(re.findall(pattern, baseline))
    return checks


def barrier_cut(ptx, scratch_parameter, reset_parameter=None):
    """Check scratch read -> discard paths and conservative pointer provenance.

    Restrict input to one entry without calls or indirect branches. This is a
    may-alias analysis: pointer writes from unrelated inputs invalidate the
    provenance proof. A reset check is local ordering, not a dominance proof.
    """
    if len(re.findall(r"\.entry\b", ptx)) != 1 or re.search(r"\b(?:call|brx)(?:\.|\s)", ptx):
        raise ValueError("barrier cut requires one entry and direct local control flow")
    instructions, labels = [], {}
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith("."):
            continue
        if re.fullmatch(r"[\w$]+:", line):
            if line[:-1] in labels:
                raise ValueError("duplicate PTX label")
            labels[line[:-1]] = len(instructions)
        elif line.endswith(";"):
            instructions.append(line.strip("{} "))
    derived, resets, other_writes = set(), set(), set()
    for _ in range(len(instructions) + 1):
        before = (derived.copy(), resets.copy(), other_writes.copy())
        for line in instructions:
            regs = re.findall(r"%rd\d+", line)
            if not regs:
                continue
            op = re.sub(r"^@!?%\w+\s+", "", line)
            if op.startswith("ld.param.u64"):
                if re.search(rf"_param_{scratch_parameter}\]", op):
                    derived.add(regs[0])
                elif reset_parameter is not None and re.search(rf"_param_{reset_parameter}\]", op):
                    resets.add(regs[0])
                else:
                    other_writes.add(regs[0])
            elif re.match(r"(?:add|sub|mul|shl|mov|cvta|cvt)\.", op):
                if any(reg in derived for reg in regs[1:]):
                    derived.add(regs[0])
                elif any(reg in resets for reg in regs[1:]):
                    resets.add(regs[0])
                else:
                    other_writes.add(regs[0])
            elif re.match(r"\S+\s+%rd\d+", op) and not re.match(r"(?:st|discard|bra|bar)\.", op):
                other_writes.add(regs[0])
        if before == (derived, resets, other_writes):
            break
    # Reject ambiguous scratch registers, including conditional overwrites.
    if derived & other_writes:
        raise ValueError("ambiguous scratch address provenance")
    reads, discards, cuts = set(), set(), set()
    reverse = [[] for _ in instructions]
    for index, line in enumerate(instructions):
        addresses = re.findall(r"\[(%rd\d+)", line)
        if "ld.global" in line and any(reg in derived for reg in addresses):
            reads.add(index)
        if "discard.global.L2" in line:
            if len(addresses) != 1 or addresses[0] not in derived:
                raise ValueError("discard address does not derive from scratch parameter")
            discards.add(index)
        if re.match(r"bar.sync\s", line):
            cuts.add(index)
        branch = re.search(r"\bbra(?:\.uni)?\s+([\w$]+)", line)
        targets = []
        if branch:
            if branch[1] not in labels or labels[branch[1]] >= len(instructions):
                raise ValueError("unresolved branch target")
            targets.append(labels[branch[1]])
        conditional = line.startswith("@")
        terminal = re.match(r"(?:ret|exit);", line)
        if index + 1 < len(instructions) and (conditional or not branch and not terminal):
            targets.append(index + 1)
        for target in targets:
            reverse[target].append(index)
    if not reads or not discards:
        raise ValueError("scratch reads and discards must both be identifiable")
    pending, seen = list(discards), set()
    while pending:
        index = pending.pop()
        if index in seen or index in cuts:
            continue
        seen.add(index)
        pending.extend(reverse[index])
    checks = {"scratch_read_barrier_cut": not bool(reads & seen)}
    if reset_parameter is not None:
        tail = instructions[max(discards) + 1:]
        checks["reset_follows_discard"] = any("st.global" in line and
            any(reg in resets for reg in re.findall(r"\[(%rd\d+)", line)) for line in tail)
    return {"checks": checks, "scratch_loads": len(reads), "discards": len(discards),
            "scope": "represented PTX paths; no ownership or memory-model proof"}
