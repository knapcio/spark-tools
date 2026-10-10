# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import subprocess
from common.config import dump, load
from gb10check.resources import check_resources, parse_cuobjdump, parse_ptxas
from gb10check.ordering import barrier_cut, instruction_checks


def inspect(config, directory):
    directory = Path(directory)
    resources = parse_ptxas((directory / config["ptxas"]).read_text())
    binary = parse_cuobjdump((directory / config["binary_resources"]).read_text()) if config.get("binary_resources") else None
    report = check_resources(resources, config, binary)
    checks = {}
    for kind in ("ptx", "sass"):
        spec = config.get("ordering", {}).get(kind)
        if spec:
            text = (directory / config[kind]).read_text()
            baseline = (directory / config["baseline_" + kind]).read_text() if config.get("baseline_" + kind) else None
            checks[kind] = instruction_checks(text, spec, baseline)
    if config.get("barrier_cut"):
        cut = barrier_cut((directory / config["ptx"]).read_text(), **config["barrier_cut"])
        checks["barrier_cut"] = cut["checks"]
        report["barrier_cut"] = cut
    report["ordering"] = checks
    if any(not passed for group in checks.values() for passed in group.values()):
        report["status"] = "FAIL"
    return report


def container_command(config, image, output, runtime="docker", platform="linux/arm64"):
    config, output = Path(config).resolve(), Path(output).resolve()
    root = Path(__file__).resolve().parent.parent
    if output == root or root in output.parents or config.parent == output or config.parent in output.parents:
        raise ValueError("compiler output must be outside the checkout and inputs")
    if not image or image.startswith("-"):
        raise ValueError("a toolkit image is required")
    # Mount only the Python packages and inputs, so an installed wheel also
    # works without exposing host-specific dependencies to the Linux image.
    # No device or network access is requested.
    return [runtime, "run", "--rm", "--pull", "never", "--platform", platform, "--network", "none",
            "--mount", f"type=bind,source={root / 'gb10check'},target=/tool/gb10check,readonly",
            "--mount", f"type=bind,source={root / 'common'},target=/tool/common,readonly",
            "--mount", f"type=bind,source={config.parent},target=/inputs,readonly",
            "--mount", f"type=bind,source={output},target=/out",
            "--env", "PYTHONPATH=/tool", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint", "python3", image, "-m", "gb10check.backend",
            "--config", "/inputs/" + config.name, "--out", "/out"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static GB10 resource and instruction checks")
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse", help="Saved compiler output inspection")
    parse.add_argument("--config", type=Path, required=True)
    parse.add_argument("--out", type=Path)
    compile_parser = commands.add_parser("compile", help="GPU-free Linux container compilation")
    compile_parser.add_argument("--config", type=Path, required=True)
    compile_parser.add_argument("--image", required=True)
    compile_parser.add_argument("--platform", choices=["linux/arm64", "linux/amd64"], default="linux/arm64")
    compile_parser.add_argument("--runtime", default="docker", choices=["docker", "podman"])
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    try:
        if args.command == "parse":
            report = inspect(load(args.config), args.config.parent)
            dump(report, args.out)
            return int(report["status"] != "PASS")
        if args.timeout <= 0:
            raise ValueError("positive timeout required")
        command = container_command(args.config, args.image, args.out, args.runtime, args.platform)
        args.out.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, timeout=args.timeout, check=False)
        return proc.returncode
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
