#!/usr/bin/env python3
"""Read-only host/container readiness checks with actionable failures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, output


def check_executable(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    return (path is not None, path or f"{name} not found on PATH")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-docker", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    checks: dict[str, dict[str, object]] = {}
    checks["python"] = {"ok": sys.version_info >= (3, 9), "detail": sys.version.split()[0]}
    checks["wsl_dxg"] = {
        "ok": Path("/dev/dxg").exists(),
        "detail": "/dev/dxg present" if Path("/dev/dxg").exists() else "/dev/dxg missing",
    }
    gpu_ok, gpu_detail = run(["nvidia-smi", "-L"])
    checks["gpu"] = {"ok": gpu_ok, "detail": gpu_detail or "nvidia-smi returned no output"}
    docker_exists, docker_path = check_executable("docker")
    docker_ok, docker_detail = (False, docker_path)
    if docker_exists:
        docker_ok, docker_detail = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    checks["docker"] = {"ok": docker_ok, "detail": docker_detail}

    if args.as_json:
        print(json.dumps(checks, indent=2))
    else:
        for name, result in checks.items():
            marker = "PASS" if result["ok"] else "FAIL"
            detail = str(result["detail"]).replace("\n", " | ")
            print(f"[{marker}] {name}: {detail}")

    failed_required = (
        (args.require_gpu and not gpu_ok)
        or (args.require_docker and not docker_ok)
        or not bool(checks["python"]["ok"])
    )
    return 1 if failed_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
