import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import PIL
import torch
import torchvision


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def main():
    parser = argparse.ArgumentParser("Capture the SRe2L++ reproduction runtime")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    payload = {
        "status": "complete",
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
        },
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_smi": command_output([
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]),
        },
        "required_environment": {
            name: os.environ.get(name)
            for name in (
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
                "TORCH_HOME",
                "LD_PRELOAD",
            )
        },
        "repository": {
            "root": str(repo_root),
            "revision": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "tracked_diff": command_output(["git", "-C", str(repo_root), "status", "--short", "--untracked-files=no"]),
        },
    }
    missing_env = [
        name for name, value in payload["required_environment"].items() if not value
    ]
    if missing_env:
        raise RuntimeError(f"missing required runtime environment: {missing_env}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
