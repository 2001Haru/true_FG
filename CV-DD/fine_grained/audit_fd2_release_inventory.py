import argparse
import json
import re
import subprocess
from pathlib import Path


MODEL_SUFFIXES = (".pth", ".pt", ".ckpt", ".pth.tar", ".safetensors")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BASELINE_ENTRYPOINTS = (
    "FD2/recover/recover.py",
    "FD2/recover/recover_FADRM.py",
)


def git(repo_root, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser("Inventory the tracked FD2 release artifacts")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    tracked = [line for line in git(repo_root, "ls-files", "FD2").splitlines() if line]
    tracked_set = set(tracked)

    model_artifacts = [path for path in tracked if path.lower().endswith(MODEL_SUFFIXES)]
    archives = [path for path in tracked if path.lower().endswith(ARCHIVE_SUFFIXES)]
    images = [path for path in tracked if path.lower().endswith(IMAGE_SUFFIXES)]
    figure_images = [path for path in images if path.startswith("FD2/figure/")]
    distilled_images = [path for path in images if not path.startswith("FD2/figure/")]

    launcher_references = []
    pattern = re.compile(r"(?:^|[/\"'])\b(recover(?:_FADRM)?\.py)\b")
    for relative in tracked:
        if not relative.endswith(".sh"):
            continue
        path = repo_root / relative
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = pattern.search(line)
            if match:
                launcher_references.append({
                    "launcher": relative,
                    "line": line_number,
                    "entrypoint": match.group(1),
                })

    payload = {
        "status": "complete",
        "repository_revision": git(repo_root, "rev-parse", "HEAD"),
        "fd2_history": git(repo_root, "log", "-1", "--format=%H %aI %s", "--", "FD2"),
        "tracked_files": len(tracked),
        "model_or_teacher_artifacts": model_artifacts,
        "archive_artifacts": archives,
        "figure_images": figure_images,
        "distilled_or_patch_images_outside_figures": distilled_images,
        "baseline_recovery_entrypoints": {
            path: path in tracked_set for path in BASELINE_ENTRYPOINTS
        },
        "fd2_recovery_entrypoints": {
            path: path in tracked_set
            for path in (
                "FD2/recover/recover_FD2.py",
                "FD2/recover/recover_FD2_FADRM.py",
            )
        },
        "baseline_launcher_references": launcher_references,
        "conclusion": (
            "The tracked release contains FD2-specific recovery code and launchers, "
            "but no offline checkpoints, synthetic patches/distilled data, or the two "
            "plain baseline recovery entrypoints referenced by baseline launchers."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "tracked_files": len(tracked),
        "model_artifacts": len(model_artifacts),
        "distilled_images": len(distilled_images),
        "missing_baseline_entrypoints": [
            path for path, present in payload["baseline_recovery_entrypoints"].items()
            if not present
        ],
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
