from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect.git"
DEFAULT_COMMIT = "76a0e3e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or update UniversalFakeDetect for benchmark inference.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--target-dir", default="benchmark/external/UniversalFakeDetect")
    return parser.parse_args()


def run_git(args: list[str], cwd: Path | None = None) -> None:
    command = ["git", *args]
    subprocess.run(command, cwd=cwd, check=True)


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def setup_repo(repo_url: str, commit: str, target_dir: Path) -> None:
    if target_dir.exists():
        if not is_git_repo(target_dir):
            raise RuntimeError(f"{target_dir} exists but is not a git repo. Move it away or remove it first.")
        run_git(["fetch", "--all", "--prune"], cwd=target_dir)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", repo_url, str(target_dir)])

    run_git(["checkout", commit], cwd=target_dir)


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir)
    setup_repo(args.repo_url, args.commit, target_dir)

    ckpt_path = target_dir / "pretrained_weights" / "fc_weights.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing UniversalFakeDetect checkpoint: {ckpt_path}")

    print(f"UniversalFakeDetect is ready: {target_dir}")
    print(f"Commit: {args.commit}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
