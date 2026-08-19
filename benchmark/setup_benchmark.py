from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up local benchmark data and external UniversalFakeDetect code.")
    parser.add_argument("--target-per-label", type=int, default=500)
    parser.add_argument("--openfake-revision", default="", help="Optional Hugging Face dataset revision, tag, or commit hash.")
    parser.add_argument("--force-data", action="store_true", help="Overwrite files in benchmark/data/openfake_1k.")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-universal", action="store_true")
    return parser.parse_args()


def run_script(script_name: str, extra_args: list[str]) -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / script_name), *extra_args], check=True)


def main() -> None:
    args = parse_args()

    if not args.skip_universal:
        run_script("setup_universalfakedetect.py", [])

    if not args.skip_data:
        download_args = ["--target-per-label", str(args.target_per_label)]
        if args.openfake_revision:
            download_args.extend(["--revision", args.openfake_revision])
        if args.force_data:
            download_args.append("--force")
        run_script("download_openfake.py", download_args)

    print("Benchmark setup is ready.")


if __name__ == "__main__":
    main()
