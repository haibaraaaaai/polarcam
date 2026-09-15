"""Launch the imported live application without changing its script imports."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polarcam",
        description="Polarcam Live: camera acquisition and polarization analysis.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for recordings and analysis output (default: current directory).",
    )
    arguments = parser.parse_args(argv)
    data_dir = arguments.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    application = Path(__file__).resolve().parent / "live" / "Spinners_gui_live.py"
    result = subprocess.run(
        [sys.executable, str(application)],
        cwd=data_dir,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
