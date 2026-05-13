#!/usr/bin/env python3
"""Thin wrapper around fus_decoder.cli."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fus_decoder.cli import main


if __name__ == "__main__":
    main()
