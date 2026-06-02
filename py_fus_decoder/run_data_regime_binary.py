#!/usr/bin/env python3
"""One-click binary data-regime experiment."""

import sys
from pathlib import Path

from fus_decoder.cli import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parent / "configs" / "data_regime_binary_experiment.json"
    sys.argv = [sys.argv[0], "--config", str(config), *sys.argv[1:]]
    main()
