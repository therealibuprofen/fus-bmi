"""Command line entrypoint for offline fUS evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvaluationConfig
from .evaluation import OfflineEvaluationRunner
from .utils import maybe_load_yaml_or_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline benchmark for fUS decoders.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = maybe_load_yaml_or_json(Path(args.config).expanduser().resolve())
    config = EvaluationConfig.from_dict(payload)
    runner = OfflineEvaluationRunner(config)
    results = runner.run()
    print(f"Experiment: {results['experiment_name']}")
    for dataset_result in results["datasets"]:
        dataset_name = dataset_result["dataset"]["name"]
        print(f"Dataset: {dataset_name}")
        for model_result in dataset_result["models"]:
            acc = model_result["summary"]["accuracy"]["mean"]
            bal = model_result["summary"]["balanced_accuracy"]["mean"]
            f1 = model_result["summary"]["f1_macro"]["mean"]
            print(
                f"  - {model_result['name']} [{model_result['family']}] "
                f"(train_fraction={model_result['train_fraction']:.2f}): "
                f"acc={acc:.4f}, bal_acc={bal:.4f}, f1_macro={f1:.4f}"
            )


if __name__ == "__main__":
    main()
