"""Command line entrypoint for offline fUS evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import EvaluationConfig
from .evaluation import OfflineEvaluationRunner
from .utils import choose_dataset_path_gui, load_json, maybe_load_yaml_or_json


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "session_smoke_test.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SESSION_FILE_PATTERN = re.compile(r"rt_fUS_data_S(?P<session>\d+)_R(?P<run>\d+)\.mat$")
PROJECT_RECORD = PACKAGE_ROOT / "dataset" / "project_record.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline benchmark for fUS decoders.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to JSON or YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--gui-select-dataset",
        action="store_true",
        default=True,
        help="Open a GUI dialog to choose dataset path(s) and override config paths.",
    )
    parser.add_argument(
        "--no-gui-select-dataset",
        dest="gui_select_dataset",
        action="store_false",
        help="Use dataset path(s) from config without opening a GUI dialog.",
    )
    return parser


def resolve_config_path(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        PROJECT_ROOT / path,
        PACKAGE_ROOT / path,
        config_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((PROJECT_ROOT / path).resolve())


def normalize_payload_paths(payload: dict, config_path: Path) -> None:
    for dataset_cfg in payload.get("datasets", []):
        if dataset_cfg.get("path"):
            dataset_cfg["path"] = resolve_config_path(dataset_cfg["path"], config_path)

    output_cfg = payload.get("output", {})
    if output_cfg.get("output_dir"):
        output_cfg["output_dir"] = resolve_config_path(output_cfg["output_dir"], config_path)


def infer_session_run(path: str) -> Optional[Tuple[int, int]]:
    match = SESSION_FILE_PATTERN.search(Path(path).name)
    if not match:
        return None
    return int(match.group("session")), int(match.group("run"))


def load_session_record(session_id: int, run_id: int) -> Optional[Dict[str, Any]]:
    if not PROJECT_RECORD.exists():
        return None
    records = load_json(PROJECT_RECORD)
    for item in records:
        if int(item.get("Session", -1)) == session_id and int(item.get("Run", -1)) == run_id:
            return item
    return None


def apply_selected_dataset_defaults(payload: dict, dataset_cfg: dict, selected_path: str) -> None:
    session_run = infer_session_run(selected_path)
    if session_run is None:
        dataset_cfg["path"] = selected_path
        return

    session_id, run_id = session_run
    record = load_session_record(session_id, run_id) or {}
    n_targets = int(record.get("nTargets", 0) or 0)
    task = str(record.get("Task", "session")).lower()
    suffix = f"s{session_id}_r{run_id}"

    dataset_cfg["path"] = selected_path
    dataset_cfg["name"] = f"{suffix}_{n_targets}target_session" if n_targets else f"{suffix}_session"
    dataset_cfg["task"] = f"{task}_{n_targets}target" if n_targets else task
    payload["experiment_name"] = f"session_smoke_test_{suffix}_{n_targets}target" if n_targets else f"session_smoke_test_{suffix}"

    extra = dataset_cfg.setdefault("extra", {})
    if n_targets == 2:
        extra["allowed_labels"] = [1, 2]
        extra["max_samples"] = 256
    elif n_targets == 8:
        extra["allowed_labels"] = [1, 2, 3, 4, 6, 7, 8, 9]
        extra["max_samples"] = 104
    else:
        extra.pop("allowed_labels", None)

    output_cfg = payload.setdefault("output", {})
    output_cfg["output_dir"] = str(PACKAGE_ROOT / "output" / payload["experiment_name"])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = maybe_load_yaml_or_json(config_path)
    normalize_payload_paths(payload, config_path)
    if args.gui_select_dataset:
        datasets = payload.get("datasets", [])
        for idx, dataset_cfg in enumerate(datasets):
            current_path = dataset_cfg.get("path", "")
            initialdir = str(Path(current_path).expanduser().resolve().parent) if current_path else str(Path.cwd())
            selected_path = choose_dataset_path_gui(
                title=f"Select dataset for {dataset_cfg.get('name', f'dataset_{idx + 1}')}",
                initialdir=initialdir,
            )
            apply_selected_dataset_defaults(payload, dataset_cfg, selected_path)
            print(f"Selected dataset: {dataset_cfg['name']} -> {dataset_cfg['path']}")
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
