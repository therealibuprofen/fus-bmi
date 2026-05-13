"""Offline evaluation runner."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from ..config import EvaluationConfig
from ..data import load_dataset
from ..models import build_model
from ..utils import require_dependency, save_json
from .analysis import analyze_prediction_slices
from .cv import generate_splits
from .metrics import aggregate_fold_metrics, compute_metrics


class OfflineEvaluationRunner:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        results = {
            "experiment_name": self.config.experiment_name,
            "datasets": [],
        }
        for dataset_cfg in self.config.datasets:
            dataset = load_dataset(dataset_cfg)
            dataset_result = self._run_dataset(dataset, dataset_cfg)
            results["datasets"].append(dataset_result)

        output_dir = self.config.output.resolved_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "summary.json", results)
        return results

    def _run_dataset(self, dataset: Any, dataset_cfg: Any) -> Dict[str, Any]:
        np = require_dependency("numpy", 'pip install -e ".[classical]"')
        X = dataset.samples
        y = np.asarray(dataset.labels)
        groups = None
        group_key = self.config.cross_validation.group_key or dataset_cfg.group_key
        if group_key and group_key in dataset.metadata:
            groups = np.asarray(dataset.metadata[group_key])

        splits = generate_splits(y, self.config.cross_validation, groups=groups)
        dataset_output = {
            "dataset": dataset.summary(),
            "cross_validation": asdict(self.config.cross_validation),
            "train_size_fractions": self.config.train_size_fractions,
            "models": [],
        }

        for model_cfg in self.config.models:
            if not model_cfg.enabled:
                continue
            for train_fraction in self.config.train_size_fractions:
                fold_metrics: List[Dict[str, Any]] = []
                fold_predictions: List[Dict[str, Any]] = []
                fraction_splits = self._build_fraction_train_splits(
                    splits,
                    y,
                    train_fraction,
                    self.config.cross_validation.random_seed,
                )
                for fold_idx, (train_idx, test_idx) in enumerate(fraction_splits):
                    model = build_model(model_cfg)
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])
                    metrics = compute_metrics(y[test_idx], y_pred)
                    metrics["fold_index"] = fold_idx
                    metrics["train_fraction"] = float(train_fraction)
                    metrics["n_train"] = int(len(train_idx))
                    metrics["n_test"] = int(len(test_idx))
                    fold_metrics.append(metrics)
                    fold_predictions.append(
                        {
                            "fold_index": fold_idx,
                            "train_fraction": float(train_fraction),
                            "y_true": np.asarray(y[test_idx]).tolist(),
                            "y_pred": np.asarray(y_pred).tolist(),
                            "indices": np.asarray(test_idx).tolist(),
                            "train_indices": np.asarray(train_idx).tolist(),
                        }
                    )

                all_true = require_dependency("numpy", 'pip install -e ".[classical]"').concatenate(
                    [item["y_true"] for item in fold_predictions]
                )
                all_pred = require_dependency("numpy", 'pip install -e ".[classical]"').concatenate(
                    [item["y_pred"] for item in fold_predictions]
                )
                all_indices = require_dependency("numpy", 'pip install -e ".[classical]"').concatenate(
                    [item["indices"] for item in fold_predictions]
                )
                sliced = analyze_prediction_slices(
                    all_true,
                    all_pred,
                    dataset.metadata,
                    ["species", "task", "session", "subject_id", group_key] if group_key else ["species", "task", "session", "subject_id"],
                    sample_indices=all_indices,
                )

                model_result = {
                    "name": model_cfg.name,
                    "family": model_cfg.family,
                    "params": model_cfg.params,
                    "train_fraction": float(train_fraction),
                    "fold_metrics": fold_metrics,
                    "summary": aggregate_fold_metrics(fold_metrics),
                    "slice_analysis": sliced,
                }
                if self.config.output.save_fold_predictions:
                    model_result["fold_predictions"] = fold_predictions
                dataset_output["models"].append(model_result)

        return dataset_output

    def _build_fraction_train_splits(
        self,
        splits: List[Any],
        y: Any,
        train_fraction: float,
        random_seed: int,
    ) -> List[Any]:
        np = require_dependency("numpy", 'pip install -e ".[classical]"')
        model_selection = require_dependency(
            "sklearn.model_selection", 'pip install -e ".[classical]"'
        )
        if train_fraction >= 1.0:
            return splits

        fraction_splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_idx = np.asarray(train_idx)
            y_train = np.asarray(y[train_idx])
            classes = np.unique(y_train)
            class_count = len(classes)
            min_required = max(class_count + 1, class_count * 2)
            requested = max(min_required, int(round(len(train_idx) * float(train_fraction))))
            requested = min(requested, len(train_idx))
            if requested == len(train_idx):
                fraction_splits.append((train_idx, test_idx))
                continue

            splitter = model_selection.StratifiedShuffleSplit(
                n_splits=1,
                train_size=requested,
                random_state=random_seed + fold_idx,
            )
            sub_train_local, _ = next(splitter.split(np.zeros(len(train_idx)), y_train))
            fraction_splits.append((train_idx[sub_train_local], test_idx))

        return fraction_splits
