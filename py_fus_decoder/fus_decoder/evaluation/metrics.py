"""Evaluation metrics and summaries."""

from __future__ import annotations

from typing import Any, Dict, List

from ..utils import require_dependency


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    metrics = require_dependency("sklearn.metrics", 'pip install -e ".[classical]"')
    np = require_dependency("numpy", 'pip install -e ".[classical]"')
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = {}
    for idx, label in enumerate(labels):
        denom = cm[idx, :].sum()
        per_class_acc[str(label)] = float(cm[idx, idx] / denom) if denom else 0.0
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(),
        "per_class_accuracy": per_class_acc,
    }


def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    np = require_dependency("numpy", 'pip install -e ".[classical]"')
    scalar_keys = ["accuracy", "balanced_accuracy", "f1_macro"]
    summary = {}
    for key in scalar_keys:
        values = [fold[key] for fold in fold_metrics]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return summary
