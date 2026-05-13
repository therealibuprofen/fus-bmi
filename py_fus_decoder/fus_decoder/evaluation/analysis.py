"""Post-hoc analysis helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import require_dependency


def analyze_prediction_slices(
    y_true: Any,
    y_pred: Any,
    metadata: Dict[str, Any],
    slice_keys: List[str],
    sample_indices: Optional[Any] = None,
) -> Dict[str, Any]:
    np = require_dependency("numpy", 'pip install -e ".[classical]"')
    results: Dict[str, Any] = {}
    sample_indices = None if sample_indices is None else np.asarray(sample_indices)
    for key in slice_keys:
        values = metadata.get(key)
        if values is None:
            continue
        values = np.asarray(values)
        if values.ndim == 0:
            values = np.repeat(values.reshape(1), len(y_true))
        elif sample_indices is not None and len(values) >= len(sample_indices):
            values = values[sample_indices]
        elif len(values) != len(y_true):
            continue
        bucket = {}
        for value in np.unique(values):
            mask = values == value
            if int(mask.sum()) == 0:
                continue
            bucket[str(value)] = {
                "count": int(mask.sum()),
                "accuracy": float((np.asarray(y_true)[mask] == np.asarray(y_pred)[mask]).mean()),
            }
        results[key] = bucket
    return results
