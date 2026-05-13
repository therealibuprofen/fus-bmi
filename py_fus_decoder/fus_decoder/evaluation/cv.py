"""Cross-validation helpers with optional group-aware splitting."""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

from ..config import CrossValidationConfig
from ..utils import require_dependency


def generate_splits(
    y: Any,
    config: CrossValidationConfig,
    groups: Any = None,
) -> List[Tuple[Any, Any]]:
    model_selection = require_dependency(
        "sklearn.model_selection", 'pip install -e ".[classical]"'
    )
    np = require_dependency("numpy", 'pip install -e ".[classical]"')
    y = np.asarray(y)

    strategy = config.strategy.lower()
    if strategy == "stratified_kfold":
        splitter = model_selection.StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_seed,
        )
        return list(splitter.split(np.zeros(len(y)), y))

    if strategy == "group_kfold":
        if groups is None:
            raise ValueError("group_kfold requires groups.")
        splitter = model_selection.GroupKFold(n_splits=config.n_splits)
        return list(splitter.split(np.zeros(len(y)), y, groups))

    if strategy == "leave_one_group_out":
        if groups is None:
            raise ValueError("leave_one_group_out requires groups.")
        splitter = model_selection.LeaveOneGroupOut()
        return list(splitter.split(np.zeros(len(y)), y, groups))

    raise ValueError(f"Unsupported CV strategy: {config.strategy}")
