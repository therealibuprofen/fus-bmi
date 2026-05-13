"""Configuration dataclasses for offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetConfig:
    name: str
    path: str
    loader: str = "auto"
    sample_key: str = "samples"
    label_key: str = "labels"
    metadata_key: str = "metadata"
    group_key: Optional[str] = None
    species: Optional[str] = None
    task: Optional[str] = None
    shape_hint: Optional[List[int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass
class ModelConfig:
    name: str
    family: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationConfig:
    strategy: str = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_seed: int = 42
    group_key: Optional[str] = None


@dataclass
class OutputConfig:
    output_dir: str = "py_fus_decoder/output"
    save_fold_predictions: bool = True
    save_confusion_matrices: bool = True

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


@dataclass
class EvaluationConfig:
    experiment_name: str
    datasets: List[DatasetConfig]
    models: List[ModelConfig]
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "balanced_accuracy", "f1_macro"]
    )
    train_size_fractions: List[float] = field(default_factory=lambda: [1.0])

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationConfig":
        datasets = [DatasetConfig(**item) for item in payload["datasets"]]
        models = [ModelConfig(**item) for item in payload["models"]]
        cv = CrossValidationConfig(**payload.get("cross_validation", {}))
        output = OutputConfig(**payload.get("output", {}))
        return cls(
            experiment_name=payload["experiment_name"],
            datasets=datasets,
            models=models,
            cross_validation=cv,
            output=output,
            metrics=payload.get(
                "metrics", ["accuracy", "balanced_accuracy", "f1_macro"]
            ),
            train_size_fractions=payload.get("train_size_fractions", [1.0]),
        )
