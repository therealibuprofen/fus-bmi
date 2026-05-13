"""Common model interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DecoderArtifact:
    model_name: str
    family: str
    estimator: Any
    config: Dict[str, Any] = field(default_factory=dict)


class DecoderModel:
    family: str = "base"

    def fit(self, X: Any, y: Any) -> "DecoderModel":
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def get_artifact(self) -> DecoderArtifact:
        return DecoderArtifact(
            model_name=self.__class__.__name__,
            family=self.family,
            estimator=self,
        )
