"""Model registry and construction helpers."""

from __future__ import annotations

from ..config import ModelConfig
from .classical import CPCALDADecoder, LinearDecoder, PCALDADecoder
from .deep import CNNDecoder, CNNLSTMDecoder


MODEL_REGISTRY = {
    "linear": LinearDecoder,
    "pca_lda": PCALDADecoder,
    "cpca_lda": CPCALDADecoder,
    "cnn": CNNDecoder,
    "cnn_lstm": CNNLSTMDecoder,
}


def build_model(config: ModelConfig):
    family = config.family.lower()
    if family not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model family: {config.family}")
    return MODEL_REGISTRY[family](params=config.params)
