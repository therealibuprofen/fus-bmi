"""Classical fUS decoders: linear, PCA+LDA, cPCA+LDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import DecoderModel
from ..utils import require_dependency


def _to_2d_array(X: Any) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[classical]"')
    X = np.asarray(X)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    X = X.astype(np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


@dataclass
class LinearDecoder(DecoderModel):
    params: Optional[Dict[str, Any]] = None
    family: str = "linear"

    def __post_init__(self) -> None:
        params = dict(self.params or {})
        sk_pipeline = require_dependency(
            "sklearn.pipeline", 'pip install -e ".[classical]"'
        )
        sk_preprocessing = require_dependency(
            "sklearn.preprocessing", 'pip install -e ".[classical]"'
        )
        sk_linear = require_dependency(
            "sklearn.linear_model", 'pip install -e ".[classical]"'
        )
        self.estimator = sk_pipeline.Pipeline(
            steps=[
                ("flatten", FunctionTransformer2D()),
                ("scale", sk_preprocessing.StandardScaler()),
                (
                    "logreg",
                    sk_linear.LogisticRegression(
                        max_iter=params.pop("max_iter", 1000),
                        class_weight=params.pop("class_weight", "balanced"),
                        solver=params.pop("solver", "lbfgs"),
                        **params,
                    ),
                ),
            ]
        )

    def fit(self, X: Any, y: Any) -> "LinearDecoder":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator.predict(X)


class ClasswisePCATransformer:
    """Simple class-wise PCA feature map.

    For each class, fit a PCA subspace and project each sample onto it.
    Concatenated projections are passed to LDA.
    """

    def __init__(
        self,
        n_components: Any = 0.95,
        whiten: bool = False,
        max_components_per_class: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.max_components_per_class = max_components_per_class
        self.class_to_pca = {}
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "ClasswisePCATransformer":
        np = require_dependency("numpy", 'pip install -e ".[classical]"')
        sk_decomp = require_dependency(
            "sklearn.decomposition", 'pip install -e ".[classical]"'
        )
        X = _to_2d_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_pca = {}
        for cls in self.classes_:
            X_cls = X[y == cls]
            n_cls_samples, n_features = X_cls.shape
            max_rank = max(1, min(n_cls_samples - 1, n_features))
            n_components = self._resolve_n_components(max_rank)
            pca = sk_decomp.PCA(
                n_components=n_components,
                whiten=self.whiten,
                svd_solver="full",
            )
            pca.fit(X_cls)
            self.class_to_pca[cls] = pca
        return self

    def transform(self, X: Any) -> Any:
        np = require_dependency("numpy", 'pip install -e ".[classical]"')
        X = _to_2d_array(X)
        blocks = [self.class_to_pca[cls].transform(X) for cls in self.classes_]
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X: Any, y: Any) -> Any:
        return self.fit(X, y).transform(X)

    def _resolve_n_components(self, max_rank: int) -> Any:
        if isinstance(self.n_components, float):
            return self.n_components
        if self.max_components_per_class is None:
            return min(int(self.n_components), max_rank)
        return min(int(self.n_components), max_rank, int(self.max_components_per_class))


@dataclass
class PCALDADecoder(DecoderModel):
    params: Optional[Dict[str, Any]] = None
    family: str = "pca_lda"

    def __post_init__(self) -> None:
        params = dict(self.params or {})
        sk_pipeline = require_dependency(
            "sklearn.pipeline", 'pip install -e ".[classical]"'
        )
        sk_preprocessing = require_dependency(
            "sklearn.preprocessing", 'pip install -e ".[classical]"'
        )
        sk_decomp = require_dependency(
            "sklearn.decomposition", 'pip install -e ".[classical]"'
        )
        sk_da = require_dependency(
            "sklearn.discriminant_analysis", 'pip install -e ".[classical]"'
        )
        lda_solver = params.pop("solver", "lsqr")
        lda_shrinkage = params.pop("shrinkage", "auto" if lda_solver in {"lsqr", "eigen"} else None)
        self.estimator = sk_pipeline.Pipeline(
            steps=[
                ("flatten", FunctionTransformer2D()),
                ("scale", sk_preprocessing.StandardScaler()),
                (
                    "pca",
                    sk_decomp.PCA(
                        n_components=params.pop("n_components", 0.95),
                        whiten=params.pop("whiten", False),
                        svd_solver=params.pop("svd_solver", "full"),
                    ),
                ),
                (
                    "lda",
                    sk_da.LinearDiscriminantAnalysis(
                        solver=lda_solver,
                        shrinkage=lda_shrinkage,
                        **params,
                    ),
                ),
            ]
        )

    def fit(self, X: Any, y: Any) -> "PCALDADecoder":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator.predict(X)


@dataclass
class CPCALDADecoder(DecoderModel):
    params: Optional[Dict[str, Any]] = None
    family: str = "cpca_lda"

    def __post_init__(self) -> None:
        params = dict(self.params or {})
        sk_preprocessing = require_dependency(
            "sklearn.preprocessing", 'pip install -e ".[classical]"'
        )
        sk_da = require_dependency(
            "sklearn.discriminant_analysis", 'pip install -e ".[classical]"'
        )
        self.scaler = sk_preprocessing.StandardScaler()
        self.cpca = ClasswisePCATransformer(
            n_components=params.pop("n_components", 0.95),
            whiten=params.pop("whiten", False),
            max_components_per_class=params.pop("max_components_per_class", None),
        )
        lda_solver = params.pop("solver", "lsqr")
        lda_shrinkage = params.pop("shrinkage", "auto" if lda_solver in {"lsqr", "eigen"} else None)
        self.lda = sk_da.LinearDiscriminantAnalysis(
            solver=lda_solver,
            shrinkage=lda_shrinkage,
            **params,
        )

    def fit(self, X: Any, y: Any) -> "CPCALDADecoder":
        X2 = self.scaler.fit_transform(_to_2d_array(X))
        Xcpca = self.cpca.fit_transform(X2, y)
        self.lda.fit(Xcpca, y)
        return self

    def predict(self, X: Any) -> Any:
        X2 = self.scaler.transform(_to_2d_array(X))
        Xcpca = self.cpca.transform(X2)
        return self.lda.predict(Xcpca)


class FunctionTransformer2D:
    """Tiny sklearn-compatible flatten transformer to avoid hard dependency at import time."""

    def fit(self, X: Any, y: Any = None) -> "FunctionTransformer2D":
        return self

    def transform(self, X: Any) -> Any:
        return _to_2d_array(X)
