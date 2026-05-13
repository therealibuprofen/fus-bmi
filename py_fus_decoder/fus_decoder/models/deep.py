"""PyTorch-based deep decoders for fUS volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import DecoderModel
from ..utils import require_dependency


def _prepare_volume_tensor(X: Any) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[deep]"')
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        raise ValueError("Deep decoders require volume input [N, H, W, T] or shape_hint-expanded data.")
    if X.ndim != 4:
        raise ValueError(f"Unsupported input ndim for deep decoder: {X.ndim}")
    return X


class _TorchDecoderMixin(DecoderModel):
    family: str = "deep"

    def _require_torch(self) -> Any:
        return require_dependency("torch", 'pip install -e ".[deep]"')

    def _require_numpy(self) -> Any:
        return require_dependency("numpy", 'pip install -e ".[deep]"')

    def _encode_labels(self, y: Any) -> Tuple[Any, Any]:
        np = self._require_numpy()
        y = np.asarray(y)
        classes = np.unique(y)
        encoded = np.searchsorted(classes, y)
        return encoded.astype("int64"), classes

    def _build_loader(self, X: Any, y: Any, batch_size: int, shuffle: bool) -> Any:
        torch = self._require_torch()
        X = _prepare_volume_tensor(X)
        y_encoded, _ = self._encode_labels(y)
        tensor_x = torch.from_numpy(X).permute(0, 3, 1, 2).unsqueeze(1)
        tensor_y = torch.from_numpy(y_encoded)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _predict_batches(self, X: Any) -> Any:
        torch = self._require_torch()
        np = self._require_numpy()
        X = _prepare_volume_tensor(X)
        tensor_x = torch.from_numpy(X).permute(0, 3, 1, 2).unsqueeze(1)
        loader = torch.utils.data.DataLoader(tensor_x, batch_size=self.batch_size, shuffle=False)
        preds = []
        self.network.eval()
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.network(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        pred_idx = np.concatenate(preds, axis=0)
        return self.classes_[pred_idx]


class FusCNNNet:
    def __init__(self, input_shape: Tuple[int, int, int], n_classes: int, params: Dict[str, Any]) -> None:
        torch = require_dependency("torch", 'pip install -e ".[deep]"')
        nn = torch.nn
        channels = params.get("channels", [4, 8])
        dropout = float(params.get("dropout", 0.15))
        kernel_size = int(params.get("kernel_size", 3))
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.extend(
                [
                    nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=(1, 2, 2)),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[2], input_shape[0], input_shape[1])
            feat_dim = int(self.features(dummy).flatten(1).shape[1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, params.get("hidden_dim", 32)),
            nn.ReLU(inplace=True),
            nn.Linear(params.get("hidden_dim", 32), n_classes),
        )
        self._nn = nn

    def module(self) -> Any:
        nn = self._nn
        outer_self = self

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = outer_self.features
                self.classifier = outer_self.classifier

            def forward(self, x: Any) -> Any:
                return self.classifier(self.features(x))

        return _Net()


class FusCNNLSTMNet:
    def __init__(self, input_shape: Tuple[int, int, int], n_classes: int, params: Dict[str, Any]) -> None:
        torch = require_dependency("torch", 'pip install -e ".[deep]"')
        nn = torch.nn
        channels = params.get("channels", [4, 8])
        dropout = float(params.get("dropout", 0.15))
        kernel_size = int(params.get("kernel_size", 3))
        spatial = []
        in_ch = 1
        for out_ch in channels:
            spatial.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch
        self.spatial_encoder = nn.Sequential(*spatial)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            feat_dim = int(self.spatial_encoder(dummy).flatten(1).shape[1])
        hidden_dim = int(params.get("hidden_dim", 32))
        num_layers = int(params.get("num_layers", 1))
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self._nn = nn

    def module(self) -> Any:
        nn = self._nn
        outer_self = self

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spatial_encoder = outer_self.spatial_encoder
                self.lstm = outer_self.lstm
                self.classifier = outer_self.classifier

            def forward(self, x: Any) -> Any:
                # x: [B, 1, T, H, W]
                bsz, _, tsteps, height, width = x.shape
                x = x.view(bsz * tsteps, 1, height, width)
                x = self.spatial_encoder(x).flatten(1)
                x = x.view(bsz, tsteps, -1)
                _, (hidden, _) = self.lstm(x)
                return self.classifier(hidden[-1])

        return _Net()


@dataclass
class CNNDecoder(_TorchDecoderMixin):
    params: Optional[Dict[str, Any]] = None
    family: str = "cnn"

    def fit(self, X: Any, y: Any) -> "CNNDecoder":
        torch = self._require_torch()
        np = self._require_numpy()
        X = _prepare_volume_tensor(X)
        y_encoded, classes = self._encode_labels(y)
        self.classes_ = classes
        self.batch_size = int((self.params or {}).get("batch_size", 8))
        self.epochs = int((self.params or {}).get("epochs", 12))
        lr = float((self.params or {}).get("learning_rate", 1e-3))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = self._build_loader(X, y, self.batch_size, shuffle=True)
        net_builder = FusCNNNet((X.shape[1], X.shape[2], X.shape[3]), len(classes), self.params or {})
        self.network = net_builder.module().to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        self.network.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: Any) -> Any:
        return self._predict_batches(X)


@dataclass
class CNNLSTMDecoder(_TorchDecoderMixin):
    params: Optional[Dict[str, Any]] = None
    family: str = "cnn_lstm"

    def fit(self, X: Any, y: Any) -> "CNNLSTMDecoder":
        torch = self._require_torch()
        X = _prepare_volume_tensor(X)
        y_encoded, classes = self._encode_labels(y)
        self.classes_ = classes
        self.batch_size = int((self.params or {}).get("batch_size", 8))
        self.epochs = int((self.params or {}).get("epochs", 12))
        lr = float((self.params or {}).get("learning_rate", 1e-3))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = self._build_loader(X, y, self.batch_size, shuffle=True)
        net_builder = FusCNNLSTMNet((X.shape[1], X.shape[2], X.shape[3]), len(classes), self.params or {})
        self.network = net_builder.module().to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        self.network.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: Any) -> Any:
        return self._predict_batches(X)
