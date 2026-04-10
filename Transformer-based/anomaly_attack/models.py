from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class AttackTypeTransformer(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(num_sensors, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        h = self.input_proj(x)
        h = h + self.pos_embed[:, :seq_len, :]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_stage2_model(num_sensors: int, num_classes: int, params: Dict) -> AttackTypeTransformer:
    return AttackTypeTransformer(
        num_sensors=num_sensors,
        num_classes=num_classes,
        d_model=int(params.get("d_model", 128)),
        nhead=int(params.get("nhead", 4)),
        num_layers=int(params.get("num_layers", 3)),
        dim_feedforward=int(params.get("dim_feedforward", 256)),
        dropout=float(params.get("dropout", 0.2)),
        max_seq_len=int(params.get("max_seq_len", 512)),
    )


def compute_class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def fit_stage2_model(
    model: AttackTypeTransformer,
    x_train: np.ndarray,
    y_train_idx: np.ndarray,
    params: Dict,
    random_state: int,
) -> None:
    set_global_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(params.get("batch_size", 128))
    epochs = int(params.get("epochs", 40))
    learning_rate = float(params.get("learning_rate", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-5))
    use_balanced_loss = bool(params.get("use_balanced_loss", True))

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_idx, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    if use_balanced_loss:
        class_weights = compute_class_weights(y_train_idx, int(y_train_idx.max()) + 1).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def predict_stage2(model: AttackTypeTransformer, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def evaluate_classifier(y_true, y_pred) -> Dict:
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
