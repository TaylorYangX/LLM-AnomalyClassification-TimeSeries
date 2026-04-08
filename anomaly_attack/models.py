from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix


class AttackTypeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_stage2_model(input_dim: int, num_classes: int, params: Dict) -> AttackTypeMLP:
    hidden_dim = int(params.get("hidden_dim", 256))
    dropout = float(params.get("dropout", 0.2))
    return AttackTypeMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def compute_class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def fit_stage2_model(
    model: AttackTypeMLP,
    x_train: np.ndarray,
    y_train_idx: np.ndarray,
    params: Dict,
    random_state: int,
) -> None:
    set_global_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(params.get("batch_size", 256))
    epochs = int(params.get("epochs", 50))
    learning_rate = float(params.get("learning_rate", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-5))
    use_balanced_loss = bool(params.get("use_balanced_loss", True))

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_idx, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if use_balanced_loss:
        class_weights = compute_class_weights(y_train_idx, int(y_train_idx.max()) + 1).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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


def predict_stage2(model: AttackTypeMLP, x: np.ndarray) -> np.ndarray:
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
