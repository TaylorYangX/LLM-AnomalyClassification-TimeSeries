from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _window_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    if len(values) < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def extract_window_features(
    window_df: pd.DataFrame,
    sensor_cols: List[str],
    include_slope: bool = True,
) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    for col in sensor_cols:
        arr = window_df[col].to_numpy(dtype=float)
        feat[f"{col}__mean"] = float(np.mean(arr))
        feat[f"{col}__std"] = float(np.std(arr))
        feat[f"{col}__min"] = float(np.min(arr))
        feat[f"{col}__max"] = float(np.max(arr))
        feat[f"{col}__delta"] = float(arr[-1] - arr[0])
        if include_slope:
            feat[f"{col}__slope"] = _window_slope(arr)
    return feat


def build_window_dataset(
    df: pd.DataFrame,
    sensor_cols: List[str],
    binary_label_col: str,
    binary_attack_value: int | str,
    normal_label_name: str,
    window_size: int,
    stride: int,
    min_attack_ratio_for_type: float,
    include_slope: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows: List[Dict[str, float]] = []
    y_stage1: List[int] = []
    y_stage2: List[str] = []

    n = len(df)
    if n < window_size:
        raise ValueError("Dataset shorter than window_size.")

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        w = df.iloc[start:end]

        features = extract_window_features(w, sensor_cols, include_slope=include_slope)
        rows.append(features)

        attack_mask = w[binary_label_col] == binary_attack_value
        attack_ratio = float(attack_mask.mean())
        is_attack = int(attack_ratio > 0.0)
        y_stage1.append(is_attack)

        if attack_ratio < min_attack_ratio_for_type:
            y_stage2.append(normal_label_name)
        else:
            attack_types = [t for t in w["attack_type"].tolist() if t != normal_label_name]
            if not attack_types:
                y_stage2.append(normal_label_name)
            else:
                y_stage2.append(Counter(attack_types).most_common(1)[0][0])

    x = pd.DataFrame(rows)
    return x, pd.Series(y_stage1, name="stage1_label"), pd.Series(y_stage2, name="stage2_label")
