from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd


def build_sequence_dataset(
    df: pd.DataFrame,
    sensor_cols: List[str],
    binary_label_col: str,
    binary_attack_value: int | str,
    normal_label_name: str,
    window_size: int,
    stride: int,
    min_attack_ratio_for_type: float,
) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    sequences = []
    y_stage1 = []
    y_stage2 = []

    n = len(df)
    if n < window_size:
        raise ValueError("Dataset shorter than window_size.")

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        w = df.iloc[start:end]

        seq = w[sensor_cols].to_numpy(dtype=np.float32)
        sequences.append(seq)

        attack_mask = w[binary_label_col] == binary_attack_value
        attack_ratio = float(attack_mask.mean())
        y_stage1.append(int(attack_ratio > 0.0))

        if attack_ratio < min_attack_ratio_for_type:
            y_stage2.append(normal_label_name)
        else:
            attack_types = [t for t in w["attack_type"].tolist() if t != normal_label_name]
            if not attack_types:
                y_stage2.append(normal_label_name)
            else:
                y_stage2.append(Counter(attack_types).most_common(1)[0][0])

    x_seq = np.stack(sequences, axis=0)
    return x_seq, pd.Series(y_stage1, name="stage1_label"), pd.Series(y_stage2, name="stage2_label")
