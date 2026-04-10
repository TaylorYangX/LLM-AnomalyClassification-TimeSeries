from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass
class DataBundle:
    df: pd.DataFrame
    sensor_cols: List[str]


def _parse_mixed_excel_datetime(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna() & parsed.isna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return parsed


def load_timeseries(
    csv_path: str | Path,
    timestamp_col: str,
    binary_label_col: str,
) -> DataBundle:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if timestamp_col not in df.columns or binary_label_col not in df.columns:
        df_retry = pd.read_csv(csv_path, header=1)
        df_retry.columns = [str(c).strip() for c in df_retry.columns]
        if timestamp_col in df_retry.columns and binary_label_col in df_retry.columns:
            df = df_retry

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if binary_label_col not in df.columns:
        raise ValueError(f"Missing binary label column: {binary_label_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=True)
    if df[timestamp_col].isna().any():
        raise ValueError("Invalid timestamp values encountered.")

    if df[binary_label_col].dtype == object:
        df[binary_label_col] = df[binary_label_col].astype(str).str.strip()

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    non_sensor = {timestamp_col, binary_label_col, "attack_type"}
    sensor_cols = [
        c for c in df.columns if c not in non_sensor and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not sensor_cols:
        raise ValueError("No numeric sensor columns found.")

    return DataBundle(df=df, sensor_cols=sensor_cols)


def load_attack_events(
    file_path: str | Path,
    start_col: str = "start_time",
    end_col: str = "end_time",
    type_col: str = "attack_type",
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        events = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix in {".csv", ".txt"}:
        events = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported attack event file format: {path}")

    required = {start_col, end_col, type_col}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing required attack event columns: {sorted(missing)}")

    events = events[[start_col, end_col, type_col]].copy()
    events[type_col] = events[type_col].astype(str).str.strip()
    events = events[events[type_col].notna() & (events[type_col] != "")]

    events[start_col] = _parse_mixed_excel_datetime(events[start_col])
    events[end_col] = _parse_mixed_excel_datetime(events[end_col])

    events = events.dropna(subset=[start_col, end_col]).reset_index(drop=True)
    if events.empty:
        raise ValueError("No valid attack events found after timestamp parsing.")

    events = events.sort_values(start_col).reset_index(drop=True)
    return events


def assign_attack_types_by_interval(
    df: pd.DataFrame,
    events: pd.DataFrame,
    timestamp_col: str,
    normal_label_name: str,
    start_col: str = "start_time",
    end_col: str = "end_time",
    type_col: str = "attack_type",
) -> pd.DataFrame:
    out = df.copy()
    out["attack_type"] = normal_label_name

    for _, row in events.iterrows():
        mask = (out[timestamp_col] >= row[start_col]) & (out[timestamp_col] <= row[end_col])
        out.loc[mask, "attack_type"] = row[type_col]

    return out


def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    split_idx = int(len(df) * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df
