from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from anomaly_attack.features import extract_window_features
from anomaly_attack.models import make_stage2_model, predict_stage2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer attack type from anomaly points.")
    p.add_argument("--segment-csv", type=str, required=True)
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--anomaly-flag-col", type=str, default=None)
    p.add_argument("--min-anomaly-ratio", type=float, default=None)
    p.add_argument("--unknown-threshold", type=float, default=0.5)
    p.add_argument("--output-json", type=str, default="artifacts/inference_result.json")
    return p.parse_args()


def _load_segment_csv(path: str, timestamp_col: str) -> pd.DataFrame:
    seg = pd.read_csv(path)
    seg.columns = [str(c).strip() for c in seg.columns]

    # SWAT-like files can have the real header on row 2.
    if timestamp_col not in seg.columns:
        seg_retry = pd.read_csv(path, header=1)
        seg_retry.columns = [str(c).strip() for c in seg_retry.columns]
        if timestamp_col in seg_retry.columns:
            seg = seg_retry

    return seg


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    checkpoint = torch.load(artifacts_dir / "stage2_model.pth", map_location="cpu")
    feature_columns = checkpoint["feature_columns"]
    classes = checkpoint["classes"]
    x_mean = np.array(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.array(checkpoint["x_std"], dtype=np.float32)

    model = make_stage2_model(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        params=checkpoint.get("model_params", {}),
    )
    model.load_state_dict(checkpoint["state_dict"])

    with (artifacts_dir / "pipeline_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    timestamp_col = meta["timestamp_col"]
    sensor_cols = meta["sensor_cols"]
    window_size = int(meta["window"]["size"])
    stride = int(meta["window"]["stride"])
    anomaly_flag_col = args.anomaly_flag_col or meta.get("anomaly_flag_col", "is_anomaly")
    min_anomaly_ratio = args.min_anomaly_ratio
    if min_anomaly_ratio is None:
        min_anomaly_ratio = float(meta.get("min_anomaly_ratio_per_window", 0.1))

    seg = _load_segment_csv(args.segment_csv, timestamp_col)
    if timestamp_col not in seg.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    # Fallback for running inference directly on SWAT train/test files.
    if anomaly_flag_col not in seg.columns and "Normal/Attack" in seg.columns:
        seg[anomaly_flag_col] = (seg["Normal/Attack"].astype(str).str.strip() != "Normal").astype(int)
        print(
            f"Warning: '{anomaly_flag_col}' not found. Derived anomaly flags from 'Normal/Attack'."
        )

    if anomaly_flag_col not in seg.columns:
        raise ValueError(f"Missing anomaly flag column: {anomaly_flag_col}")

    seg[timestamp_col] = pd.to_datetime(seg[timestamp_col], errors="coerce", dayfirst=True)
    seg = seg.sort_values(timestamp_col).reset_index(drop=True)

    for c in sensor_cols:
        if c not in seg.columns:
            raise ValueError(f"Missing sensor column in segment: {c}")

    seg[anomaly_flag_col] = seg[anomaly_flag_col].astype(int)
    total_anomaly_points = int(seg[anomaly_flag_col].sum())

    features = []
    window_meta = []
    anomaly_ratios = []
    for start in range(0, len(seg) - window_size + 1, stride):
        end = start + window_size
        w = seg.iloc[start:end]

        ratio = float(w[anomaly_flag_col].mean())
        if ratio < min_anomaly_ratio:
            continue

        feat = extract_window_features(w, sensor_cols, include_slope=meta["include_slope"])
        features.append(feat)
        anomaly_ratios.append(ratio)
        window_meta.append(
            {
                "start_time": str(w[timestamp_col].iloc[0]),
                "end_time": str(w[timestamp_col].iloc[-1]),
            }
        )

    if not features:
        # Fallback: if threshold is too strict but anomalies exist, keep windows with ratio > 0.
        if total_anomaly_points > 0:
            for start in range(0, len(seg) - window_size + 1, stride):
                end = start + window_size
                w = seg.iloc[start:end]
                ratio = float(w[anomaly_flag_col].mean())
                if ratio <= 0.0:
                    continue

                feat = extract_window_features(w, sensor_cols, include_slope=meta["include_slope"])
                features.append(feat)
                anomaly_ratios.append(ratio)
                window_meta.append(
                    {
                        "start_time": str(w[timestamp_col].iloc[0]),
                        "end_time": str(w[timestamp_col].iloc[-1]),
                    }
                )

        # If still empty, return a clear result instead of throwing.
        if not features:
            aggregate = {
                "total_windows": 0,
                "attack_windows": 0,
                "dominant_predicted_type": meta["normal_label_name"],
                "reason": (
                    "No anomaly points found in input segment"
                    if total_anomaly_points == 0
                    else "No windows could be formed from anomaly points"
                ),
            }
            output = {
                "aggregate": aggregate,
                "window_predictions": [],
            }

            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=True)

            print(json.dumps(aggregate, indent=2, ensure_ascii=True))
            print(f"Full output written to: {out_path}")
            print(
                "Hint: this file appears to contain no attacks. Use an anomalous segment or lower --min-anomaly-ratio."
            )
            return

    x = pd.DataFrame(features).reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=np.float32)
    x_std[x_std == 0.0] = 1.0
    x_scaled = (x - x_mean) / x_std
    probs = predict_stage2(model, x_scaled)

    results = []
    top_labels = []

    for i in range(len(x_scaled)):
        type_proba = probs[i]
        max_idx = int(np.argmax(type_proba))
        pred_type = str(classes[max_idx])
        pred_prob = float(type_proba[max_idx])

        if pred_prob < args.unknown_threshold:
            pred_type = "unknown_attack"

        top_labels.append(pred_type)
        results.append(
            {
                **window_meta[i],
                "is_attack": True,
                "anomaly_ratio": float(anomaly_ratios[i]),
                "predicted_type": pred_type,
                "predicted_type_probability": pred_prob,
            }
        )

    aggregate = {
        "total_windows": len(results),
        "attack_windows": len(results),
        "dominant_predicted_type": Counter(top_labels).most_common(1)[0][0] if top_labels else meta["normal_label_name"],
    }

    output = {
        "aggregate": aggregate,
        "window_predictions": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(json.dumps(aggregate, indent=2, ensure_ascii=True))
    print(f"Full output written to: {out_path}")


if __name__ == "__main__":
    main()
