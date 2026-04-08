from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from anomaly_attack.config import load_config
from anomaly_attack.data import (
    assign_attack_types_by_interval,
    chronological_train_test_split,
    load_attack_events,
    load_timeseries,
)
from anomaly_attack.features import build_window_dataset
from anomaly_attack.models import (
    evaluate_classifier,
    fit_stage2_model,
    make_stage2_model,
    predict_stage2,
)
from anomaly_attack.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 2 attack-type classifier.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).raw

    data_cfg = cfg["data"]
    win_cfg = cfg["window"]
    split_cfg = cfg["split"]
    train_cfg = cfg["training"]

    bundle = load_timeseries(
        csv_path=data_cfg["train_csv"],
        timestamp_col=data_cfg["timestamp_col"],
        binary_label_col=data_cfg["binary_label_col"],
    )
    events = load_attack_events(
        file_path=data_cfg["attack_events_file"],
        start_col=data_cfg["attack_start_col"],
        end_col=data_cfg["attack_end_col"],
        type_col=data_cfg["attack_type_col"],
        sheet_name=data_cfg.get("attack_events_sheet", 0),
    )
    enriched = assign_attack_types_by_interval(
        df=bundle.df,
        events=events,
        timestamp_col=data_cfg["timestamp_col"],
        normal_label_name=data_cfg["normal_label_name"],
        start_col=data_cfg["attack_start_col"],
        end_col=data_cfg["attack_end_col"],
        type_col=data_cfg["attack_type_col"],
    )

    train_df, test_df = chronological_train_test_split(enriched, split_cfg["test_ratio"])

    x_train, _, y2_train = build_window_dataset(
        df=train_df,
        sensor_cols=bundle.sensor_cols,
        binary_label_col=data_cfg["binary_label_col"],
        binary_attack_value=data_cfg["binary_attack_value"],
        normal_label_name=data_cfg["normal_label_name"],
        window_size=win_cfg["size"],
        stride=win_cfg["stride"],
        min_attack_ratio_for_type=win_cfg["min_attack_ratio_for_type"],
        include_slope=cfg["features"]["include_slope"],
    )

    x_test, _, y2_test = build_window_dataset(
        df=test_df,
        sensor_cols=bundle.sensor_cols,
        binary_label_col=data_cfg["binary_label_col"],
        binary_attack_value=data_cfg["binary_attack_value"],
        normal_label_name=data_cfg["normal_label_name"],
        window_size=win_cfg["size"],
        stride=win_cfg["stride"],
        min_attack_ratio_for_type=win_cfg["min_attack_ratio_for_type"],
        include_slope=cfg["features"]["include_slope"],
    )

    normal_name = data_cfg["normal_label_name"]
    train_mask = y2_train != normal_name
    test_mask = y2_test != normal_name

    x_train_attack = x_train.loc[train_mask]
    y2_train_attack = y2_train.loc[train_mask]
    x_test_attack = x_test.loc[test_mask]
    y2_test_attack = y2_test.loc[test_mask]

    if len(y2_train_attack) == 0:
        raise RuntimeError("No attack windows found for Stage 2 training.")

    stage2_params = dict(train_cfg["stage2"])
    random_state = int(train_cfg["random_state"])

    feature_columns = list(x_train.columns)
    x_train_np = x_train_attack.to_numpy(dtype=np.float32)
    x_test_np = x_test_attack.to_numpy(dtype=np.float32)

    x_mean = x_train_np.mean(axis=0)
    x_std = x_train_np.std(axis=0)
    x_std[x_std == 0.0] = 1.0

    x_train_scaled = (x_train_np - x_mean) / x_std
    x_test_scaled = (x_test_np - x_mean) / x_std

    classes = sorted(y2_train_attack.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_train_idx = np.array([class_to_idx[c] for c in y2_train_attack.tolist()], dtype=np.int64)

    # Chronological split can place some attack types only in test data.
    known_test_mask = y2_test_attack.isin(class_to_idx)
    unseen_test_labels = sorted(set(y2_test_attack.loc[~known_test_mask].tolist()))
    if unseen_test_labels:
        print(
            "Warning: unseen attack types in test split were excluded from metrics: "
            + ", ".join(map(str, unseen_test_labels))
        )

    x_test_attack_eval = x_test_attack.loc[known_test_mask].reset_index(drop=True)
    y2_test_attack_eval = y2_test_attack.loc[known_test_mask].reset_index(drop=True)

    model = make_stage2_model(
        input_dim=x_train_scaled.shape[1],
        num_classes=len(classes),
        params=stage2_params,
    )
    fit_stage2_model(
        model=model,
        x_train=x_train_scaled,
        y_train_idx=y_train_idx,
        params=stage2_params,
        random_state=random_state,
    )

    x_test_eval_np = x_test_attack_eval.to_numpy(dtype=np.float32)
    x_test_eval_scaled = (x_test_eval_np - x_mean) / x_std if len(x_test_eval_np) else x_test_eval_np

    if len(x_test_eval_scaled) == 0:
        metrics = {
            "labels": classes,
            "classification_report": {},
            "confusion_matrix": [],
            "note": "No test windows with labels seen in training. Metrics skipped.",
            "excluded_unseen_test_labels": unseen_test_labels,
        }
    else:
        test_probs = predict_stage2(model, x_test_eval_scaled)
        y_pred_idx = np.argmax(test_probs, axis=1)
        y_pred = [classes[i] for i in y_pred_idx]
        metrics = evaluate_classifier(y2_test_attack_eval.tolist(), y_pred)
        metrics["excluded_unseen_test_labels"] = unseen_test_labels

    artifacts_dir = ensure_dir(cfg["artifacts"]["dir"])
    model_path = Path(artifacts_dir) / "stage2_model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train_scaled.shape[1]),
            "num_classes": int(len(classes)),
            "classes": classes,
            "feature_columns": feature_columns,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "model_params": {
                "hidden_dim": int(stage2_params.get("hidden_dim", 256)),
                "dropout": float(stage2_params.get("dropout", 0.2)),
            },
        },
        model_path,
    )

    write_json(artifacts_dir / "stage2_metrics.json", metrics)
    write_json(
        artifacts_dir / "pipeline_meta.json",
        {
            "sensor_cols": bundle.sensor_cols,
            "window": win_cfg,
            "normal_label_name": data_cfg["normal_label_name"],
            "timestamp_col": data_cfg["timestamp_col"],
            "include_slope": cfg["features"]["include_slope"],
            "anomaly_flag_col": cfg["inference"]["anomaly_flag_col"],
            "min_anomaly_ratio_per_window": cfg["inference"]["min_anomaly_ratio_per_window"],
        },
    )

    print(f"Saved Stage 2 .pth model and metrics in: {artifacts_dir}")


if __name__ == "__main__":
    main()
