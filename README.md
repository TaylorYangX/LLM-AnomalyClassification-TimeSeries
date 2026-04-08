# LLM-AnomalyClassification-TimeSeries

Attack-type classification pipeline for ICS time-series attack analysis.

You already have anomaly points from an external detector (for example, Anomaly Transformer), and this project focuses only on identifying which attack type caused those anomalies.

This project is designed for SWAT/WADI-style data where you have:

- Time-series rows with sensor values and binary labels (`normal` vs `attack`).
- An attack-event table with attack type and time intervals (csv or xlsx).

## 1) Expected Data Format

### `train_timeseries.csv`

Required columns:

- `timestamp`: parseable datetime
- `label`: binary attack indicator (default attack value is `1`)
- numeric sensor columns (all other numeric columns are treated as features)

### Attack event file (csv/xlsx)

Required columns:

- `start_time`: parseable datetime
- `end_time`: parseable datetime
- `attack_type`: string class name

This file maps known attack intervals to attack types.

## 2) Project Structure

- `configs/default.yaml`: main config
- `scripts/train_stage2.py`: train attack-type classifier
- `scripts/infer_attack_type.py`: infer attack type on anomaly points
- `artifacts/`: saved models and metrics

## 3) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4) Configure Paths

Edit `configs/default.yaml`:

- `data.train_csv`
- `data.attack_events_file` (supports your `List_of_attacks_Final.xlsx`)
- `data.attack_start_col`, `data.attack_end_col`, `data.attack_type_col`
- `inference.anomaly_flag_col` (column produced by your anomaly detector)

For `List_of_attacks_Final.xlsx`, defaults are already set to:

- `attack_start_col: "Start Time"`
- `attack_end_col: "End Time"`
- `attack_type_col: "Attack Point"`

## 5) Train

```bash
python scripts/train_stage2.py --config configs/default.yaml
```

Or run helper script:

```bash
bash scripts/run_all.sh configs/default.yaml
```

Outputs in `artifacts/`:

- `stage2_model.pth`
- `pipeline_meta.json`
- `stage2_metrics.json`

## 6) Infer Attack Type For Anomalous Segment

Input segment CSV should contain:

- `timestamp`
- same sensor columns as training
- anomaly flag column (default: `is_anomaly`, values 0/1)

Run:

```bash
python scripts/infer_attack_type.py \
  --segment-csv data/raw/anomalous_segment.csv \
  --artifacts-dir artifacts \
  --anomaly-flag-col is_anomaly \
  --min-anomaly-ratio 0.1 \
  --unknown-threshold 0.5 \
  --output-json artifacts/inference_result.json
```

Output includes:

- aggregate decision (`dominant_predicted_type`)
- per-window predictions and probabilities for anomalous windows only

## 7) Notes

- If timestamps are offset from event labels, align them before training.
- For class imbalance, tune class weights and threshold values.
- For production use, validate event-level metrics and confusion by attack type.
