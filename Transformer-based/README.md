# LLM-AnomalyClassification-TimeSeries

Transformer-based attack-type classification for ICS time series.

This project assumes anomaly points are already detected upstream (for example by Anomaly Transformer) and focuses on attack-type attribution.

## Files

- configs/default.yaml
- scripts/train_stage2.py
- scripts/infer_attack_type.py
- scripts/run_all.sh
- anomaly_attack/data.py
- anomaly_attack/features.py
- anomaly_attack/models.py

## Install

python3 -m pip install -e .

## Train Transformer stage2

python3 scripts/train_stage2.py --config configs/default.yaml

Or:

bash scripts/run_all.sh configs/default.yaml

Artifacts written to artifacts:

- stage2_model.pth
- pipeline_meta.json
- stage2_metrics.json

## Inference

python3 scripts/infer_attack_type.py --segment-csv extracted_rows_227831_228231.csv --artifacts-dir artifacts --anomaly-flag-col is_anomaly --min-anomaly-ratio 0.1 --unknown-threshold 0.5 --output-json artifacts/inference_result.json
