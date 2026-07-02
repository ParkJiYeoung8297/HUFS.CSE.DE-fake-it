# Ablation Study

This folder contains scripts for reviewer-facing ablation experiments.

## Performance Ablation

Run the same train/evaluation protocol for:

- `cnn_only`
- `cnn_lstm`
- `cnn_lstm_multitask`

Example:

```bash
python model/ablation_study.py \
  --data-root /content/drive/MyDrive/Capstone/Dataset/ff++ \
  --metadata /content/drive/MyDrive/Capstone/Dataset/ff++/Global_metadata.csv \
  --backbone EfficientNet-b0 \
  --sequence-length 60 \
  --batch-size 8 \
  --epochs 5 \
  --output model/ablation_results/performance_ablation.csv
```

For a quick smoke test:

```bash
python model/ablation_study.py \
  --data-root /content/drive/MyDrive/Capstone/Dataset/ff++ \
  --metadata /content/drive/MyDrive/Capstone/Dataset/ff++/Global_metadata.csv \
  --epochs 1 \
  --limit-train 16 \
  --limit-val 8
```

The output table reports:

- Binary Accuracy
- AUC, using FAKE as the positive class
- Macro F1

The dataset keeps the original notebook label convention:

- `FAKE = 0`
- `REAL = 1`

## Explainability Ablation

The LLM does not change detection accuracy. It adds interpretability.
Therefore, report explanation capability separately from model classification
performance.

```bash
python model/explainability_ablation.py \
  --output model/ablation_results/explainability_ablation.csv
```

Suggested paper table:

| Variant | Binary Acc. | AUC | Macro F1 | Explanation Output |
|---|---:|---:|---:|---|
| CNN only | measured | measured | measured | No |
| CNN + LSTM | measured | measured | measured | No |
| CNN + LSTM + multitask | measured | measured | measured | No |
| CNN + LSTM + Grad-CAM | same detector result | same detector result | same detector result | Visual only |
| CNN + LSTM + Grad-CAM + ROI | same detector result | same detector result | same detector result | Visual + ROI |
| Full model: CNN + LSTM + Grad-CAM + ROI + LLM | same detector result | same detector result | same detector result | Visual + ROI + Text |

Use the first three rows to demonstrate classification contribution.
Use the last three rows to demonstrate explanation contribution.
