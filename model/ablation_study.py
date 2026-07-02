import argparse
import csv
import glob
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ablation_models import AblationModel


METHOD_TO_ID = {
    "original": 0,
    "Deepfakes": 1,
    "FaceShifter": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "Face2Face": 5,
    "others": 6,
    "unknown": 6,
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_metadata(metadata_path):
    df = pd.read_csv(metadata_path)

    rename_map = {}
    if "file_name" in df.columns and "file" not in df.columns:
        rename_map["file_name"] = "file"
    if "filename" in df.columns and "file" not in df.columns:
        rename_map["filename"] = "file"
    df = df.rename(columns=rename_map)

    required = {"file", "label", "method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    df["label"] = df["label"].astype(str).str.upper()
    df["method"] = df["method"].astype(str)
    return df


def resolve_video_path(row, data_root):
    if "folder_path" in row and isinstance(row["folder_path"], str):
        candidate = Path(data_root) / row["folder_path"]
        if candidate.exists():
            return str(candidate)

    matches = glob.glob(str(Path(data_root) / "**" / row["file"]), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find video for {row['file']} under {data_root}")
    return matches[0]


def build_path_index(data_root):
    log(f"Indexing videos under: {data_root}")
    started = time.perf_counter()
    video_paths = []
    for extension in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        video_paths.extend(glob.glob(str(Path(data_root) / "**" / extension), recursive=True))

    path_index = {}
    for path in video_paths:
        path_index.setdefault(os.path.basename(path), path)

    log(f"Indexed {len(video_paths)} videos ({len(path_index)} unique names) in {time.perf_counter() - started:.2f}s")
    return path_index


def apply_fallback_split(df, split, val_ratio, seed):
    split_name = str(split).lower()
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(df))
    val_size = max(1, int(len(df) * val_ratio))
    val_indices = set(shuffled_indices[:val_size].tolist())

    if split_name in {"validation", "val", "test"}:
        selected = [index in val_indices for index in range(len(df))]
    else:
        selected = [index not in val_indices for index in range(len(df))]

    return df.iloc[selected].reset_index(drop=True)


def build_records(metadata_path, data_root, split=None, limit=None, path_index=None, val_ratio=0.2, seed=42):
    log(f"Loading metadata: {metadata_path}")
    df = normalize_metadata(metadata_path)
    log(f"Metadata rows before split filter: {len(df)}")

    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
        log(f"Metadata rows after split='{split}' filter: {len(df)}")
    elif split:
        df = apply_fallback_split(df, split, val_ratio, seed)
        log(
            f"Split column not found. Applied deterministic fallback split='{split}' "
            f"with val_ratio={val_ratio}: {len(df)} rows"
        )

    records = []
    skipped_label = 0
    missing_files = 0

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        label_name = row["label"]
        if label_name not in {"FAKE", "REAL"}:
            skipped_label += 1
            continue

        # Keep the original notebook convention: FAKE=0, REAL=1.
        binary_label = 0 if label_name == "FAKE" else 1
        method_label = METHOD_TO_ID.get(row["method"], 6)
        video_path = path_index.get(row["file"]) if path_index else None

        if video_path is None:
            try:
                video_path = resolve_video_path(row, data_root)
            except FileNotFoundError:
                missing_files += 1
                continue

        records.append({
            "file": row["file"],
            "path": video_path,
            "binary_label": binary_label,
            "method_label": method_label,
        })

        if index == 1 or index % 500 == 0:
            log(f"Resolved records for split='{split}': {len(records)} usable / {index} scanned")

    if limit:
        log(f"Applying limit for split='{split}': {limit}")
        records = records[:limit]

    log(
        f"Built records for split='{split}': usable={len(records)}, "
        f"skipped_label={skipped_label}, missing_files={missing_files}"
    )
    return records


class VideoDataset(Dataset):
    def __init__(self, records, sequence_length=60, transform=None):
        self.records = records
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        frames = self._load_frames(record["path"])
        return (
            frames,
            torch.tensor(record["binary_label"], dtype=torch.long),
            torch.tensor(record["method_label"], dtype=torch.long),
        )

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.sequence_length:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))

        cap.release()

        if not frames:
            raise ValueError(f"No frames read from {video_path}")

        while len(frames) < self.sequence_length:
            frames.append(frames[-1].clone())

        return torch.stack(frames[:self.sequence_length])


def build_loaders(args):
    path_index = build_path_index(args.data_root)
    train_records = build_records(
        args.metadata,
        args.data_root,
        split=args.train_split,
        limit=args.limit_train,
        path_index=path_index,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    val_records = build_records(
        args.metadata,
        args.data_root,
        split=args.val_split,
        limit=args.limit_val,
        path_index=path_index,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if not train_records:
        raise ValueError("No training records found.")
    if not val_records:
        raise ValueError("No validation/test records found.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = VideoDataset(train_records, args.sequence_length, transform)
    val_dataset = VideoDataset(val_records, args.sequence_length, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    log(
        "DataLoaders ready: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"sequence_length={args.sequence_length}, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}"
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion_bin, criterion_method, device, method_loss_weight, log_every, scaler, use_amp):
    model.train()
    total_loss = 0.0
    started = time.perf_counter()
    total_batches = len(loader)

    for batch_index, (inputs, targets_bin, targets_method) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        inputs = inputs.to(device)
        targets_bin = targets_bin.to(device)
        targets_method = targets_method.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, output_bin, output_method = model(inputs)
            loss = criterion_bin(output_bin, targets_bin)
            if output_method is not None and method_loss_weight > 0:
                loss = loss + method_loss_weight * criterion_method(output_method, targets_method)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * inputs.size(0)

        if log_every and (batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches):
            log(
                f"train batch {batch_index}/{total_batches} "
                f"loss={loss.item():.4f} elapsed={time.perf_counter() - started:.1f}s"
            )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, log_every, use_amp):
    model.eval()
    y_true = []
    y_pred = []
    fake_scores = []
    started = time.perf_counter()
    total_batches = len(loader)

    for batch_index, (inputs, targets_bin, _) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        inputs = inputs.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, output_bin, _ = model(inputs)
        probs = torch.softmax(output_bin, dim=1)
        preds = torch.argmax(output_bin, dim=1)

        y_true.extend(targets_bin.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        fake_scores.extend(probs[:, 0].cpu().numpy().tolist())

        if log_every and (batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches):
            log(f"eval batch {batch_index}/{total_batches} elapsed={time.perf_counter() - started:.1f}s")

    # Existing label convention is FAKE=0, REAL=1. For AUC, use FAKE as positive.
    y_true_fake_positive = [1 if label == 0 else 0 for label in y_true]
    y_pred_fake_positive = [1 if label == 0 else 0 for label in y_pred]

    metrics = {
        "binary_acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1_fake_positive": f1_score(
            y_true_fake_positive,
            y_pred_fake_positive,
            average="macro",
            zero_division=0,
        ),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true_fake_positive, fake_scores)
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics


def run_variant(args, variant, train_loader, val_loader, device):
    log("=" * 72)
    log(f"Starting variant: {variant}")
    log(
        f"Config: backbone={args.backbone}, epochs={args.epochs}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, hidden_dim={args.hidden_dim}, "
        f"feature_chunk_size={args.feature_chunk_size}, "
        f"gradient_checkpointing={args.gradient_checkpointing}, amp={args.amp}"
    )
    model = AblationModel(
        variant=variant,
        model_name=args.backbone,
        hidden_dim=args.hidden_dim,
        feature_chunk_size=args.feature_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    criterion_bin = nn.CrossEntropyLoss()
    criterion_method = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    method_loss_weight = 1.0 if variant == "cnn_lstm_multitask" else 0.0
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = None
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        log(f"[{variant}] epoch {epoch}/{args.epochs} started")
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_bin,
            criterion_method,
            device,
            method_loss_weight,
            args.log_every,
            scaler,
            use_amp,
        )
        log(f"[{variant}] epoch {epoch} training finished: loss={loss:.4f}")
        metrics = evaluate(model, val_loader, device, args.log_every, use_amp)
        log(
            f"[{variant}] epoch={epoch} loss={loss:.4f} "
            f"acc={metrics['binary_acc']:.4f} auc={metrics['auc']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} elapsed={time.perf_counter() - epoch_started:.1f}s"
        )

        auc_value = metrics["auc"]
        if not np.isnan(auc_value) and auc_value > best_auc:
            best_auc = auc_value
            best = {**metrics, "epoch": epoch, "loss": loss}
            log(f"[{variant}] new best checkpoint metric: auc={best_auc:.4f} at epoch={epoch}")

    if best is None:
        best = {**metrics, "epoch": args.epochs, "loss": loss}

    log(f"Finished variant: {variant}")
    return best


def write_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "variant",
        "binary_acc",
        "auc",
        "macro_f1",
        "epoch",
        "loss",
        "explanation_output",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log(f"Results written: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN/LSTM/multitask ablation study.")
    parser.add_argument("--data-root", required=True, help="Root directory containing preprocessed videos.")
    parser.add_argument("--metadata", required=True, help="Global_metadata.csv path.")
    parser.add_argument("--output", default="model/ablation_results/performance_ablation.csv")
    parser.add_argument("--backbone", default="EfficientNet-b0", choices=["EfficientNet-b0", "resnext50_32x4d", "xception"])
    parser.add_argument("--variants", default="cnn_only,cnn_lstm,cnn_lstm_multitask")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--feature-chunk-size", type=int, default=32, help="Number of frames passed through the CNN backbone at once.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Reduce GPU memory by recomputing CNN activations during backward.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision to reduce GPU memory.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fallback validation ratio when metadata has no split column.")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N batches. Use 0 to disable batch logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    log(f"Using device: {device}")
    log(f"Arguments: {vars(args)}")

    train_loader, val_loader = build_loaders(args)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    log(f"Variants to run: {variants}")
    explanation_map = {
        "cnn_only": "No",
        "cnn_lstm": "No",
        "cnn_lstm_multitask": "No",
    }

    results = []
    for variant in variants:
        metrics = run_variant(args, variant, train_loader, val_loader, device)
        results.append({
            "variant": variant,
            "binary_acc": round(metrics["binary_acc"], 4),
            "auc": round(metrics["auc"], 4) if not np.isnan(metrics["auc"]) else "nan",
            "macro_f1": round(metrics["macro_f1"], 4),
            "epoch": metrics["epoch"],
            "loss": round(metrics["loss"], 4),
            "explanation_output": explanation_map.get(variant, "No"),
        })

    write_results(results, args.output)
    log("Final result table:")
    print(pd.DataFrame(results), flush=True)
    log(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
