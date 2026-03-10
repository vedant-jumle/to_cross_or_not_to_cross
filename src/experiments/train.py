import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from ..data_pipeline.dataloader import read_jsonl
from ..data_pipeline.pie_loader import load_pie
from ..data_pipeline.visual_dataset import make_visual_loaders
from ..models.visual_model import VisualCrossingPredictor

# Try to get metrics from person 1
try:
    from ..evaluation.metrics import compute_metrics
except ImportError:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    def compute_metrics(y_true, y_prob):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    return parser.parse_args()


def load_config_file(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _prepare_combined_jsonl(cfg) -> Path:
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    if "jsonl_path" in data_cfg:
        return Path(data_cfg["jsonl_path"])

    jsonl_paths = data_cfg.get("jsonl_paths")
    if not jsonl_paths:
        raise KeyError("Config data section must define either 'jsonl_path' or 'jsonl_paths'.")

    combined_path = Path(
        data_cfg.get(
            "combined_jsonl_path",
            Path(output_cfg["checkpoint_dir"]) / "_combined_visual_dataset.jsonl",
        )
    )
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    split_order = ("train", "val", "test")
    with combined_path.open("w", encoding="utf-8") as handle:
        for split_name in split_order:
            split_path = jsonl_paths.get(split_name)
            if not split_path:
                continue
            for record in read_jsonl(split_path):
                json.dump(record, handle)
                handle.write("\n")

    return combined_path


def build_loaders(cfg):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    jsonl_path = _prepare_combined_jsonl(cfg)
    print("Loading PIE database (this may take a minute)...")
    _, db = load_pie(
        data_path=data_cfg["pie_root"],
        regen=bool(data_cfg.get("pie_regen", False)),
    )
    print(f"PIE database loaded. Sets available: {list(db.keys())}")

    print("Building visual datasets...")
    train_loader, val_loader, _, stats = make_visual_loaders(
        jsonl_path=jsonl_path,
        db=db,
        data_root=data_cfg["pie_root"],
        T=int(data_cfg["T"]),
        crop_size=int(data_cfg["crop_size"]),
        label_field=data_cfg["label_field"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        use_weighted_sampler=bool(train_cfg.get("use_weighted_sampler", False)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )

    print(f"Datasets ready — train={len(train_loader.dataset)}, val={len(val_loader.dataset)} samples")
    return train_loader.dataset, val_loader.dataset, train_loader, val_loader, stats


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for batch in pbar:
        x_tab = batch["x_tab"].to(device)
        x_vis = batch["x_vis"].to(device)
        y = batch["y"].to(device).float()

        optimizer.zero_grad()
        logits = model(x_tab, x_vis)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total_samples, 1)


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc="  val  ", leave=False, unit="batch")
    with torch.no_grad():
        for batch in pbar:
            x_tab = batch["x_tab"].to(device)
            x_vis = batch["x_vis"].to(device)
            y = batch["y"].to(device).float()

            logits = model(x_tab, x_vis)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = running_loss / max(total_samples, 1)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)

    return avg_loss, y_prob, y_true


def save_checkpoint(checkpoint_location: Path, epoch: int, model, best_val_f1: float, cfg):
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_f1": best_val_f1, "config": cfg}

    torch.save(checkpoint, checkpoint_location)


def save_log_csv(log_rows, csv_location: Path):
    csv_location.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_loss", "val_f1", "val_auc_roc", "val_precision", "val_recall"]

    with open(csv_location, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


def _make_phase_optimizer(model, train_cfg, phase_name: str):
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    head_lr = float(train_cfg["head_lr"])

    if phase_name == "phase1":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=head_lr, weight_decay=weight_decay)

    backbone_lr = float(train_cfg["backbone_lr"])
    return torch.optim.AdamW(
        model.parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr),
        weight_decay=weight_decay,
    )


def main():
    args = parse_args()
    cfg = load_config_file(args.config)

    output_cfg = cfg["output"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_location = checkpoint_dir / output_cfg["best_model_name"]

    log_csv_location = Path(output_cfg["log_csv_location"])

    device = torch.device(train_cfg.get("device", "cpu"))

    ds_train, ds_val, train_loader, val_loader, loader_stats = build_loaders(cfg)

    train_stats = loader_stats["train"]
    pos_weight_value = train_stats["pos_weight"]
    if pos_weight_value is None:
        raise ValueError("Could not compute pos_weight from training set.")

    model = VisualCrossingPredictor(
        T=int(cfg["data"]["T"]),
        n_tab_features=int(model_cfg.get("n_tab_features", len(loader_stats["feature_keys"]))),
        backbone_frozen=bool(model_cfg.get("backbone_frozen", True)),
        d_model=int(model_cfg.get("d_model", 512)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        tab_hidden=int(model_cfg.get("tab_hidden", 64)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    print(f"pos_weight: {pos_weight_value:.4f}")

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    phase1_epochs = int(train_cfg.get("phase1_epochs", train_cfg.get("epochs", 0)))
    phase2_epochs = int(train_cfg.get("phase2_epochs", 0))
    phase_schedule = [
        ("phase1", phase1_epochs),
        ("phase2", phase2_epochs),
    ]

    best_val_f1 = -1.0
    patience_counter = 0
    patience = int(train_cfg["early_stopping_patience"])
    log_rows = []
    current_epoch = 0
    stop_training = False

    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples:   {len(ds_val)}")
    print(f"Train class stats: {train_stats}")
    print(f"Using device: {device}")
    print("-" * 80)

    for phase_name, phase_epochs in phase_schedule:
        if phase_epochs <= 0:
            continue

        if phase_name == "phase2":
            model.unfreeze_backbone(from_layer=int(train_cfg["unfreeze_from_layer"]))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Backbone unfrozen from layer {train_cfg['unfreeze_from_layer']}. Trainable params: {trainable:,}")

        optimizer = _make_phase_optimizer(model, train_cfg, phase_name)
        print(f"Starting {phase_name} for {phase_epochs} epoch(s).")

        for _ in range(phase_epochs):
            current_epoch += 1

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_probs, val_labels = val_epoch(model, val_loader, criterion, device)

            metrics = compute_metrics(val_labels, val_probs)
            val_f1 = float(metrics.get("f1", 0.0))
            val_auc = float(metrics.get("auc_roc", 0.0))
            val_precision = float(metrics.get("precision", 0.0))
            val_recall = float(metrics.get("recall", 0.0))

            log_rows.append(
                {
                    "epoch": current_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_auc_roc": val_auc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            )

            print(
                f"Epoch {current_epoch:03d} | "
                f"phase={phase_name} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"val_auc_roc={val_auc:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                save_checkpoint(checkpoint_location, current_epoch, model, best_val_f1, cfg)
                print(f"  -> Saved new best checkpoint to {checkpoint_location}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {current_epoch}.")
                stop_training = True
                break

        if stop_training:
            break

    save_log_csv(log_rows, log_csv_location)
    print("-" * 80)
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Saved training log to: {log_csv_location}")
    print(f"Best checkpoint path: {checkpoint_location}")


if __name__ == "__main__":
    main()
