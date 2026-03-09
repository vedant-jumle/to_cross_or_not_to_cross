import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data_pipeline.dataloader import PIEDataset

# Try to get models from person 2
try:
    from src.models.mlp import MLP
except ImportError:
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout, use_batchnorm):
            super().__init__()
            layers = []
            in_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)
    
# Try to get metrics from person 1    
try:
    from src.evaluation.metrics import compute_metrics
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
    with open(config_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def build_loaders(cfg):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    
    ds_train = PIEDataset(
        jsonl_path=data_cfg["jsonl_path"],
        split="train",
        label_field=data_cfg["label_field"],
    ) 
    ds_val = PIEDataset(
        jsonl_path=data_cfg["jsonl_path"],
        split="val",
        label_field=data_cfg["label_field"],
    ) 
    
    train_loader = DataLoader(
        ds_train, 
        batch_size=train_cfg["batch_size"], 
        shuffle=not train_cfg.get("use_weighted_sampler", False), 
        num_workers=train_cfg["num_workers", 0]
    )
    
    val_loader = DataLoader(
        ds_val, 
        batch_size=train_cfg["batch_size"], 
        shuffle=not train_cfg.get("use_weighted_sampler", False), 
        num_workers=train_cfg["num_workers", 0]
    )
    
    return ds_train, ds_val, train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return running_loss / max(total_samples, 1)

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            
            
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_probs.append(probs.cpu().numpy())
            all_probs.append(y.cpu().numpy())
    
    avg_loss = running_loss / max(total_samples, 1) 
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)
            
        
    return avg_loss, y_prob, y_true

def save_checkpoint(checkpoint_location: Path, epoch: int, model, best_val_f1: float, cfg ):
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_f1": best_val_f1, "config": cfg}
    
    torch.save(checkpoint, checkpoint_location)
    
    
def save_log_csv(log_rows, csv_location: Path):
    csv_location.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["epoch", "train_loss", "val_loss", "val_f1", "val_auc_roc", "val_precision", "val_recall",]
    
    with open(csv_location, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictReader(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.write_rows(log_rows)
        

    