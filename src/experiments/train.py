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
    
    