import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have same dimensions")

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(np.unique(y_true)) < 2:
        metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred)

    return metrics


# Alias used by train.py
compute_metrics = compute_binary_metrics