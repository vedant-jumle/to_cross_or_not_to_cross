import argparse
import json
from pathlib import Path

import torch
import numpy as np

from src.evaluation.metrics import compute_binary_metrics
from src.models.mlp import MLP

def load_json(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

FEATURE_COLUMNS = [
    "looking",
    "walking",
    "standing",
    "traffic_light",
    "crosswalk",
    "signalized",
    "vehicle_speed",
    "gps_speed",
    "heading_angle",
]

def extract_features_and_labels(samples):
    X = []
    y = []

    for i, sample in enumerate(samples):
        row = []
        for col in FEATURE_COLUMNS:
            if col not in sample:
                raise KeyError(f"missing feature '{col}' in sample index '{i}': {sample}")
            row.append(sample[col])

        if "label" not in sample:
            raise KeyError(f"Missing 'label' in sample index '{i}': {sample}")
        
        X.append(row)
        y.append(sample["label"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def build_model_from_checkpoint(checkpoint, input_dim):
    model_config = checkpoint.get("model_config", {})

    model = MLP()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    samples = load_json(args.data)
    X, y_true = extract_features_and_labels(samples)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, input_dim=X.shape[1]).to(device)

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor).squeeze(-1)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    metrics = compute_binary_metrics(y_true, y_prob, threshold=args.threshold)

    print(json.dumps(metrics, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()