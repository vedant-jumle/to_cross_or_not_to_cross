import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

Split = Literal["train", "test", "val"]

train_sets = {"set01", "set02", "set04"}
val_sets = {"set05", "set06"}
test_sets = {"set03"}

def split_from_set_id(set_id: str) -> Split:
    if set_id in train_sets:
        return "train"
    if set_id in val_sets:
        return "val"
    if set_id in test_sets:
        return "test"
    raise ValueError(f"Unknown set_id={set_id}")

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Bad JSON at line {ln} in {path}: {error}") from error
    return records


# Feature order definition to guarantee order
@dataclass(frozen=True)
class PIEFeatureSpec:
    looking: str = "looking"
    walking: str = "walking"
    standing: str = "standing"
    traffic_light: str = "traffic_light"
    crosswalk: str = "crosswalk"
    signalized: str = "signalized"
    vehicle_speed_kmh: str = "vehicle_speed_kmh"
    gps_lat: str = "gps_lat"
    gps_lon: str = "gps_lon"
    
    def keys(self) -> List[str]:
        return [
            self.looking, self.walking, 
            self.standing, self.traffic_light, 
            self.crosswalk, self.signalized, 
            self.vehicle_speed_kmh, self.gps_lat, 
            self.gps_lon,
        ]

class PIEDataset(Dataset):
    
    # Loads and filters records
    def __init__(self, jsonl_path: str | Path, 
                 split: Split, *, 
                 feature_spec: Optional[PIEFeatureSpec] = None, 
                 label_field: str = "label_crossing", 
                 enforce_official_split: bool = True, 
                 drop_missing_label: bool = True,):
        
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be one of: train, val, test")
        
        self.jsonl_path = Path(jsonl_path)
        self.split = split  # keep as string
        self.feature_spec = feature_spec or PIEFeatureSpec()  # keep as PIEFeatureSpec object
        self.label_field = label_field  # keep as string key
        
        all_records = read_jsonl(self.jsonl_path)
        
        filtered: List[Dict[str, Any]] = []
        for rec in all_records:
            set_id = rec.get("set_id")
            if not set_id:
                continue
            official = split_from_set_id(set_id)
            rec["split_official"] = official
            if enforce_official_split and official != split:
                continue
            
            # Filter out any missing labels
            y = rec.get(self.label_field, None)
            if drop_missing_label and y not in (0,1):
                continue
            
            filtered.append(rec)
            
        if not filtered:
            raise RuntimeError(
                f"No sample found for split='{split}' in {self.jsonl_path}. "
                f"Check set_id values and label_field='{self.label_field}'. "
            )
        
        
        # Stores records and builds tensors
        self.records = filtered
        self.X = self._build_X(self.records)
        self.y = self._build_y(self.records)
        
    # Turns dictionary features into a matrix
    def _build_X(self, records: List[Dict[str, Any]]) -> torch.Tensor:
        keys = self.feature_spec.keys()
        X = np.zeros((len(records), len(keys)), dtype=np.float32)

        for i, rec in enumerate(records):
            feats = rec.get("features", {}) or {}
            row = []
            for k in keys:
                v = feats.get(k, 0.0)
                try:
                    row.append(float(v))
                except Exception:
                    row.append(0.0)
            X[i] = np.array(row, dtype=np.float32)

        return torch.from_numpy(X)

    # Extracts the appropriate labels
    def _build_y(self, records: List[Dict[str, Any]]) -> torch.Tensor:
        y = np.zeros((len(records),), dtype=np.float32)
        for i, rec in enumerate(records):
            v = rec.get(self.label_field, None)
            y[i] = float(v) if v in (0, 1) else np.nan
        return torch.from_numpy(y)
    
    # Pretty self-explanatory -> Get the number of samples
    def __len__(self) -> int:
        return len(self.records)

    # Pretty self-explanatory -> Return a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "meta": {
                "set_id": rec.get("set_id"),
                "video_id": rec.get("video_id"),
                "pedestrian_id": rec.get("pedestrian_id"),
                "frame_id": rec.get("frame_id"),
            }
        }

    # Use to report on imbalances
    def class_stats(self) -> Dict[str, Any]:
        y = self.y.numpy()
        y = y[~np.isnan(y)]
        neg = int(np.sum(y == 0))
        pos = int(np.sum(y == 1))
        pos_weight = (neg / pos) if pos > 0 else None
        return {
            "n": int(len(y)),
            "neg": neg,
            "pos": pos,
            "pos_over_neg": (pos / max(neg, 1)),
            "pos_weight": pos_weight,
        }

    # Use to balance sampling: creates a weight per sample basically
    def sample_weights_inverse_freq(self) -> List[float]:
        y = self.y.numpy()
        valid = y[~np.isnan(y)]
        neg = max(int(np.sum(valid == 0)), 1)
        pos = max(int(np.sum(valid == 1)), 1)

        weights = []
        for v in y:
            if np.isnan(v):
                weights.append(0.0)
            elif int(v) == 0:
                weights.append(1.0 / neg)
            else:
                weights.append(1.0 / pos)
        return weights

def make_loaders(
    jsonl_path: str | Path,
    *,
    label_field: str = "label_crossing",
    batch_size: int = 256,
    num_workers: int = 0,
    use_weighted_sampler: bool = False, ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    
    ds_train = PIEDataset(jsonl_path, "train", label_field=label_field)
    ds_val   = PIEDataset(jsonl_path, "val",   label_field=label_field)
    ds_test  = PIEDataset(jsonl_path, "test",  label_field=label_field)

    stats = {
        "train": ds_train.class_stats(),
        "val": ds_val.class_stats(),
        "test": ds_test.class_stats(),
        "feature_keys": ds_train.feature_spec.keys(),
        "label_field": label_field,
        "official_split": {
            "train": sorted(train_sets),
            "val": sorted(val_sets),
            "test": sorted(test_sets),
        }
    }

    if use_weighted_sampler:
        weights = ds_train.sample_weights_inverse_freq()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, stats


