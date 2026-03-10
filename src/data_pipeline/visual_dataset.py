from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data_pipeline.dataloader import (
    read_jsonl,
    split_from_set_id,
    PIEFeatureSpec,
    train_sets,
    val_sets,
    test_sets,
)

DEFAULT_T: int = 8
DEFAULT_CROP_SIZE: int = 64


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _resolve_mp4_path(data_root: Path, set_id: str, video_id: str) -> Optional[Path]:
    """Returns the Path to the mp4 if it exists on disk, else None."""
    p = data_root / "clips" / set_id / f"{video_id}.mp4"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------

def _read_frames_from_mp4(
    mp4_path: Path,
    frame_ids: List[int],
) -> List[Optional[np.ndarray]]:
    """
    Opens mp4_path once, seeks to each frame_id (sorted, deduped by caller),
    and returns a list of BGR uint8 arrays (or None on failure).
    Opens and closes VideoCapture per call — safe under num_workers > 0.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return [None] * len(frame_ids)
    results = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
        ok, frame = cap.read()
        results.append(frame if ok else None)
    cap.release()
    return results


# ---------------------------------------------------------------------------
# Crop + resize
# ---------------------------------------------------------------------------

def _crop_and_resize(
    frame_bgr: np.ndarray,
    bbox: List[float],
    crop_size: int,
) -> torch.Tensor:
    """
    Crops the pedestrian bbox from frame_bgr, resizes to (crop_size, crop_size),
    normalises to [0, 1] float32, and returns a (3, H, W) tensor (RGB).
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return torch.zeros(3, crop_size, crop_size, dtype=torch.float32)

    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)  # (H, W, 3)
    return tensor.permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class VisualPIEDataset(Dataset):
    """
    Multi-modal PIE dataset: returns tabular features + a T-frame crop sequence
    for each pedestrian anchor sample.

    Reads frames directly from mp4 via cv2.CAP_PROP_POS_FRAMES.
    If the video for a given set is not on disk, x_vis is zeros and
    meta['video_available'] is False.

    Args:
        jsonl_path:             Path to a processed JSONL file.
        split:                  "train" | "val" | "test"
        db:                     PIE database dict from load_pie()[1].
        data_root:              Root of PIE dataset (clips/ lives here).
        T:                      Temporal window length (default 8).
        crop_size:              Spatial size of each crop (default 64).
        feature_spec:           PIEFeatureSpec; defaults to PIEFeatureSpec().
        label_field:            Label key to use (default "label_crossing").
        enforce_official_split: Drop records that don't belong to split.
        cache_dir:              Optional dir for .npy crop caches.
        augment:                Apply random flip + color jitter (train only).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        split: str,
        db: Dict[str, Any],
        *,
        data_root: str | Path = Path("data/PIE_dataset"),
        T: int = DEFAULT_T,
        crop_size: int = DEFAULT_CROP_SIZE,
        feature_spec: Optional[PIEFeatureSpec] = None,
        label_field: str = "label_crossing",
        enforce_official_split: bool = True,
        cache_dir: Optional[Path] = None,
        augment: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")

        self.split = split
        self.db = db
        self.data_root = Path(data_root)
        self.T = T
        self.crop_size = crop_size
        self.feature_spec = feature_spec or PIEFeatureSpec()
        self.label_field = label_field
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        all_records = read_jsonl(jsonl_path)
        self.records = self._filter_records(all_records, enforce_official_split)

        if not self.records:
            raise RuntimeError(
                f"No records found for split='{split}' in {jsonl_path} "
                f"with label_field='{label_field}'."
            )

        self.X, self.y = self._build_tabular_tensors()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_records(
        self,
        all_records: List[Dict[str, Any]],
        enforce_official_split: bool,
    ) -> List[Dict[str, Any]]:
        filtered = []
        for rec in all_records:
            set_id = rec.get("set_id")
            if not set_id:
                continue
            if enforce_official_split and split_from_set_id(set_id) != self.split:
                continue
            label = rec.get(self.label_field)
            if label not in (0, 1):
                continue
            mp4 = _resolve_mp4_path(self.data_root, set_id, rec.get("video_id", ""))
            rec["_video_available"] = mp4 is not None
            rec["_mp4_path"] = str(mp4) if mp4 else None
            filtered.append(rec)
        return filtered

    # ------------------------------------------------------------------
    # Tabular tensors (mirrors PIEDataset logic exactly)
    # ------------------------------------------------------------------

    def _build_tabular_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.feature_spec.keys()
        X = np.zeros((len(self.records), len(keys)), dtype=np.float32)
        y = np.zeros((len(self.records),), dtype=np.float32)
        for i, rec in enumerate(self.records):
            feats = rec.get("features", {}) or {}
            for j, k in enumerate(keys):
                v = feats.get(k, 0.0)
                try:
                    X[i, j] = float(v)
                except Exception:
                    X[i, j] = 0.0
            label = rec.get(self.label_field)
            y[i] = float(label) if label in (0, 1) else np.nan
        return torch.from_numpy(X), torch.from_numpy(y)

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        x_tab = self.X[idx]
        y = self.y[idx]

        if rec["_video_available"]:
            x_vis = self._get_visual_sequence(rec)
        else:
            x_vis = torch.zeros(self.T, 3, self.crop_size, self.crop_size, dtype=torch.float32)

        if self.augment and rec["_video_available"]:
            x_vis = self._maybe_augment(x_vis)

        return {
            "x_tab": x_tab,
            "x_vis": x_vis,
            "y": y,
            "meta": {
                "set_id": rec.get("set_id"),
                "video_id": rec.get("video_id"),
                "pedestrian_id": rec.get("pedestrian_id"),
                "frame_id": rec.get("frame_id"),
                "video_available": rec["_video_available"],
            },
        }

    # ------------------------------------------------------------------
    # Visual sequence extraction
    # ------------------------------------------------------------------

    def _get_visual_sequence(self, rec: Dict[str, Any]) -> torch.Tensor:
        """Returns (T, 3, crop_size, crop_size) float32 tensor."""
        set_id = rec["set_id"]
        video_id = rec["video_id"]
        ped_id = rec["pedestrian_id"]
        anchor_fid = int(rec["frame_id"])
        mp4_path = Path(rec["_mp4_path"])

        # Check disk cache first
        if self.cache_dir is not None:
            cache_key = f"{set_id}__{video_id}__{ped_id}__{anchor_fid}.npy"
            cache_file = self.cache_dir / cache_key
            if cache_file.exists():
                arr = np.load(str(cache_file))
                return torch.from_numpy(arr)

        # Build T-frame window ending at anchor
        try:
            ped_data = self.db[set_id][video_id]["ped_annotations"][ped_id]
        except KeyError:
            return torch.zeros(self.T, 3, self.crop_size, self.crop_size, dtype=torch.float32)

        ped_frames: List[int] = ped_data["frames"]
        ped_bboxes: List[List[float]] = ped_data["bbox"]

        # Find anchor index
        if anchor_fid in ped_frames:
            anchor_idx = ped_frames.index(anchor_fid)
        else:
            anchor_idx = min(range(len(ped_frames)), key=lambda i: abs(ped_frames[i] - anchor_fid))

        window_start = anchor_idx - self.T + 1
        pad_count = max(0, -window_start)
        real_start = max(0, window_start)

        window_fids = list(ped_frames[real_start: anchor_idx + 1])
        window_bboxes = list(ped_bboxes[real_start: anchor_idx + 1])

        # Pad short tracks by repeating first real frame
        final_fids = [window_fids[0]] * pad_count + window_fids
        final_bboxes = [window_bboxes[0]] * pad_count + window_bboxes

        # Read unique frames from mp4
        unique_fids = sorted(set(final_fids))
        raw_frames = _read_frames_from_mp4(mp4_path, unique_fids)
        frames_dict: Dict[int, Optional[np.ndarray]] = dict(zip(unique_fids, raw_frames))

        # Crop each of the T frames
        crops = []
        for fid, bbox in zip(final_fids, final_bboxes):
            frame = frames_dict.get(fid)
            if frame is None:
                crops.append(torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32))
            else:
                crops.append(_crop_and_resize(frame, bbox, self.crop_size))

        x_vis = torch.stack(crops, dim=0)  # (T, 3, H, W)

        # Save to cache
        if self.cache_dir is not None:
            np.save(str(cache_file), x_vis.numpy())

        return x_vis

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _maybe_augment(self, x_vis: torch.Tensor) -> torch.Tensor:
        """Applies the same spatial/colour transform to all T frames."""
        # Random horizontal flip
        if random.random() < 0.5:
            x_vis = torch.stack([TF.hflip(x_vis[t]) for t in range(self.T)])

        # Color jitter — sample parameters once, apply identically to all frames
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.9, 1.1)

        result = []
        for t in range(self.T):
            f = x_vis[t]
            f = TF.adjust_brightness(f, brightness)
            f = TF.adjust_contrast(f, contrast)
            f = TF.adjust_saturation(f, saturation)
            result.append(f)
        return torch.stack(result)

    # ------------------------------------------------------------------
    # Class statistics (mirrors PIEDataset.class_stats)
    # ------------------------------------------------------------------

    def class_stats(self) -> Dict[str, Any]:
        y = self.y.numpy()
        y = y[~np.isnan(y)]
        neg = int(np.sum(y == 0))
        pos = int(np.sum(y == 1))
        return {
            "n": int(len(y)),
            "neg": neg,
            "pos": pos,
            "pos_weight": (neg / pos) if pos > 0 else None,
        }

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


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_visual_loaders(
    jsonl_path: str | Path,
    db: Dict[str, Any],
    *,
    data_root: str | Path = Path("data/PIE_dataset"),
    T: int = DEFAULT_T,
    crop_size: int = DEFAULT_CROP_SIZE,
    label_field: str = "label_crossing",
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Returns (train_loader, val_loader, test_loader, stats).
    Mirrors the make_loaders() interface from dataloader.py.
    """
    ds_train = VisualPIEDataset(
        jsonl_path, "train", db,
        data_root=data_root, T=T, crop_size=crop_size,
        label_field=label_field, cache_dir=cache_dir, augment=True,
    )
    ds_val = VisualPIEDataset(
        jsonl_path, "val", db,
        data_root=data_root, T=T, crop_size=crop_size,
        label_field=label_field, cache_dir=cache_dir, augment=False,
    )
    ds_test = VisualPIEDataset(
        jsonl_path, "test", db,
        data_root=data_root, T=T, crop_size=crop_size,
        label_field=label_field, cache_dir=cache_dir, augment=False,
    )

    stats = {
        "train": ds_train.class_stats(),
        "val": ds_val.class_stats(),
        "test": ds_test.class_stats(),
        "feature_keys": ds_train.feature_spec.keys(),
        "label_field": label_field,
        "T": T,
        "crop_size": crop_size,
        "official_split": {
            "train": sorted(train_sets),
            "val": sorted(val_sets),
            "test": sorted(test_sets),
        },
    }

    if use_weighted_sampler:
        weights = ds_train.sample_weights_inverse_freq()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, stats
