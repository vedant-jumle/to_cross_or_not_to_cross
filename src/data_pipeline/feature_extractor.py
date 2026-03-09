import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data_pipeline.pie_loader import load_pie


OFFICIAL_SPLITS = {
    "train": ["set01", "set02", "set04"],
    "val": ["set05", "set06"],
    "test": ["set03"],
    "all": ["set01", "set02", "set03", "set04", "set05", "set06"],
}


def _is_valid_features(features: Dict[str, Any]) -> bool:
    required = [
        "looking",
        "walking",
        "standing",
        "traffic_light",
        "crosswalk",
        "signalized",
        "vehicle_speed_kmh",
        "gps_lat",
        "gps_lon",
    ]
    for key in required:
        if features.get(key) is None:
            return False

    for key in ["looking", "walking", "standing", "crosswalk", "signalized"]:
        if features[key] not in (0, 1):
            return False

    if int(features["traffic_light"]) not in {0, 1, 2, 3}:
        return False

    if float(features["vehicle_speed_kmh"]) < 0:
        return False

    return True


def _build_scene_index(video_record: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    """
    Per-frame scene lookup:
      frame_id -> {"traffic_light": state_code, "crosswalk": 0/1}
    """
    scene: Dict[int, Dict[str, int]] = {}
    tl_states_by_frame: Dict[int, List[int]] = {}

    for obj in video_record["traffic_annotations"].values():
        obj_class = obj.get("obj_class")
        frames = obj.get("frames", [])

        if obj_class == "crosswalk":
            for frame_id in frames:
                item = scene.setdefault(int(frame_id), {"traffic_light": 0, "crosswalk": 0})
                item["crosswalk"] = 1

        if obj_class == "traffic_light":
            states = obj.get("state", [])
            for idx, frame_id in enumerate(frames):
                if idx >= len(states):
                    continue
                tl_states_by_frame.setdefault(int(frame_id), []).append(int(states[idx]))

    for frame_id, states in tl_states_by_frame.items():
        # Conservative merge when multiple lights are visible.
        if 1 in states:
            final_state = 1
        elif 2 in states:
            final_state = 2
        elif 3 in states:
            final_state = 3
        else:
            final_state = 0
        item = scene.setdefault(frame_id, {"traffic_light": 0, "crosswalk": 0})
        item["traffic_light"] = final_state

    return scene


def _get_anchor_index(ped_data: Dict[str, Any], anchor: str) -> int:
    if anchor == "last":
        return len(ped_data["frames"]) - 1

    critical_frame = int(ped_data["attributes"]["critical_point"])
    frames = ped_data["frames"]
    if critical_frame in frames:
        return frames.index(critical_frame)

    return min(range(len(frames)), key=lambda i: abs(frames[i] - critical_frame))


def extract_pie_features(
    db: Dict[str, Any],
    split: str,
    anchor: str = "critical",
    include_irrelevant: bool = False,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    dropped_invalid = 0
    dropped_irrelevant = 0
    total_tracks = 0

    for set_id in OFFICIAL_SPLITS[split]:
        if set_id not in db:
            continue
        for video_id, video_record in db[set_id].items():
            scene_index = _build_scene_index(video_record)
            vehicle_ann = video_record["vehicle_annotations"]

            for ped_id, ped_data in video_record["ped_annotations"].items():
                total_tracks += 1
                try:
                    attrs = ped_data["attributes"]
                    crossing_track_label = int(attrs["crossing"])  # 1,0,-1
                    if crossing_track_label == -1 and not include_irrelevant:
                        dropped_irrelevant += 1
                        continue

                    idx = _get_anchor_index(ped_data, anchor)
                    frame_id = int(ped_data["frames"][idx])

                    action = int(ped_data["behavior"]["action"][idx])  # 0 standing, 1 walking
                    look = int(ped_data["behavior"]["look"][idx])      # 0 not-looking, 1 looking

                    scene = scene_index.get(frame_id, {"traffic_light": 0, "crosswalk": 0})
                    traffic_light = int(scene["traffic_light"])
                    crosswalk = int(scene["crosswalk"])

                    signalized_bin = int(int(attrs["signalized"]) > 0)  # 0=n/a, 1=C, 2=S, 3=CS

                    veh = vehicle_ann.get(frame_id)
                    if veh is None:
                        dropped_invalid += 1
                        continue

                    features = {
                        "looking": look,
                        "walking": int(action == 1),
                        "standing": int(action == 0),
                        "traffic_light": traffic_light,
                        "crosswalk": crosswalk,
                        "signalized": signalized_bin,
                        "vehicle_speed_kmh": float(veh["OBD_speed"]),
                        "gps_lat": float(veh["latitude"]),
                        "gps_lon": float(veh["longitude"]),
                    }

                    if not _is_valid_features(features):
                        dropped_invalid += 1
                        continue

                    # Keep both labels explicit: per-track label and per-frame action label.
                    sample = {
                        "dataset": "PIE",
                        "split": split,
                        "set_id": set_id,
                        "video_id": video_id,
                        "pedestrian_id": ped_id,
                        "frame_id": frame_id,
                        "features": features,
                        "label_crossing": int(crossing_track_label > 0),
                        "label_crossing_track": int(crossing_track_label > 0),
                        "label_crossing_frame": int(ped_data["behavior"]["cross"][idx] == 1),
                        "intention_prob": float(attrs.get("intention_prob", -1.0)),
                    }
                    samples.append(sample)
                except (KeyError, IndexError, TypeError, ValueError):
                    dropped_invalid += 1
                    continue

    print(
        f"[extract] split={split} total_tracks={total_tracks} kept={len(samples)} "
        f"dropped_invalid={dropped_invalid} dropped_irrelevant={dropped_irrelevant}"
    )
    return samples


def write_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clean PIE feature dicts per sample from parsed annotation database."
    )
    parser.add_argument(
        "--pie-root",
        type=Path,
        default=Path("data/PIE_dataset"),
        help="PIE dataset root used by pie_loader.load_pie()",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Official PIE split to extract",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="critical",
        choices=["critical", "last"],
        help="Anchor frame selection per track",
    )
    parser.add_argument(
        "--include-irrelevant",
        action="store_true",
        help="Include tracks with crossing=-1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: processed/features_pie_<split>.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"processed/features_pie_{args.split}.jsonl")
    _, db = load_pie(data_path=args.pie_root, regen=False)
    samples = extract_pie_features(
        db=db,
        split=args.split,
        anchor=args.anchor,
        include_irrelevant=args.include_irrelevant,
    )
    write_jsonl(samples, output)
    print(f"[write] samples={len(samples)} output={output}")


if __name__ == "__main__":
    main()
