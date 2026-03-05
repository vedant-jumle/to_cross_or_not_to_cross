import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRAFFIC_LIGHT_MAP = {
    0: "none",
    1: "red",
    2: "yellow",
    3: "green",
}

SPLIT_TO_SETS = {
    "train": ["set01", "set02", "set04"],
    "val": ["set05", "set06"],
    "test": ["set03"],
    "all": ["set01", "set02", "set03", "set04", "set05", "set06"],
}


def _to_scalar(value: Any) -> Any:
    """Unwrap values that are often stored as single-item lists in PIE outputs."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _parse_pid(pid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    PIE pid format is usually: <set_id>_<video_id>_<object_id>, e.g. 2_3_197.
    Returns (setXX, video_XXXX) when parseable, else (None, None).
    """
    parts = pid.split("_")
    if len(parts) < 3:
        return None, None
    try:
        set_id = f"set{int(parts[0]):02d}"
        video_id = f"video_{int(parts[1]):04d}"
        return set_id, video_id
    except ValueError:
        return None, None


def _frame_id_from_image_path(image_path: str) -> Optional[int]:
    try:
        return int(Path(image_path).stem)
    except (ValueError, TypeError):
        return None


def _extract_traffic_light(traffic_step: Any) -> str:
    traffic_step = _to_scalar(traffic_step)
    if isinstance(traffic_step, dict):
        code = int(traffic_step.get("traffic_light", 0))
        return TRAFFIC_LIGHT_MAP.get(code, "none")
    return "none"


def _extract_crosswalk(traffic_step: Any) -> int:
    traffic_step = _to_scalar(traffic_step)
    if isinstance(traffic_step, dict):
        return int(traffic_step.get("ped_crossing", 0) == 1)
    return 0


def _text_to_cross_state(value: str) -> int:
    mapping = {"not-crossing": 0, "crossing": 1, "crossing-irrelevant": -1}
    return mapping.get(value, 0)


def _is_valid_sample(features: Dict[str, Any]) -> bool:
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

    if features["traffic_light"] not in {"none", "red", "yellow", "green"}:
        return False

    if float(features["vehicle_speed_kmh"]) < 0:
        return False

    return True


def _find_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_pie_interface(pie_root: Path):
    # PIE code and PIE data folders may be nested differently in local setups.
    utilities_path = _find_existing_path(
        [
            pie_root / "utilities",
            pie_root.parent / "utilities",
        ]
    )
    if utilities_path is None or not utilities_path.exists():
        raise FileNotFoundError(f"Could not find PIE utilities path: {utilities_path}")

    data_path = pie_root
    # Common local layout in this repo:
    # PIE-master/
    #   utilities/
    #   annotations/
    #     annotations/
    #     annotations_attributes/
    #     annotations_vehicle/
    if (pie_root / "annotations" / "annotations").exists():
        data_path = pie_root / "annotations"

    sys.path.insert(0, str(utilities_path))
    from pie_data import PIE  # type: ignore

    return PIE(data_path=str(data_path))


def _pick_anchor_index(
    i: int,
    anchor: str,
    sequence: Dict[str, Any],
) -> int:
    if anchor == "critical":
        _, critical_idx = sequence["intent_frames_idx"][i]
        return int(critical_idx)
    return len(sequence["pid"][i]) - 1


def _extract_box_attributes(box_elem: ET.Element) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for attr_elem in box_elem.findall("attribute"):
        name = attr_elem.get("name")
        if name:
            attrs[name] = (attr_elem.text or "").strip()
    return attrs


def _resolve_pie_data_root(pie_root: Path) -> Path:
    candidates = [
        pie_root,
        Path("Data/PIE_dataset"),
        Path("Data/PIE/Annotations/PIE-master"),
        Path("Data/PIE/Annotations/PIE-master/annotations"),
    ]
    for c in candidates:
        if (c / "annotations").exists() and (c / "annotations_vehicle").exists():
            return c
        if (c / "annotations" / "annotations").exists() and (c / "annotations" / "annotations_vehicle").exists():
            return c / "annotations"
    return pie_root


def _parse_vehicle_obd(obd_xml_path: Path) -> Dict[int, Tuple[float, float, float]]:
    vehicle_by_frame: Dict[int, Tuple[float, float, float]] = {}
    root = ET.parse(obd_xml_path).getroot()
    for frame_elem in root.findall("frame"):
        try:
            frame_id = int(frame_elem.get("id"))
            obd_speed = float(frame_elem.get("OBD_speed"))
            lat = float(frame_elem.get("latitude"))
            lon = float(frame_elem.get("longitude"))
            vehicle_by_frame[frame_id] = (obd_speed, lat, lon)
        except (TypeError, ValueError):
            continue
    return vehicle_by_frame


def _parse_ped_attributes(attrs_xml_path: Path) -> Dict[str, Dict[str, Any]]:
    attrs_by_pid: Dict[str, Dict[str, Any]] = {}
    root = ET.parse(attrs_xml_path).getroot()
    for ped_elem in root.findall("pedestrian"):
        pid = ped_elem.get("id")
        if not pid:
            continue
        try:
            attrs_by_pid[pid] = {
                "critical_point": int(ped_elem.get("critical_point")),
                "crossing": int(ped_elem.get("crossing")),
                "signalized": ped_elem.get("signalized", "n/a"),
            }
        except (TypeError, ValueError):
            continue
    return attrs_by_pid


def _parse_scene_and_peds(
    annt_xml_path: Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Returns:
      scene_by_frame[frame] = {"traffic_light": "red|yellow|green|none", "crosswalk": 0|1}
      ped_by_pid[pid][frame] = {"looking": 0|1, "walking": 0|1, "cross": -1|0|1}
    """
    scene_states: Dict[int, Dict[str, Any]] = {}
    ped_by_pid: Dict[str, Dict[int, Dict[str, Any]]] = {}

    root = ET.parse(annt_xml_path).getroot()
    for track in root.findall("track"):
        label = track.get("label", "")

        if label == "traffic_light":
            for box in track.findall("box"):
                if int(box.get("outside", "0")) == 1:
                    continue
                frame_id = int(box.get("frame"))
                attrs = _extract_box_attributes(box)
                state = attrs.get("state", "__undefined__").lower()
                if state not in {"red", "yellow", "green"}:
                    state = "none"
                entry = scene_states.setdefault(frame_id, {"traffic_light_states": set(), "crosswalk": 0})
                entry["traffic_light_states"].add(state)

        elif label == "crosswalk":
            for box in track.findall("box"):
                if int(box.get("outside", "0")) == 1:
                    continue
                frame_id = int(box.get("frame"))
                entry = scene_states.setdefault(frame_id, {"traffic_light_states": set(), "crosswalk": 0})
                entry["crosswalk"] = 1

        elif label == "pedestrian":
            for box in track.findall("box"):
                if int(box.get("outside", "0")) == 1:
                    continue
                frame_id = int(box.get("frame"))
                attrs = _extract_box_attributes(box)
                pid = attrs.get("id")
                if not pid:
                    continue
                look_txt = attrs.get("look", "not-looking")
                action_txt = attrs.get("action", "standing")
                cross_txt = attrs.get("cross", "not-crossing")

                ped_frames = ped_by_pid.setdefault(pid, {})
                ped_frames[frame_id] = {
                    "looking": int(look_txt == "looking"),
                    "walking": int(action_txt == "walking"),
                    "cross": _text_to_cross_state(cross_txt),
                }

    scene_by_frame: Dict[int, Dict[str, Any]] = {}
    for frame_id, state_data in scene_states.items():
        states = state_data.get("traffic_light_states", set())
        if "red" in states:
            tl = "red"
        elif "yellow" in states:
            tl = "yellow"
        elif "green" in states:
            tl = "green"
        else:
            tl = "none"
        scene_by_frame[frame_id] = {
            "traffic_light": tl,
            "crosswalk": int(state_data.get("crosswalk", 0)),
        }
    return scene_by_frame, ped_by_pid


def _extract_with_xml_fallback(
    pie_root: Path,
    split: str,
    anchor: str = "critical",
    include_irrelevant: bool = False,
) -> List[Dict[str, Any]]:
    pie_data_root = _resolve_pie_data_root(pie_root)
    annt_root = pie_data_root / "annotations"
    attrs_root = pie_data_root / "annotations_attributes"
    obd_root = pie_data_root / "annotations_vehicle"

    if not (annt_root.exists() and attrs_root.exists() and obd_root.exists()):
        raise FileNotFoundError(
            f"PIE XML roots not found under {pie_data_root}. "
            f"Expected annotations/, annotations_attributes/, annotations_vehicle/."
        )

    samples: List[Dict[str, Any]] = []
    dropped_invalid = 0
    dropped_irrelevant = 0
    total_tracks = 0

    for set_id in SPLIT_TO_SETS[split]:
        for annt_xml in sorted((annt_root / set_id).glob("*_annt.xml")):
            video_id = annt_xml.stem.replace("_annt", "")
            attrs_xml = attrs_root / set_id / f"{video_id}_attributes.xml"
            obd_xml = obd_root / set_id / f"{video_id}_obd.xml"

            if not (attrs_xml.exists() and obd_xml.exists()):
                continue

            scene_by_frame, ped_by_pid = _parse_scene_and_peds(annt_xml)
            attrs_by_pid = _parse_ped_attributes(attrs_xml)
            vehicle_by_frame = _parse_vehicle_obd(obd_xml)

            for pid, frame_map in ped_by_pid.items():
                total_tracks += 1
                try:
                    if pid not in attrs_by_pid:
                        dropped_invalid += 1
                        continue
                    attrs = attrs_by_pid[pid]

                    if attrs["crossing"] == -1 and not include_irrelevant:
                        dropped_irrelevant += 1
                        continue

                    frames_sorted = sorted(frame_map.keys())
                    if not frames_sorted:
                        dropped_invalid += 1
                        continue

                    if anchor == "critical":
                        critical_frame = int(attrs["critical_point"])
                        if critical_frame in frame_map:
                            frame_id = critical_frame
                        else:
                            # Best effort fallback: nearest available frame to critical point.
                            frame_id = min(frames_sorted, key=lambda f: abs(f - critical_frame))
                    else:
                        frame_id = frames_sorted[-1]

                    ped_step = frame_map.get(frame_id)
                    veh_step = vehicle_by_frame.get(frame_id)
                    scene_step = scene_by_frame.get(frame_id, {"traffic_light": "none", "crosswalk": 0})
                    if ped_step is None or veh_step is None:
                        dropped_invalid += 1
                        continue

                    signalized_raw = str(attrs.get("signalized", "n/a")).upper()
                    signalized_bin = int(signalized_raw in {"C", "S", "CS"})

                    walking_val = int(ped_step["walking"])
                    features = {
                        "looking": int(ped_step["looking"]),
                        "walking": walking_val,
                        "standing": int(1 - walking_val),
                        "traffic_light": str(scene_step.get("traffic_light", "none")),
                        "crosswalk": int(scene_step.get("crosswalk", 0)),
                        "signalized": signalized_bin,
                        "vehicle_speed_kmh": float(veh_step[0]),
                        "gps_lat": float(veh_step[1]),
                        "gps_lon": float(veh_step[2]),
                    }
                    if not _is_valid_sample(features):
                        dropped_invalid += 1
                        continue

                    sample = {
                        "dataset": "PIE",
                        "split": split,
                        "set_id": set_id,
                        "video_id": video_id,
                        "pedestrian_id": pid,
                        "frame_id": frame_id,
                        "features": features,
                        "label_crossing": int(attrs["crossing"] > 0),
                    }
                    samples.append(sample)
                except (TypeError, ValueError, KeyError):
                    dropped_invalid += 1
                    continue

    print(
        f"[extract] split={split} total_tracks={total_tracks} kept={len(samples)} "
        f"dropped_invalid={dropped_invalid} dropped_irrelevant={dropped_irrelevant}"
    )
    return samples


def extract_pie_split(
    pie_root: Path,
    split: str,
    anchor: str = "critical",
    include_irrelevant: bool = False,
) -> List[Dict[str, Any]]:
    try:
        pie = load_pie_interface(pie_root)
        sequence = pie.generate_data_trajectory_sequence(
            split,
            seq_type="all",
            data_split_type="default",
            fstride=1,
            min_track_size=0,
        )
    except (ModuleNotFoundError, FileNotFoundError, ImportError):
        return _extract_with_xml_fallback(
            pie_root=pie_root,
            split=split,
            anchor=anchor,
            include_irrelevant=include_irrelevant,
        )

    samples: List[Dict[str, Any]] = []
    dropped_invalid = 0
    dropped_irrelevant = 0

    n_tracks = len(sequence["pid"])
    for i in range(n_tracks):
        try:
            idx = _pick_anchor_index(i, anchor, sequence)

            pid = str(_to_scalar(sequence["pid"][i][idx]))
            set_id, video_id = _parse_pid(pid)
            image_path = str(sequence["image"][i][idx])
            frame_id = _frame_id_from_image_path(image_path)

            look_val = int(_to_scalar(sequence["looks"][i][idx]))
            action_val = int(_to_scalar(sequence["actions"][i][idx]))  # 0 standing, 1 walking
            standing_val = int(action_val == 0)
            walking_val = int(action_val == 1)

            signalized_raw = int(_to_scalar(sequence["signalized"][i][idx]))
            signalized_bin = int(signalized_raw > 0)

            traffic_step = sequence["traffic"][i][idx]
            traffic_light = _extract_traffic_light(traffic_step)
            crosswalk = _extract_crosswalk(traffic_step)

            vehicle_speed = float(_to_scalar(sequence["obd_speed"][i][idx]))

            gps = sequence["gps_coord"][i][idx]
            gps_lat = float(gps[0])
            gps_lon = float(gps[1])

            crossing_state = int(_to_scalar(sequence["cross"][i][idx]))
            if crossing_state == -1 and not include_irrelevant:
                dropped_irrelevant += 1
                continue

            label_crossing = int(_to_scalar(sequence["activities"][i][idx]))

            features = {
                "looking": look_val,
                "walking": walking_val,
                "standing": standing_val,
                "traffic_light": traffic_light,
                "crosswalk": crosswalk,
                "signalized": signalized_bin,
                "vehicle_speed_kmh": vehicle_speed,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
            }

            if not _is_valid_sample(features):
                dropped_invalid += 1
                continue

            sample = {
                "dataset": "PIE",
                "split": split,
                "set_id": set_id,
                "video_id": video_id,
                "pedestrian_id": pid,
                "frame_id": frame_id,
                "features": features,
                "label_crossing": label_crossing,
            }
            samples.append(sample)
        except (IndexError, KeyError, ValueError, TypeError):
            dropped_invalid += 1

    print(
        f"[extract] split={split} total_tracks={n_tracks} kept={len(samples)} "
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
        description="Extract clean per-sample features for pedestrian crossing analysis."
    )
    parser.add_argument(
        "--pie-root",
        type=Path,
        default=Path("Data/PIE_dataset"),
        help="Path to PIE root (supports PIE interface layout or raw XML layout).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="critical",
        choices=["critical", "last"],
        help="Which frame to sample from each pedestrian track",
    )
    parser.add_argument(
        "--include-irrelevant",
        action="store_true",
        help="Include tracks whose crossing action label is -1 at the anchor frame",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/features_pie_train.jsonl"),
        help="Output JSONL file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = extract_pie_split(
        pie_root=args.pie_root,
        split=args.split,
        anchor=args.anchor,
        include_irrelevant=args.include_irrelevant,
    )
    write_jsonl(samples, args.output)
    print(f"[write] samples={len(samples)} output={args.output}")


if __name__ == "__main__":
    main()
