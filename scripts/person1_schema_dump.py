import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pie_interface.pie_data import PIE

pie = PIE(regen_database=False, data_path="data/PIE_dataset")
db = pie.generate_database()

# pick one example video + one example pedestrian
sid = sorted(db.keys())[0]
vid = sorted(db[sid].keys())[0]
rec = db[sid][vid]

pid = next(iter(rec["ped_annotations"].keys()))
ped = rec["ped_annotations"][pid]

# pick one traffic object
traffic_id = next(iter(rec["traffic_annotations"].keys()))
tr = rec["traffic_annotations"][traffic_id]

# pick one vehicle frame
veh_frame_id = sorted(rec["vehicle_annotations"].keys())[0]
veh = rec["vehicle_annotations"][veh_frame_id]

print("=== TOP LEVEL ===")
print("sets:", sorted(db.keys()))
print("db[set_id][video_id] keys:", rec.keys())

print("\n=== PER PEDESTRIAN ===")
print("example pid:", pid)
print("ped keys:", ped.keys())
print("behavior keys:", ped["behavior"].keys())
print("attributes keys:", ped["attributes"].keys())
print("lengths:", {
    "frames": len(ped["frames"]),
    "bbox": len(ped["bbox"]),
    "occlusion": len(ped["occlusion"]),
    **{k: len(ped["behavior"][k]) for k in ped["behavior"]}
})

print("\n=== PER TRAFFIC OBJECT ===")
print("example traffic id:", traffic_id)
print("traffic keys:", tr.keys())
print("traffic obj_class:", tr["obj_class"], "obj_type:", tr["obj_type"])
print("len(frames):", len(tr["frames"]), "len(bbox):", len(tr["bbox"]))
if tr["obj_class"] == "traffic_light":
    print("len(state):", len(tr["state"]))

print("\n=== PER VEHICLE FRAME ===")
print("example vehicle frame id:", veh_frame_id)
print("vehicle keys:", sorted(veh.keys()))