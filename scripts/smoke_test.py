from src.data_pipeline.pie_loader import load_pie

def main():
    pie, db = load_pie(regen=False)

    print("Loaded sets:", sorted(db.keys()))

    sid = sorted(db.keys())[0]
    vid = sorted(db[sid].keys())[0]
    rec = db[sid][vid]

    print("Example:", sid, vid)
    print("Keys in db[sid][vid]:", rec.keys())

    pid = next(iter(rec["ped_annotations"].keys()))
    ped = rec["ped_annotations"][pid]

    print("Example pid:", pid)
    print("Ped keys:", ped.keys())
    print("Behavior keys:", ped["behavior"].keys())
    print("Attributes keys:", ped["attributes"].keys())

    # Alignment assertions (important)
    n = len(ped["frames"])
    assert n == len(ped["bbox"]) == len(ped["occlusion"])
    for k in ped["behavior"]:
        assert n == len(ped["behavior"][k])

    print("Num frames:", n, "Num boxes:", len(ped["bbox"]))
    print("Smoke test passed.")

if __name__ == "__main__":
    main()