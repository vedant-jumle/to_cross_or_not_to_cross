import torch
from src.data_pipeline.dataloader import make_loaders

import json
from collections import Counter

c = Counter()
with open("processed/features_pie_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        c[r["set_id"]] += 1
print(c)

train_loader, val_loader, test_loader, stats = make_loaders(
    "processed/features_pie_test.jsonl",      # or Person 2 JSONL if not joined
    label_field="label_crossing",     # or "label_intention" if you add it later
    batch_size=128,
    use_weighted_sampler=False,       # set True for balanced batches
)

pos_weight = stats["train"]["pos_weight"]
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

print(stats)