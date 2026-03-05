# Repository Structure

Everyone works in their designated folder. Do not modify files outside your folder without discussing first.

```
to_cross_or_not_to_cross/
│
├── src/                           # All source code lives here
│   ├── data/                      # Person 1 & 2: Data loading and feature extraction
│   │   ├── pie_loader.py          # Person 1: PIE() wrapper, generate_database()
│   │   ├── feature_extractor.py   # Person 2: Extract feature vectors from annotations
│   │   └── dataloader.py          # Person 3: Dataset class, splits, labels
│   │
│   ├── models/                    # Person 3: Model definition and training
│   │   ├── baseline.py            # MLP / LSTM classifier
│   │   └── train.py               # Training script
│   │
│   ├── counterfactuals/           # (Week 3+) Counterfactual generation
│   │   ├── perturbations.py       # Feature perturbation logic
│   │   ├── search.py              # Brute force + greedy search
│   │   └── generator.py           # Main interface
│   │
│   ├── evaluation/                # (Week 7) Metrics and analysis
│   │   ├── metrics.py             # Flip rate, sparsity, consistency
│   │   └── feature_importance.py  # Ranking features by flip frequency
│   │
│   └── experiments/               # (Week 3+) Experiment scripts
│       ├── exp1_single_feature.py
│       └── exp2_minimal_cf.py
│
├── notebooks/                     # Person 4: EDA and exploration
│   └── 01_data_exploration.ipynb  # Person 4: Distributions, sanity checks, viz
│
├── results/
│   ├── figures/                   # Saved plots (not committed to git)
│   └── tables/                    # Saved CSVs/results (not committed to git)
│
├── PIE/                           # PIE GitHub repo (do not modify)
├── PIE_dataset/                   # Local dataset root (not committed to git)
│   ├── annotations/      -> PIE/annotations/annotations
│   ├── annotations_attributes/ -> PIE/annotations/annotations_attributes
│   ├── annotations_vehicle/ -> PIE/annotations/annotations_vehicle
│   └── PIE_clips/                 # Symlinks to raw_data/set01-06
│
├── raw_data/                      # Raw zip files and extracted videos (not committed to git)
│   ├── set01.zip
│   ├── set01/
│   └── ...
│
├── environment.yml
├── project_doc.md
├── STRUCTURE.md
└── README.md
```

## Rules

1. **Each person owns their folder** — coordinate before touching someone else's files
2. **No large files in git** — dataset, pkl cache, model weights are all gitignored
3. **Notebooks go in `notebooks/`** — named with a number prefix (`01_`, `02_`, etc.)
4. **Import paths** — always import from project root, e.g.:
   ```python
   from src.data.pie_loader import PIELoader
   from src.data.feature_extractor import FeatureExtractor
   ```
5. **PIE interface** — use `PIE/utilities/pie_data.py` directly, do not copy or modify it

## Who owns what

| Person | File/Folder | Deliverable |
|--------|-------------|-------------|
| 1 | `src/data/pie_loader.py` | Working PIE() instance + pkl cache |
| 2 | `src/data/feature_extractor.py` | Feature vectors per sample |
| 3 | `src/data/dataloader.py`, `src/models/` | Dataset splits + baseline model |
| 4 | `notebooks/01_data_exploration.ipynb` | EDA notebook with findings |
