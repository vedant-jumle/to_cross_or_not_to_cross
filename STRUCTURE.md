# Repository Structure

Everyone works in their designated folder. Do not modify files outside your folder without discussing first.

```
to_cross_or_not_to_cross/
│
├── src/                               # All source code
│   ├── pie_interface/                 # PIE data interface (do not modify)
│   │   ├── pie_data.py
│   │   ├── data_gen_utils.py
│   │   └── utils.py
│   │
│   ├── data_pipeline/                 # Person 1 & 2
│   │   ├── pie_loader.py              # Person 1: PIE() wrapper, generate_database()
│   │   ├── feature_extractor.py       # Person 2: Extract feature vectors
│   │   └── dataloader.py              # Person 3: Dataset class, splits, labels
│   │
│   ├── models/                        # Person 3
│   │   ├── baseline.py                # MLP / LSTM classifier
│   │   └── train.py                   # Training script
│   │
│   ├── counterfactuals/               # Week 3+
│   │   ├── perturbations.py
│   │   ├── search.py
│   │   └── generator.py
│   │
│   ├── evaluation/                    # Week 7
│   │   ├── metrics.py
│   │   └── feature_importance.py
│   │
│   └── experiments/                   # Week 3+
│       ├── exp1_single_feature.py
│       └── exp2_minimal_cf.py
│
├── notebooks/                         # Person 4
│   └── 01_data_exploration.ipynb
│
├── results/
│   ├── figures/                       # Not committed to git
│   └── tables/                        # Not committed to git
│
├── data/                              # Not committed to git (gitignored)
│   └── PIE_dataset/
│       ├── annotations/
│       ├── annotations_attributes/
│       ├── annotations_vehicle/
│       └── clips/
│           ├── set01/
│           ├── set01.zip
│           └── ...
│
├── environment.yml
├── project_doc.md
├── STRUCTURE.md
└── README.md
```

## Rules

1. **Each person owns their folder** — coordinate before touching someone else's files
2. **No large files in git** — `data/` is gitignored; dataset, pkl cache, model weights stay local
3. **Notebooks go in `notebooks/`** — named with number prefix (`01_`, `02_`, etc.)
4. **Import paths** — always import from project root:
   ```python
   from src.pie_interface.pie_data import PIE
   from src.data_pipeline.feature_extractor import FeatureExtractor
   ```
5. **PIE interface** — use `src/pie_interface/pie_data.py`, do not modify it

## Who owns what

| Person | File | Deliverable |
|--------|------|-------------|
| 1 | `src/data_pipeline/pie_loader.py` | Working PIE() instance + pkl cache |
| 2 | `src/data_pipeline/feature_extractor.py` | Feature vectors per sample |
| 3 | `src/data_pipeline/dataloader.py`, `src/models/` | Dataset splits + baseline model |
| 4 | `notebooks/01_data_exploration.ipynb` | EDA notebook with findings |
