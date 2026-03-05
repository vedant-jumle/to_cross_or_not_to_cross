# to_cross_or_not_to_cross

Counterfactual explanations for pedestrian crossing prediction. Computer Vision course project at TU Delft.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/vedant-jumle/to_cross_or_not_to_cross
cd to_cross_or_not_to_cross
```

### 2. Set up the environment
```bash
conda env create -f environment.yml
conda activate cv_project
```

### 3. Set up the dataset

Create the dataset directory:
```bash
mkdir -p data/PIE_dataset/clips
```

Download the PIE dataset zip files and place them in `data/PIE_dataset/clips/`:
```
data/PIE_dataset/clips/
├── set01.zip
├── set02.zip
├── set03.zip
├── set04.zip
├── set05.zip
└── set06.zip
```

Extract each set:
```bash
cd data/PIE_dataset/clips
unzip set01.zip
unzip set02.zip
unzip set03.zip
unzip set04.zip
unzip set05.zip
unzip set06.zip
```

### 4. Set up annotations

Download the PIE annotations and extract them into `data/PIE_dataset/`:
- `annotations/` — per-frame bounding boxes and behavior labels
- `annotations_attributes/` — pedestrian attributes (age, gender, group size, etc.)
- `annotations_vehicle/` — OBD sensor data (vehicle speed, GPS)

Final structure should look like:
```
data/PIE_dataset/
├── annotations/
│   ├── set01/
│   │   ├── video_0001_annt.xml
│   │   └── ...
│   └── ...
├── annotations_attributes/
│   ├── set01/
│   │   ├── video_0001_attributes.xml
│   │   └── ...
│   └── ...
├── annotations_vehicle/
│   ├── set01/
│   │   ├── video_0001_obd.xml
│   │   └── ...
│   └── ...
└── clips/
    ├── set01/
    │   ├── video_0001.mp4
    │   └── ...
    └── ...
```

> **Note:** The `data/` directory is gitignored. Everyone sets it up locally.

## Project Structure
See [STRUCTURE.md](STRUCTURE.md) for the full breakdown of source folders and team ownership.

## Overview

This project builds a counterfactual explanation framework on top of a pedestrian crossing prediction model trained on the [PIE dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/). Given a model prediction ("will cross" / "won't cross"), we generate minimal feature perturbations that flip the prediction — answering questions like:

> "The pedestrian would NOT cross if the traffic light were red."

See [project_doc.md](project_doc.md) for the full project specification.
