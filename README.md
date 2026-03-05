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
Create the `raw_data/` directory and place the PIE dataset zip files in it:
```bash
mkdir Data
```

```
raw_data/
├── set01.zip
├── set02.zip
├── set03.zip
├── set04.zip
├── set05.zip
└── set06.zip
```

Extract each set:
```bash
cd Data
unzip set01.zip
unzip set02.zip
# ... etc
```

### 4. Set up PIE_dataset directory
```bash
mkdir -p PIE_dataset/PIE_clips

# Symlink annotations
ln -s ../PIE/annotations/annotations ./PIE_dataset/annotations
ln -s ../PIE/annotations/annotations_attributes ./PIE_dataset/annotations_attributes
ln -s ../PIE/annotations/annotations_vehicle ./PIE_dataset/annotations_vehicle

# Symlink video sets
ln -s ../../raw_data/set01 ./PIE_dataset/PIE_clips/set01
ln -s ../../raw_data/set02 ./PIE_dataset/PIE_clips/set02
# ... etc for each set you have downloaded
```

### 5. Extract PIE annotations
```bash
cd PIE/annotations
unzip annotations.zip
unzip annotations_attributes.zip
unzip annotations_vehicle.zip
```

## Structure
See [STRUCTURE.md](STRUCTURE.md) for the full breakdown of folders and team ownership.
