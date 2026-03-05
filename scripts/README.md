# Scripts

All setup scripts for Windows are in `scripts/windows/`.

> Run all scripts from the **project root directory**, not from inside `scripts/`.

---

## scripts/windows/setup.bat

Sets up the full project environment in one go:
- Creates the `cv_project` conda environment from `environment.yml`
- Clones the PIE GitHub repo, extracts annotations into `data/PIE_dataset/`
- Copies PIE interface utilities into `src/pie_interface/`
- Deletes the temporary PIE repo clone

**Usage:**
```bat
scripts\windows\setup.bat
```

Run this once after cloning the repo.

---

## scripts/windows/download_clips.bat

Downloads PIE video clips from Google Drive and extracts them into `data/PIE_dataset/clips/`.

Requires the `cv_project` conda environment to be active (uses `gdown`).

**Usage:**

Download a specific set:
```bat
conda activate cv_project
scripts\windows\download_clips.bat 1    :: downloads set01
scripts\windows\download_clips.bat 3    :: downloads set03
```

Download all sets:
```bat
conda activate cv_project
scripts\windows\download_clips.bat all
```

Run without arguments for an interactive prompt:
```bat
conda activate cv_project
scripts\windows\download_clips.bat
```

Already-downloaded zips are skipped automatically.

---

## Recommended order

```bat
:: 1. Set up environment and annotations
scripts\windows\setup.bat

:: 2. Activate env
conda activate cv_project

:: 3. Download video clips
scripts\windows\download_clips.bat all
```
