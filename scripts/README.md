# Scripts

> Run all scripts from the **project root directory**, not from inside `scripts/`.

---

## Windows — `scripts/windows/`

### setup.bat
Sets up the full project environment:
- Creates the `cv_project` conda environment
- Clones PIE repo, extracts annotations into `data/PIE_dataset/`
- Copies PIE utilities into `src/pie_interface/`
- Deletes the temporary clone

```bat
scripts\windows\setup.bat
```

### download_clips.bat
Downloads PIE video clips from Google Drive and extracts them into `data/PIE_dataset/clips/`.

```bat
:: Activate env first
conda activate cv_project

:: Download a specific set (1-6)
scripts\windows\download_clips.bat 1

:: Download all sets
scripts\windows\download_clips.bat all

:: Interactive prompt
scripts\windows\download_clips.bat
```

---

## Linux — `scripts/linux/`

### setup.sh
Same as the Windows version, for Linux/WSL.

```bash
bash scripts/linux/setup.sh
```

### download_clips.sh
Same as the Windows version, for Linux/WSL.

```bash
# Activate env first
conda activate cv_project

# Download a specific set (1-6)
bash scripts/linux/download_clips.sh 1

# Download all sets
bash scripts/linux/download_clips.sh all

# Interactive prompt
bash scripts/linux/download_clips.sh
```

---

## Recommended order

```bash
# 1. Set up environment and annotations
bash scripts/linux/setup.sh          # Linux
# scripts\windows\setup.bat          # Windows

# 2. Activate env
conda activate cv_project

# 3. Download video clips
bash scripts/linux/download_clips.sh all       # Linux
# scripts\windows\download_clips.bat all       # Windows
```
