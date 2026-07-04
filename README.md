# Messina et al. (2026) — Source Data & Plots

This repository contains **lightweight source data** and **clean plotting scripts** for all figures in **Messina et al. (2026)**.

## Repository Structure

```bash
messina_2026/
├── Figures/                  # Generated Figure panels
├── Scripts/                   # Python plotting scripts
├── Source_light/              # Lightweight source data (CSV + .npy)
└── README.md
```

## How to Reproduce the Figures
### 1. Clone the repository

```bash
git clone https://github.com/yourusername/messina_2026.git
cd messina_2026
```

### 2. Run all figures at once

```bash
cd Scripts
python run_all_figures.py
```

### 3. Run a specific figure

```bash
cd Scripts
python Figure_1.py
python Figure_2.py
python Figure_3.py
python Figure_4.py
python Figure_S1.py
python Figure_S2.py
python Figure_S3.py
python Figure_S4.py
```

### Raw Traces 

Raw chromatin tracing tables are hosted at our https://osf.io/vbpuy/?view_only=9968c35f4d9242b4ad4b329f64ad48cc Open Science Foundation (OSF) Repository.

Bardou, M. et al. Chromatin tracing at the rut locus in the Drosophila Melanogaster adult brain. Open Research Europe (2026).

### Citation
Messina et al. (2026) — Enhancer–promoter proximity predicts transcriptional competence but not transcriptional output in the fly mushroom body.

### Notes

- All scripts use relative paths and light source data only.
- Large raw files are not included in the repository.
- Use cleanup_figures.py to remove all generated plots.
