# Messina et al. (2026) — Source Data & Reproducible Plots

This repository contains **lightweight source data** and **clean plotting scripts** for all figures in **Messina et al. (2026)**.

## Repository Structure

```bash
messina_2026/
├── Figure_1/                  # Generated Figure 1 panels
├── Figure_2/                  # Generated Figure 2 panels
├── Figure_3/                  # Generated Figure 3 panels
├── Figure_4/                  # Generated Figure 4 panels
├── Figure_1_supp/             # Supplementary Figure S1
├── Figure_2_supp/             # Supplementary Figure S2
├── Figure_3_supp/             # Supplementary Figure S3
├── Figure_4_supp/             # Supplementary Figure S4
├── Scripts/                   # Python plotting scripts
├── Source_light/              # Lightweight source data (CSV + .npy)
├── cleanup_figures.py         # Remove all generated figures
├── run_all_figures.py         # Run all plotting scripts
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

### Citation
Messina et al. (2026) — Enhancer–promoter proximity predicts transcriptional competence but not transcriptional output in the fly mushroom body.

### Notes

- All scripts use relative paths and light source data only.
- Large raw files are not included in the repository.
- Use cleanup_figures.py to remove all generated plots.
