#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 3 Messina et al. 2026
------------------------------------------------------------------------------------------------------------------------

                   ############################
                   #           .-"-.          #   
                   #          /  ~0_0         #
                   #          \_  (__\        #
                   #          //   \\         #
                   #         ((  ®  ))        #
                   #==========""===""=========#
                   #            |||           #
                   #            '|'           #
                   #             |            #
                   ############################

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_LIGHT = REPO_ROOT / "Source_light"

# Add custom functions
sys.path.append(str(SCRIPT_DIR))
from Function_Messina_et_al import *

# Colormaps
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

print("Starting Figure 3 plotting with light source data...\n")
# ===================================================

#%% Figure 3b - Distance Histogram Grid
print("Plotting 3b: Distance Histogram Grid")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3b"
output_dir.mkdir(parents=True, exist_ok=True)

kde_grid = np.load(SOURCE_LIGHT / "figure3_kde_grid.npy")
bins = np.arange(0, 2, 0.05)
n = 25

fig, axs = plt.subplots(figsize=(n, n), ncols=n, nrows=n, sharex=True)

for i in range(n):
    for j in range(n):
        kde_vals = kde_grid[i, j]
        axs[i, j].plot(bins, kde_vals, 'b-', linewidth=1)
        axs[i, j].fill_between(bins, kde_vals, color='lightblue', alpha=0.5)
        
        if np.max(kde_vals) > 0:
            max_x = bins[np.argmax(kde_vals)]
            axs[i, j].axvline(x=max_x, color='red', linestyle='--', linewidth=1)
        
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticklabels([])

plt.savefig(output_dir / 'Histogram_Matrix_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 3b Histogram grid done\n")

#%% Figure 3b - Median PWD Matrix (OK107)
print("Plotting 3b: Median PWD Matrix (OK107)")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3b"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_median = np.load(SOURCE_LIGHT / "figure2_OK107_median.npy")
np.fill_diagonal(Matrix_median, 0)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('OK107 Median PWD matrix')
plt.savefig(output_dir / 'Matrix_PWD_OK107.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 3b Median PWD done\n")

print("🎉 Figure 3 panels completed!")
