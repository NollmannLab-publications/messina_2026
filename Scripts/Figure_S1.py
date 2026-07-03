#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S1 Messina et al. 2026
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
from matplotlib.colors import LinearSegmentedColormap
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_LIGHT = REPO_ROOT / "Source_light"

print("Starting Figure S1 plotting with light source data...\n")

#%% Figure S1a - Violin Plot (Rut ratio KC vs not KC)
print("Plotting S1a: Violin Plot")
output_dir = REPO_ROOT / "Figure_1_supp" / "Figure_S1a"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figureS1a_data.npz")
kc_mask = data['kc_mask']
KC_ratio = data['rut_ratio_KC']
non_KC_ratio = data['rut_ratio_nonKC']

datasets = [non_KC_ratio, KC_ratio]
colors = ['blue', 'green']

fig, ax = plt.subplots(figsize=(12, 6))
vp = ax.violinplot(datasets, showmeans=False, showmedians=True)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[part].set_color('black')

ax.set_xticks([1, 2])
ax.set_xticklabels(['not KCs', 'KC'])
ax.set_title('Violin Plot of Rut Expression in KCs and not KCs')
ax.set_xlabel('Cell Type')
ax.set_ylabel('Rut Expression')
ax.set_ylim(-0.1, 3)

plt.savefig(output_dir / 'Expression_KCs_notKC_KCs_no_outliers.svg', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ S1a done\n")

#%% Figure S1e - 10 Replicates Median PWD
print("Plotting S1e: 10 Replicates Median PWD")
output_dir = REPO_ROOT / "Figure_1_supp" / "Figure_S1b"
output_dir.mkdir(parents=True, exist_ok=True)

Colormap_distance = 'RdGy'

for i in range(1, 11):
    Matrix_rep = np.load(SOURCE_LIGHT / f"S1b_Matrix_Brain_rep_{i}.npy")
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Matrix_rep, cmap=Colormap_distance, vmin=0.1, vmax=0.5)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Brain Median PWD matrix Replicate {i}')
    plt.savefig(output_dir / f'Matrix_Contact_HiM_Brain_rep_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S1b done\n")

#%% Figure S1d - Bootstrapping
print("Plotting S1c: Bootstrapping")
output_dir = REPO_ROOT / "Figure_1_supp" / "Figure_S1c"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_boot = np.load(SOURCE_LIGHT / "S1c_bootstrapping_HiM.npy", allow_pickle=True)

fig, ax = plt.subplots()
vp = ax.violinplot(Matrix_boot, showmeans=False, showmedians=True)
colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow']
for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color)

for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

ax.set_xticks(np.arange(1, 7))
ax.set_xticklabels([5,10,25,50,100,500])
ax.set_ylim(0.35, 1.05)
plt.savefig(output_dir / 'Bootstrapping_HiM.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ S1c done\n")

#%% Figure S1B - N-matrix + Histogram Grid
print("Plotting S1d: N-matrix + Histogram Grid")
output_dir = REPO_ROOT / "Figure_1_supp" / "Figure_S1b"
output_dir.mkdir(parents=True, exist_ok=True)

normalized_matrix = np.load(SOURCE_LIGHT / "figureS1_N_matrix.npy")

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(normalized_matrix, cmap='Blues', vmin=0, vmax=10)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Brain N matrix')
plt.savefig(output_dir / 'N_Matrix_Brain.png', dpi=300, bbox_inches='tight')
plt.close()

# Histogram Grid
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
print("   ✓ S1d done\n")

print("🎉 All Figure S1 panels completed!")