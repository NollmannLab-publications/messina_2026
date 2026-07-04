#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 2 Messina et al. 2026
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

print("Starting Figure 2 plotting with light source data...\n")
# ===================================================

# Load light data
Matrix_OK107_contact = np.load(SOURCE_LIGHT / "figure2_OK107_contact.npy")
Matrix_not_OK107_contact = np.load(SOURCE_LIGHT / "figure2_notOK107_contact.npy")
Matrix_OK107_median = np.load(SOURCE_LIGHT / "figure2_OK107_median.npy")
Matrix_not_OK107_median = np.load(SOURCE_LIGHT / "figure2_notOK107_median.npy")

#%% Figure 2c - Contact maps
print("Plotting 2c: Contact maps")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2c"
output_dir.mkdir(parents=True, exist_ok=True)

# OK107
fig, axs = plt.subplots(figsize=(10, 8))
axs.imshow(Matrix_OK107_contact, cmap=Colormap_topic, vmin=0, vmax=0.35)
fig.colorbar(axs.images[0], ax=axs, shrink=1)
plt.title('OK107 Contact 150nm')
plt.savefig(output_dir / 'Contact_OK107.png', dpi=300, bbox_inches='tight')
plt.close()

# not OK107
fig, axs = plt.subplots(figsize=(10, 8))
axs.imshow(Matrix_not_OK107_contact, cmap=Colormap_topic, vmin=0, vmax=0.35)
fig.colorbar(axs.images[0], ax=axs, shrink=1)
plt.title('not OK107 Contact 150nm')
plt.savefig(output_dir / 'Contact_not_OK107.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 2c done\n")

#%% Figure 2d - Differential PWD (log2 ratio)
print("Plotting 2d: Differential PWD")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2d"
output_dir.mkdir(parents=True, exist_ok=True)

np.fill_diagonal(Matrix_OK107_median, np.nan)
np.fill_diagonal(Matrix_not_OK107_median, np.nan)

Differential_matrix = np.log2(Matrix_OK107_median / Matrix_not_OK107_median)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Differential_matrix, cmap='RdBu', vmin=-0.4, vmax=0.05)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Ratio PWD matrix (OK107 / not_OK107)')
plt.savefig(output_dir / 'Matrix_OK107_not_OK107_differential_PWD_log2.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 2d done\n")

#%% Figure 2e - Volcano plot
print("Plotting 2e: Volcano plot")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2e"
output_dir.mkdir(parents=True, exist_ok=True)

differential_matrix = np.load(SOURCE_LIGHT / "figure2_volcano_differential.npy")
pvalues = np.load(SOURCE_LIGHT / "figure2_volcano_pvalues.npy")

y_values = -np.log10(pvalues)
cmap = plt.get_cmap('RdBu')
colors = [cmap(0.0) if x < 0 else cmap(1.0) for x in differential_matrix.flatten()]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(differential_matrix.flatten(), y_values.flatten(), c=colors, alpha=0.33, s=15)

plt.xlim(-0.5, 0.5)
plt.ylim(0, 28)

# Annotations
annotations = [(4,10,'Rut_E1'), (5,10,'Rut_E1E2'), (6,10,'Rut_E2'),
               (2,10,'Rut_E0a'), (3,10,'Rut_E0b')]

for idx, idy, label in annotations:
    x = differential_matrix[idx, idy]
    y = y_values[idx, idy]
    offset = (25, -50) if idx % 2 == 1 else (-25, 50)
    ax.annotate(label, (x, y), xytext=offset, textcoords='offset points',
                ha='center', arrowprops=dict(arrowstyle='->', color='black'))

plt.title('Volcano plot OK107 vs not_OK107')
plt.xlabel('Log2 Fold Change (PWD)')
plt.ylabel('-log10(p-value)')
plt.savefig(output_dir / 'Volcano_plot_OK107_notOK107_PWD.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 2e done\n")

print("🎉 All Figure 2 panels completed!")
