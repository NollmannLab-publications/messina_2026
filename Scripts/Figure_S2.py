#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S2 Messina et al. 2026
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

print("Starting Figure S2 plotting with light source data...\n")

#%% Figure S2a - Power-law Fits (Median & Contact)
print("Plotting S2a: Power-law Fits")
output_dir = REPO_ROOT / "Figure_2_supp" / "Figure_S2a"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figure1_powerlaw_fits.npz")

# Median Distance vs Genomic Distance
plt.figure(figsize=(8, 6))
plt.scatter(data['genomic_distances'], data['median_distances'],
            color='white', edgecolors='blue', s=50, label='Median Distance')
plt.plot(data['x_values_median'], data['fitted_median'],
         color='red', linewidth=3, linestyle='--',
         label=f'Power-law Fit (b = {data["params_median"][1]:.3f})')

plt.xlabel("Genomic Distance (kb)")
plt.ylabel("Median Distance (nm)")
plt.title("Median Distance vs Genomic Distance with Power-law Fit")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.png', dpi=300)
plt.savefig(output_dir / 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.svg')
plt.close()

# Contact vs Genomic Distance
output_dir = REPO_ROOT / "Figure_2_supp" / "Figure_S2c"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.scatter(data['genomic_distances'], data['contact_values'],
            color='white', edgecolors='blue', s=50, label='Contact')
plt.plot(data['x_values_contact'], data['fitted_contact'],
         color='red', linestyle='--', linewidth=3,
         label=f'Power-law Fit (b = {data["params_contact"][1]:.3f})')

plt.xlabel("Genomic Distance (kb)")
plt.ylabel("Proximity frequency")
plt.title("Contact vs Genomic Distance with Power-law Fit")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'Contact_vs_Genomic_Distance_Power_Law_Fit.png', dpi=300)
plt.savefig(output_dir / 'Contact_vs_Genomic_Distance_Power_Law_Fit.svg')
plt.close()
print("   ✓ S2a Power-law fits done\n")

#%% Figure S2b - HiC vs HiM Correlation
print("Plotting S2b: HiC vs HiM Correlation")
output_dir = REPO_ROOT / "Figure_2_supp" / "Figure_S2b"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figureS2b_correlation.npz")

plt.figure(figsize=(10, 6))
plt.plot(data['thresholds'], data['correlation'], marker='o', markersize=12, linewidth=2,
         color='black', markerfacecolor='blue')

plt.xticks(range(0, max(data['thresholds']) + 1, 100))
plt.xlabel('Threshold (nm)')
plt.ylabel('Correlation')
plt.title('Correlation versus Threshold')
plt.savefig(output_dir / 'HiC_HiM_Correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ S2b Correlation done\n")

print("🎉 All Figure S2 panels completed!")