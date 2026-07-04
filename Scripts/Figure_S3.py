#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S3 Messina et al. 2026
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

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_LIGHT = REPO_ROOT / "Source_light"

print("Starting Figure S3 plotting with light source data...\n")

#%% Figure S3a - 4M plot for Anchor 11
print("Plotting S3a: 4M plot Anchor 11")
output_dir = REPO_ROOT / "Figures/Figure_3_supp" / "Figure_S3a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
data = np.load(SOURCE_LIGHT / "figureS3a_4M_data.npz")

Matrix_OK107_contact = data['OK107_contact']
Matrix_not_OK107_contact = data['not_OK107_contact']

i = 10  # Anchor 11 (0-based index)

Matrix_4M_OK107 = Matrix_OK107_contact[i, :]
Matrix_4M_not_OK107 = Matrix_not_OK107_contact[i, :]

x_values = np.arange(1, 26)

plt.figure(figsize=(8, 6))
plt.plot(x_values, Matrix_4M_OK107, 'o-', label='KCs', color='green', markersize=8)
plt.plot(x_values, Matrix_4M_not_OK107, 'o-', label='not_KCs', color='blue', markersize=8)

plt.xlabel("Barcodes")
plt.ylabel("Contact Probability")
plt.title("4M Plot OK107 vs not_OK107 - Anchor 11")
plt.axvline(x=11, color='k', linestyle='--', label='RutPr')
plt.legend()
plt.ylim(0.02, 0.65)

plt.savefig(output_dir / '4M_plot_OK107_notOK107_anchor_11.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S3a done\n")

print("🎉 Figure S3 completed!")
