#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S4 Messina et al. 2026
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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_LIGHT = REPO_ROOT / "Source_light"

print("Starting Figure S4 plotting with light source data...\n")

#%% Figure S4a - UMAP Rut expression
print("Plotting S4a: UMAP Rut expression")
output_dir = REPO_ROOT / "Figure_4_supp" / "Figure_S4a"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE_LIGHT / 'figure1c_umap_source.csv')
emb_x = df['UMAP1'].values
emb_y = df['UMAP2'].values
kc_mask = df['is_KC'].values.astype(bool)
rut_expr = df['rut_expression'].values

gray_to_green = LinearSegmentedColormap.from_list('gray_to_green', ['gray', 'green'])

plt.figure(figsize=(8, 6))
plt.scatter(emb_x, emb_y, c='gray', s=1, linewidth=0)
mask = rut_expr > 3
scatter = plt.scatter(emb_x[mask], emb_y[mask], c=rut_expr[mask],
                      cmap=gray_to_green, s=5, linewidth=0, vmin=0, vmax=6)

plt.xlim([-16.5, 13])
plt.ylim([-15, 14])
plt.colorbar(scatter, label='Rut expression')
plt.title('UMAP plot - Rut expression')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.savefig(output_dir / 'UMAP_KCs_expression.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ S4a done\n")

#%% Figure S4d - 4M plots (AB, ABp, G)
print("Plotting S4d: 4M plots")
output_dir = REPO_ROOT / "Figure_4_supp" / "Figure_S4d"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figureS4d_4M_data.npz")
x_values = np.arange(1, 26)
anchor = 10  # RutPr anchor

colors = {
    'AB': (113/255, 44/255, 35/255),
    'ABp': (227/255, 91/255, 8/255),
    'G': (255/255, 0, 0),
    'not_KCs': (0, 0, 0)
}

# AB
plt.figure(figsize=(8, 6))
plt.plot(x_values, data['AB_contact'][anchor, :], 'o-', label='AB', color=colors['AB'], markersize=8)
plt.plot(x_values, data['not_OK107_contact'][anchor, :], 'o-', label='not_KCs', color=colors['not_KCs'], markersize=8)
plt.xlabel("Barcodes")
plt.ylabel("Contact Probability")
plt.title("4M Plot AB")
plt.axvline(x=11, color='k', linestyle='--', label='RutPr')
plt.legend()
plt.ylim(0.02, 0.65)
plt.savefig(output_dir / '4M_plot_AB.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '4M_plot_AB.svg', bbox_inches='tight')
plt.close()

# ABp
plt.figure(figsize=(8, 6))
plt.plot(x_values, data['ABp_contact'][anchor, :], 'o-', label='ABp', color=colors['ABp'], markersize=8)
plt.plot(x_values, data['not_OK107_contact'][anchor, :], 'o-', label='not_KCs', color=colors['not_KCs'], markersize=8)
plt.xlabel("Barcodes")
plt.ylabel("Contact Probability")
plt.title("4M Plot ABp")
plt.axvline(x=11, color='k', linestyle='--', label='RutPr')
plt.legend()
plt.ylim(0.02, 0.65)
plt.savefig(output_dir / '4M_plot_ABp.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '4M_plot_ABp.svg', bbox_inches='tight')
plt.close()

# G
plt.figure(figsize=(8, 6))
plt.plot(x_values, data['G_contact'][anchor, :], 'o-', label='G', color=colors['G'], markersize=8)
plt.plot(x_values, data['not_OK107_contact'][anchor, :], 'o-', label='not_KCs', color=colors['not_KCs'], markersize=8)
plt.xlabel("Barcodes")
plt.ylabel("Contact Probability")
plt.title("4M Plot G")
plt.axvline(x=11, color='k', linestyle='--', label='RutPr')
plt.legend()
plt.ylim(0.02, 0.65)
plt.savefig(output_dir / '4M_plot_G.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '4M_plot_G.svg', bbox_inches='tight')
plt.close()

print("   ✓ S4d 4M plots done\n")

print("🎉 Figure S4 completed!")