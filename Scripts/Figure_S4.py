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
import matplotlib.pyplot as plt
import pandas as pd
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

#%% Figure S4e - BC10 log2 contact vs log2 ATAC difference
print("Plotting S4e: BC10 log2 difference")
output_dir = REPO_ROOT / "Figure_4_supp" / "Figure_S4e"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figureS4e_log2_diff_data.npz", allow_pickle=True)
contact_matrices = data['contact_matrices'].item()
integrals = data['integrals'].item()
regions = data['regions'].item()

cell_types = list(contact_matrices.keys())
region_names = list(regions.keys())
target_bc_name = 'BC10'
target_bc_idx = region_names.index(target_bc_name)

pseudocount = 1e-6

log2_integrals = np.log2(np.array([[integrals[ct][rn] for ct in cell_types] for rn in region_names]) + pseudocount)
not_KCs_idx = cell_types.index('not_KCs')
log2_diff_ATAC = log2_integrals - log2_integrals[:, not_KCs_idx][:, np.newaxis]

log2_contact = np.log2(np.stack([contact_matrices[ct] for ct in cell_types], axis=2) + pseudocount)
log2_diff_contact = log2_contact - log2_contact[:, :, not_KCs_idx][:, :, np.newaxis]

custom_colors = {
    'AB': (113/255, 44/255, 35/255),
    'ABp': (227/255, 91/255, 8/255),
    'G': (255/255, 0, 0),
    'not_KCs': (0, 0, 0)
}

annotated_bins = [3,4,5,6]
label_map = {3:'Rut_B4', 4:'Rut_E1', 5:'Rut_E1p', 6:'Rut_E2'}

fig, ax = plt.subplots(figsize=(7, 6))

for ct_idx, cell_type in enumerate(cell_types):
    if cell_type == 'not_KCs':
        continue
    for i in range(len(region_names)):
        if i == target_bc_idx:
            continue
        x = log2_diff_contact[target_bc_idx, i, ct_idx]
        y = log2_diff_ATAC[i, ct_idx]
        if i in annotated_bins:
            ax.scatter(x, y, color=custom_colors[cell_type], s=110, alpha=0.9, zorder=3)
            ax.text(x, y, label_map[i], fontsize=9, ha='left', va='bottom')
        else:
            ax.scatter(x, y, color=custom_colors[cell_type], s=70, alpha=0.2, zorder=1)

ax.set_xlabel('log2(Contact frequency / Non-KCs)')
ax.set_ylabel('log2(ATAC-seq signal / Non-KCs)')
ax.set_title('BC10 log2 contact vs log2 ATAC accessibility difference')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_colors[ct], markersize=8, label=ct)
           for ct in cell_types if ct != 'not_KCs']
ax.legend(handles=handles, title='Cell type', frameon=False)

plt.tight_layout()
plt.savefig(output_dir / 'BC10_log2_contact_vs_log2_ATAC_diff_vs_NonKCs_annotated.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'BC10_log2_contact_vs_log2_ATAC_diff_vs_NonKCs_annotated.svg', bbox_inches='tight')
plt.close()
print("   ✓ S4e log2 difference done\n")

print("🎉 Figure S4 completed!")