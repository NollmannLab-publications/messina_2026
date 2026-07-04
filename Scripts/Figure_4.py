#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 4 Messina et al. 2026
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
import json
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_LIGHT = REPO_ROOT / "Source_light"

print("Starting Figure 4 plotting with light source data...\n")

#%% Figure 4a - Violin Plot
print("Plotting 4a: Violin Plot")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4a"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figure4a_violin_data.npz")
datasets = [data['not_kc'], data['kc_G'], data['kc_AB'], data['kc_ABp']]
colors = ['gray', 'blue', 'red', 'green']
labels = ['Not KC', 'G-KC', 'A/B-KC', 'A/B*-KC']

fig, ax = plt.subplots(figsize=(12, 6))
vp = ax.violinplot(datasets, showmeans=False, showmedians=True)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)
for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[part].set_color('black')

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels)
ax.set_title('Rut / Act5C Ratio across Cell Types')
ax.set_xlabel('Cell Type')
ax.set_ylabel('Ratio (Rut / Act5C)')
ax.set_ylim(-0.1, 5.5)

plt.savefig(output_dir / 'Expression_KC_and_notKC_Types.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Expression_KC_and_notKC_Types.svg', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 4a done\n")

#%% Figure 4c - KC subtypes Contact Maps
print("Plotting 4c: KC subtypes Contact Maps")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4c"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figure4_kc_contact_maps.npz")

for name, title in [('G_contact', 'KCs G'), ('AB_contact', 'KCs AB'), ('ABp_contact', 'KCs ABp')]:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(data[name], cmap='RdBu_r', vmin=0, vmax=0.35)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'{title} Contact 150nm')
    plt.savefig(output_dir / f'Contact_KCs_{name.split("_")[0]}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 4c done\n")

#%% Figure 4e - Distance vs ATAC-seq
print("Plotting 4e: Distance vs ATAC-seq")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4e"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figure4e_light_data.npz")
with open(SOURCE_LIGHT / "figure4e_metadata.json", "r") as f:
    metadata = json.load(f)

custom_colors = {
    'not_KCs': (0.0, 0.4, 0.8),
    'AB': (0.44, 0.17, 0.14),
    'ABp': (0.89, 0.36, 0.04),
    'G': (0.9, 0.1, 0.1)
}
region_markers = {'Rut_E1': 'o', 'Rut_E1p': 's', 'Rut_E2': '^'}
region_labels = {
    'Rut_E1': r'$P_{rut} - E_1$',
    'Rut_E1p': r'$P_{rut} - E_1^\prime$',
    'Rut_E2': r'$P_{rut} - E_2$'
}

print(f"Loaded {len(metadata['points'])} points")

fig, ax = plt.subplots(figsize=(11, 8))

for p in metadata['points']:
    ct = p['cell_type']
    region = p.get('region')
    color = custom_colors.get(ct, (0.5, 0.5, 0.5))
  
    if region in region_markers:
        marker = region_markers[region]
        ax.errorbar(p['x'], p['y'], xerr=p['xerr'], yerr=p['yerr'],
                    fmt=marker, color=color, markersize=14,
                    capsize=4, capthick=1.8, mec='black', mfc=color, alpha=0.95, zorder=5)
    else:
        ax.errorbar(p['x'], p['y'], xerr=p['xerr'], yerr=p['yerr'],
                    fmt='o', color=color, markersize=14,
                    capsize=4, capthick=1.5, alpha=0.95, zorder=4)

ax.plot(data['all_xgrid'], data['all_yhat'], 'r--', lw=2.8, label=f'Pearson All (r={metadata["all_r"]:.2f})')
ax.fill_between(data['all_xgrid'], data['all_yhat'] - data['all_ci'], data['all_yhat'] + data['all_ci'], color='red', alpha=0.18)
ax.plot(data['kc_xgrid'], data['kc_yhat'], 'k--', lw=2.5, label=f'Pearson KCs only (r={metadata["kc_r"]:.2f})')
ax.fill_between(data['kc_xgrid'], data['kc_yhat'] - data['kc_ci'], data['kc_yhat'] + data['kc_ci'], color='black', alpha=0.13)

ax.set_xlabel("Median Pairwise Distance (μm)", fontsize=22)
ax.set_ylabel("ATAC-seq Integral Signal", fontsize=22)
ax.set_xlim(0.17, 0.29)
ax.set_ylim(0, 140)

handles_main = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=label)
                for ct, c in custom_colors.items()
                for label in [{'not_KCs': 'non KCs', 'AB': 'KCs αβ', 'ABp': "KCs α'β'", 'G': 'KCs γ'}.get(ct, ct)]]

handles_main += [
    plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5, label=f'Pearson All (r={metadata["all_r"]:.2f})'),
    plt.Line2D([0], [0], color='black', linestyle='--', lw=2.5, label=f'Pearson KCs only (r={metadata["kc_r"]:.2f})')
]

leg1 = ax.legend(handles=handles_main, loc='upper right', fontsize=14, title="")

handles_rut = [plt.Line2D([0], [0], marker=m, color='black', markerfacecolor='black', markersize=14, markeredgecolor='black', label=region_labels[r])
               for r, m in region_markers.items()]

leg2 = ax.legend(handles=handles_rut, loc='upper right', fontsize=14, bbox_to_anchor=(1, 0.7))
ax.add_artist(leg1)

plt.tight_layout()
plt.savefig(output_dir / 'Figure_4a_final.png', dpi=400, bbox_inches='tight')
plt.savefig(output_dir / 'Figure_4a_final.svg', bbox_inches='tight')
plt.show()
plt.close()
print("   ✓ 4e done\n")

#%% Figure 4f - Correlation plot
print("Plotting 4f: Correlation plot")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4f"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(SOURCE_LIGHT / "figure4f_correlation_data.npz")
with open(SOURCE_LIGHT / "figure4f_metadata.json", "r") as f:
    metadata = json.load(f)

custom_colors = {
    'not_KCs': (0.0, 0.4, 0.8),
    'AB': (0.44, 0.17, 0.14),
    'ABp': (0.89, 0.36, 0.04),
    'G': (0.9, 0.1, 0.1)
}
region_markers = {'Rut_E1': 'o', 'Rut_E1p': 's', 'Rut_E2': '^'}
region_labels = {
    'Rut_E1': r'$P_{rut} - E_1$',
    'Rut_E1p': r'$P_{rut} - E_1^\prime$',
    'Rut_E2': r'$P_{rut} - E_2$'
}

fig, ax = plt.subplots(figsize=(11, 8))

for p in metadata['points']:
    ct = p['cell_type']
    region = p.get('region')
    color = custom_colors.get(ct, (0.5, 0.5, 0.5))
   
    if region in region_markers:
        marker = region_markers[region]
        ax.errorbar(p['x'], p['y'], xerr=p['xerr'], yerr=p['yerr'],
                    fmt=marker, color=color, markersize=14,
                    capsize=4, capthick=1.8, mec='black', mfc=color,
                    alpha=0.95, zorder=5)
    else:
        ax.errorbar(p['x'], p['y'], xerr=p['xerr'], yerr=p['yerr'],
                    fmt='o', color=color, markersize=14,
                    capsize=4, capthick=1.5, alpha=0.95, zorder=4)

ax.plot(data['all_xgrid'], data['all_yhat'], 'r--', lw=2.8,
        label=f'Pearson All (r={metadata["all_r"]:.2f})')
ax.fill_between(data['all_xgrid'],
                data['all_yhat'] - data['all_ci'],
                data['all_yhat'] + data['all_ci'],
                color='red', alpha=0.18)

ax.plot(data['kc_xgrid'], data['kc_yhat'], 'k--', lw=2.5,
        label=f'Pearson KCs only (r={metadata["kc_r"]:.2f})')
ax.fill_between(data['kc_xgrid'],
                data['kc_yhat'] - data['kc_ci'],
                data['kc_yhat'] + data['kc_ci'],
                color='black', alpha=0.13)

ax.set_xlabel("Median Pairwise Distance (μm)", fontsize=22)
ax.set_ylabel("Normalized rut expression (scRNA-seq)", fontsize=22)
ax.set_xlim(0.17, 0.29)
ax.set_ylim(0, 1.25)

handles_main = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9, label=ct)
                for ct, c in custom_colors.items()]

handles_main += [
    plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5, label=f'Pearson All (r={metadata["all_r"]:.2f})'),
    plt.Line2D([0], [0], color='black', linestyle='--', lw=2.5, label=f'Pearson KCs only (r={metadata["kc_r"]:.2f})')
]

leg1 = ax.legend(handles=handles_main, loc='upper right', fontsize=14)

handles_rut = [plt.Line2D([0], [0], marker=m, color='black', markerfacecolor='black', markersize=10, markeredgecolor='black', label=region_labels[r])
               for r, m in region_markers.items()]

leg2 = ax.legend(handles=handles_rut, loc='upper right', fontsize=14, bbox_to_anchor=(1, 0.7))
ax.add_artist(leg1)

plt.tight_layout()
plt.savefig(output_dir / 'Figure_4F_final.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("   ✓ 4f done\n")

print("🎉 All Figure 4 panels completed!")
