#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Aug, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 1 Messina et al. 2026
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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
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

print("Starting Figure 1 plotting with light source data...\n")
# ===================================================

#%% Figure 1C - UMAP Rut expression
print("Plotting 1C: UMAP Rut expression")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1c"
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
print("   ✓ 1C done\n")

#%% Figure 1E - Median PWD Matrix
print("Plotting 1E: Median PWD Matrix")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1e"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_median = np.load(SOURCE_LIGHT / "figure1e_median_pwd_matrix.npy")

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Brain Median PWD matrix')
plt.savefig(output_dir / 'Matrix_PWD_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1E done\n")

#%% Figure 1G - Contact Matrices
print("Plotting 1G: Contact Matrices")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1g"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_Hi_M_contact = np.load(SOURCE_LIGHT / "figure1e_HiM_contact_matrix.npy")
Hi_C_Brain = np.load(SOURCE_LIGHT / "figure1e_HiC_contact_matrix.npy")

# Hi-M Contact
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_Hi_M_contact, cmap=Colormap_topic, vmin=0, vmax=0.35)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Brain Contact 150nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Brain_rut.png', dpi=300, bbox_inches='tight')
plt.close()

# MicroC Contact
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Hi_C_Brain, cmap=Colormap_topic, vmin=0.00, vmax=0.022)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('MicroC Brain Mohana 4kb Contact')
plt.savefig(output_dir / 'Matrix_Contact_MicroC_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1G done\n")

#%% Figure 1H - Domainogram + Insulation Score
print("Plotting 1H: Domainogram & Insulation")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1h"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_median = np.load(SOURCE_LIGHT / "figure1e_median_pwd_matrix.npy")
Hi_C_Brain = np.load(SOURCE_LIGHT / "figure1e_HiC_contact_matrix.npy")

# Domainogram (Hi-M)
is_matrix = []
for i in range(1, 17):
    sq_size = i
    is_scores = get_insulation_score(Matrix_median, sq_size)
    is_ratio = np.log2(is_scores / np.nanmean(is_scores))
    smoothed_data = smooth_data(is_ratio, 4)
    is_matrix.append(smoothed_data)

is_matrix = np.array(is_matrix)
is_matrix_reduced = is_matrix[:6, :]
is_matrix_reduced[np.isnan(is_matrix_reduced)] = 0

x = np.arange(1, is_matrix_reduced.shape[1] + 1)
y = np.arange(1, is_matrix_reduced.shape[0] + 1)
X, Y = np.meshgrid(x, y)
x2 = np.linspace(1, is_matrix_reduced.shape[1], int(is_matrix_reduced.shape[1] * 100))
y2 = np.linspace(1, is_matrix_reduced.shape[0], int(is_matrix_reduced.shape[0] * 100))
X2, Y2 = np.meshgrid(x2, y2)

from scipy.interpolate import griddata
out_data = griddata((X.flatten(), Y.flatten()), is_matrix_reduced.flatten(), (X2, Y2), method='linear')

plt.imshow(out_data, aspect='auto', extent=[1, is_matrix_reduced.shape[1], 1, is_matrix_reduced.shape[0]],
           origin='lower', cmap='coolwarm')
plt.clim(-0.15, 0.15)
plt.savefig(output_dir / 'Domainogram_HiM_rut.png', dpi=300, bbox_inches='tight')
plt.close()

# Insulation score line plot
is_matrix_HiM = []
is_matrix_HiC = []
for i in range(1, 17):
    sq_size = i
    is_scores = get_insulation_score(Matrix_median, sq_size)
    is_ratio = np.log2(is_scores / np.nanmean(is_scores))
    smoothed = smooth_data(is_ratio, 4)
    is_matrix_HiM.append(smoothed)
    
    is_scores_hic = get_insulation_score(1. / Hi_C_Brain, sq_size)
    is_ratio_hic = np.log2(is_scores_hic / np.nanmean(is_scores_hic))
    smoothed_hic = smooth_data(is_ratio_hic, 4)
    is_matrix_HiC.append(smoothed_hic)

is_matrix_HiM = np.array(is_matrix_HiM)
is_matrix_HiC = np.array(is_matrix_HiC)

Window_HiM = is_matrix_HiM[2, :]
Window_HiC = is_matrix_HiC[2, :]

def normalize_to_range(arr):
    return 2 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) - 1

Window_HiM_n = normalize_to_range(Window_HiM)
Window_HiC_n = normalize_to_range(Window_HiC)

x = np.arange(len(Window_HiM)) + 1
plt.figure(figsize=(10, 3))
plt.plot(x, Window_HiM_n, label="HiM", color="blue", linewidth=3)
plt.plot(x, Window_HiC_n, label="HiC", color="red", linewidth=3)
plt.title("Normalized insulation score")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.savefig(output_dir / 'Insulation_score_window_3_HiC_HiM.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1H done\n")

#%% Figure 1I - Log2 Observed/Expected
print("Plotting 1I: Log2 O/E")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1i"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_Hi_M_contact = np.load(SOURCE_LIGHT / "figure1f_HiM_contact_matrix.npy")
distance_matrix = np.load(SOURCE_LIGHT / "figure1f_distance_matrix.npy")
fitted_contact = np.load(SOURCE_LIGHT / "figure1f_fitted_contact.npy")
x_values_contact = np.load(SOURCE_LIGHT / "figure1f_x_values_contact.npy")

interpolator = interp1d(x_values_contact, fitted_contact, kind='linear', fill_value="extrapolate")
expected = interpolator(distance_matrix / 1000)
log2_oe = np.log2(Matrix_Hi_M_contact / expected)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(log2_oe, cmap=Colormap_topic, vmin=-1, vmax=1)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Log2 Observed vs Expected contact Hi-M')
plt.savefig(output_dir / 'OE_Matrix_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1I done\n")

print("🎉 All Figure 1 panels completed!")