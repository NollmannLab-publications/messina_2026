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
REPO_ROOT  = SCRIPT_DIR.parent
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

Matrix_median = np.load(SOURCE_LIGHT / "figure1e_median_pwd_matrix.npy")   # reused from 1b

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Brain Median PWD matrix')
plt.savefig(output_dir / 'Matrix_PWD_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1E done\n")

#%% Figure 1G - Contact Matrices (Hi-M + Hi-C)
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

# Hi-C Contact
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Hi_C_Brain, cmap=Colormap_topic, vmin=0.00, vmax=0.022)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Micro-C Brain Mohana 4kb Contact')
plt.savefig(output_dir / 'Matrix_Contact_MicroC_Brain.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 1G done\n")

#%% Figure 1H - Domainogram + Insulation Score
print("Plotting 1H: Domainogram & Insulation")
output_dir = REPO_ROOT / "Figure_1" / "Figure_1h"
output_dir.mkdir(parents=True, exist_ok=True)

Matrix_median = np.load(SOURCE_LIGHT / "figure1e_median_pwd_matrix.npy")   # reuse
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

# Insulation score line plot (window 3)
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
distance_matrix     = np.load(SOURCE_LIGHT / "figure1f_distance_matrix.npy")
fitted_contact      = np.load(SOURCE_LIGHT / "figure1f_fitted_contact.npy")
x_values_contact    = np.load(SOURCE_LIGHT / "figure1f_x_values_contact.npy")

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






# #%%

# #%% Figure 1E SUpp?

# # ====================== RELATIVE PATHS ======================
# SCRIPT_DIR = Path(__file__).resolve().parent
# REPO_ROOT  = SCRIPT_DIR.parent

# source_dir = REPO_ROOT / "Source_light"
# output_dir = REPO_ROOT / "Figure_1" / "Figure_1e"

# output_dir.mkdir(parents=True, exist_ok=True)
# # ===========================================================

# # Load light source data
# Matrix_median = np.load(source_dir / "figure1e_median_pwd_matrix.npy")
# df = pd.read_csv(source_dir / "figure1e_genomic_loci.csv")

# # Create genomic distance matrix
# num_bins = len(df)
# distance_matrix = np.zeros((num_bins, num_bins))
# for i in range(num_bins):
#     for j in range(num_bins):
#         distance_matrix[i, j] = abs(df['Center'][i] - df['Center'][j])

# genomic_distances = distance_matrix.flatten() / 1000          # in kb
# median_distances = Matrix_median.flatten()

# # Filter valid data
# valid_indices = np.where(genomic_distances > 0)[0]
# valid_genomic = genomic_distances[valid_indices]
# valid_median  = median_distances[valid_indices]

# # Power-law fit
# def power_law(x, a, b):
#     return a * np.power(x, b)

# popt, _ = curve_fit(power_law, valid_genomic, valid_median, p0=[0.1, 0.3])
# a_opt, b_opt = popt

# print(f"Power-law fit: a = {a_opt:.4f}, b = {b_opt:.4f}")

# # Plot
# x_fit = np.linspace(valid_genomic.min(), valid_genomic.max(), 50)
# y_fit = power_law(x_fit, a_opt, b_opt)

# plt.figure(figsize=(8, 6))
# plt.scatter(valid_genomic, valid_median, color='white', edgecolors='blue', s=50, label='Median Distance')
# plt.plot(x_fit, y_fit, color='red', linewidth=3, linestyle='--', 
#          label=f'Power-law Fit (b = {b_opt:.3f})')

# plt.xlabel("Genomic Distance (kb)")
# plt.ylabel("Median Distance (nm)")
# plt.title("Median Distance vs Genomic Distance with Power-law Fit")
# plt.legend()
# plt.tight_layout()

# plt.savefig(output_dir / 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.png', dpi=300)
# plt.savefig(output_dir / 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.svg')
# plt.show()

# print(f"✅ Plot saved in: {output_dir}")


# #%%









# #%% Figure 1d








# #%% Figure 1b

# #Set Directory
# Figure_name = 'Figure_1b'
# Figure_folder = Input_folder+'/'+Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)

# #Load Data
# Matrix_Hi_M_brain1 = np.load(Input_folder+'/Source_data/'+'Traces_combined_all_traces_KC_G_split_LEAR_Matrix_PWDscMatrix.npy')
# Matrix_Hi_M_brain2 = np.load(Input_folder+'/Source_data/'+'Traces_combined_all_traces_KC_AB_split_LEAR_Matrix_PWDscMatrix.npy')
# Matrix_Hi_M_brain3 = np.load(Input_folder+'/Source_data/'+'Traces_combined_all_traces_KC_ABp_split_LEAR_Matrix_PWDscMatrix.npy')
# Matrix_Hi_M_brain4 = np.load(Input_folder+'/Source_data/'+'Traces_combined_all_traces_KC_all_split_LEAR_Matrix_PWDscMatrix.npy')
# Matrix_Hi_M_brain5 = np.load(Input_folder+'/Source_data/'+'Traces_combined_all_traces_not_KC_split_imputed_LEAR_Matrix_PWDscMatrix.npy')
# Matrix_Hi_M_brain = np.concatenate(
#     [Matrix_Hi_M_brain1, Matrix_Hi_M_brain2, Matrix_Hi_M_brain3, Matrix_Hi_M_brain4, Matrix_Hi_M_brain5],
#     axis=2
# )

# Matrix_Hi_C_brain = np.load(Input_folder+'/Source_data/'+'rut_Hi-C_Mohana_4kb_interpolated.npy')

# #Compute median 
# Matrix_Hi_M_brain_contact,_ = PWD_to_Contact(Matrix_Hi_M_brain,1000,150)
# Matrix_median = np.nanmedian(Matrix_Hi_M_brain,axis=2)
# np.fill_diagonal(Matrix_median, 0)

# # Data for Locus_N, Start, and End positions
# data = {
#     'Locus_N': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
#     'Start': [14785864, 14789398, 14792433, 14795381, 14799003, 14802255, 14805412, 14809057, 14812113, 14814823,
#               14824792, 14829662, 14833223, 14838104, 14841518, 14844805, 14848436, 14851801, 14853799, 14857068,
#               14860470, 14864234, 14865932, 14869591, 14873097],
#     'End': [14789298, 14792430, 14795380, 14798629, 14802202, 14805371, 14809056, 14812112, 14814817, 14818656,
#             14829604, 14833222, 14835453, 14841356, 14844734, 14848435, 14851800, 14853798, 14856954, 14860469,
#             14864233, 14865931, 14869590, 14873096, 14876400]
# }
# df = pd.DataFrame(data)

# # Calculate the center of each bin
# df['Center'] = (df['Start'] + df['End']) / 2

# # Create the distance matrix
# num_bins = len(df)
# distance_matrix = np.zeros((num_bins, num_bins))

# # Fill the matrix with pairwise distances
# for i in range(num_bins):
#     for j in range(num_bins):
#         distance_matrix[i, j] = abs(df['Center'][i] - df['Center'][j])

# # Convert distance matrix to DataFrame for better visualization
# distance_matrix_df = pd.DataFrame(distance_matrix, columns=df['Locus_N'], index=df['Locus_N'])

# # Convert genomic distances from bp to kb by dividing by 1000
# genomic_distances = distance_matrix_df.values.flatten() / 1000  # Convert to kb
# median_distances = Matrix_median.flatten()

# # Remove NaN and infinite values from the data before fitting
# #valid_indices_median = np.isfinite(genomic_distances) & np.isfinite(median_distances)
# #valid_indices_median = np.where(genomic_distances > 0)[0]
# valid_indices_median = np.where(genomic_distances > 0)[0]

# valid_genomic_distances_median = genomic_distances[valid_indices_median]
# valid_median_distances = median_distances[valid_indices_median]

# # Define the power-law function: y = a * x^b
# def power_law(x, a, b):
#     return a * np.power(x, b)

# # Perform non-linear curve fitting
# popt, pcov = curve_fit(power_law, valid_genomic_distances_median, valid_median_distances)

# # Extract the optimized parameters a and b
# a_opt, b_opt = popt

# #a_opt = 0.1105
# #b_opt = 0.289

# print(f"Optimized parameters: a = {a_opt:.2f}, b = {b_opt:.2f}")

# # Generate x values for plotting the fit
# x_values_median = np.linspace(valid_genomic_distances_median.min(), valid_genomic_distances_median.max(), 20)
# fitted_median = power_law(x_values_median, a_opt, b_opt)

# # Plot the original data and the power-law fit
# plt.figure(figsize=(8, 6))
# plt.scatter(valid_genomic_distances_median, valid_median_distances, color='white', alpha=1, label='Median Distance',edgecolors='blue',s=50)

# # Plot the power-law fit
# plt.plot(x_values_median, fitted_median, color='red', label=f'Power-law Fit (b = {b_opt:.2f})',linewidth=3,linestyle = '--')

# # Add labels and title
# plt.xlabel("Genomic Distance (kb)")
# plt.ylabel("Median Distance in nm")
# plt.title("Median Distance vs Genomic Distance (kb) with Power-law Fit")
# plt.legend()
# plt.tight_layout()
# plt.savefig(Figure_folder + '/' + 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.png')
# plt.savefig(Figure_folder + '/' + 'Median_Distance_vs_Genomic_Distance_Power_Law_Fit.svg')
# plt.show()

# #%% Figure 1c

# # Set Directory
# Figure_name = 'Figure_1c'
# Figure_folder = Input_folder + '/' + Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)
    
# contact_values = Matrix_Hi_M_brain_contact.flatten()   

# valid_genomic_distances_contact = genomic_distances[valid_indices_median]
# valid_contact_values = contact_values[valid_indices_median]

# popt, pcov = curve_fit(power_law, valid_genomic_distances_contact, valid_contact_values)
# a_opt, b_opt = popt

# # Generate x values for plotting the fit
# x_values_contact = np.linspace(valid_genomic_distances_contact.min(), valid_genomic_distances_contact.max(), 500)
# fitted_contact = power_law(x_values_contact, a_opt, b_opt)

# # Plot the original data (Contact vs Genomic Distance) and the power-law fit
# plt.figure(figsize=(8, 6))
# # Scatter plot with white fill and blue edge for circles, size 50
# plt.scatter(valid_genomic_distances_contact, valid_contact_values, color='white', alpha=1, label='Contact', edgecolors='blue', s=50)

# # Plot the power-law fit with a dashed red line
# plt.plot(x_values_contact, fitted_contact, color='red', linestyle='--', label=f'Power-law Fit (b = {b_opt:.2f})', linewidth=3)

# # Add labels and title
# plt.xlabel("Genomic Distance (kb)")
# plt.ylabel("Proximity frequency")
# plt.title("Contact vs Genomic Distance (kb) with Power-law Fit")
# plt.legend()
# plt.tight_layout()
# plt.savefig(Figure_folder + '/' + 'Contact_vs_Genomic_Distance_Power_Law_Fit.png')
# plt.savefig(Figure_folder + '/' + 'Contact_vs_Genomic_Distance_Power_Law_Fit.svg')
# plt.show()

# #%% Figure 1d

# # Set Directory
# Figure_name = 'Figure_1d'
# Figure_folder = Input_folder + '/' + Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)

# # Plot figure 
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('Brain Median PWD matrix')
# plt.savefig(Figure_folder+'/'+'Matrix_PWD_Brain.png',dpi=300)
# plt.close()

# # Domainogram
# is_matrix = []

# for i in range(1, 17):
#     sq_size = i  # size of square for insulation score in bins
#     is_scores = get_insulation_score(Matrix_median, sq_size)
#     is_ratio = np.log2(is_scores / np.nanmean(is_scores))
#     smoothed_data = smooth_data(is_ratio, 4)
#     is_matrix.append(smoothed_data)

# is_matrix = np.array(is_matrix)
# is_matrix_reduced = is_matrix[:6, :]
# is_matrix_reduced[np.isnan(is_matrix_reduced)] = 0

# x = np.arange(1, is_matrix_reduced.shape[1] + 1)
# y = np.arange(1, is_matrix_reduced.shape[0] + 1)
# X, Y = np.meshgrid(x, y)

# x2 = np.linspace(1, is_matrix_reduced.shape[1], num=int(is_matrix_reduced.shape[1] * 100))
# y2 = np.linspace(1, is_matrix_reduced.shape[0], num=int(is_matrix_reduced.shape[0] * 100))
# X2, Y2 = np.meshgrid(x2, y2)

# out_data = griddata((X.flatten(), Y.flatten()), is_matrix_reduced.flatten(), (X2, Y2), method='linear')
# plt.imshow(out_data, aspect='auto', extent=[1, is_matrix_reduced.shape[1], 1, is_matrix_reduced.shape[0]], origin='lower', cmap='coolwarm')
# plt.clim(-0.15, 0.15)
# plt.savefig(Figure_folder+'/'+'Domainogram_HiM_rut.png',dpi=300)

# #%% Figure 1e

# #Set Directory
# Figure_name = 'Figure_1e'
# Figure_folder = Input_folder+'/'+Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)

# Matrix_Hi_M_brain_contact,_ = PWD_to_Contact(Matrix_Hi_M_brain,1000,150)

# # Plot figure (Upper plot)
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(Matrix_Hi_M_brain_contact, cmap=Colormap_topic, vmin=0, vmax=0.35)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('Brain Contact 150nm')
# plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Brain_rut.png',dpi=300)
# #plt.close()

# #Load Data
# Hi_C_Brain =  np.load(Input_folder+'/Source_data/'+'rut_Hi-C_Mohana_4kb_interpolated.npy')

# # Plot figure (Lower plot)
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(Hi_C_Brain, cmap=Colormap_topic, vmin=0.00, vmax=0.022)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('HiC Brain Mohana 4kb Contact')
# plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiC_Brain.png',dpi=300)
# plt.close()

# #%% Figure 1f

# #Set Directory
# Figure_name = 'Figure_1f'
# Figure_folder = Input_folder+'/'+Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)

# interpolator = interp1d(x_values_contact, fitted_contact, kind='linear', fill_value="extrapolate")
# expected_contact_matrix = interpolator(distance_matrix/1000)
# log2_observed_expected = np.log2(Matrix_Hi_M_brain_contact / (expected_contact_matrix))

# # Plot figure (Lower plot)
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(log2_observed_expected, cmap=Colormap_topic, vmin=-1, vmax=1)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('Log2 Observed vs Expected contact Hi-M')
# plt.savefig(Figure_folder+'/'+'OE_Matrix_Brain.png',dpi=300)
# plt.close()


# #%%

# # # Set Directory
# # Figure_name = 'Figure_1d'
# # Figure_folder = Input_folder + '/' + Figure_name

# # if not os.path.exists(Figure_folder):
# #     os.makedirs(Figure_folder)

# # # Plot figure 
# # fig, axs = plt.subplots(figsize=(10, 8))
# # im1 = plt.imshow(Matrix_Hi_C_brain, cmap=Colormap_distance, vmin=0, vmax=0.02)
# # fig.colorbar(im1, ax=axs, shrink=1)  
# # plt.title('Brain Median PWD matrix')
# # #plt.savefig(Figure_folder+'/'+'Matrix_PWD_Brain.png',dpi=300)
# # #plt.close()

# #%%
# # # Domainogram HiC
# # is_matrix = []

# # for i in range(1, 17):
# #     sq_size = i  # size of square for insulation score in bins
# #     is_scores = get_insulation_score(1./Matrix_Hi_C_brain, sq_size)
# #     is_ratio = np.log2(is_scores / np.nanmean(is_scores))
# #     smoothed_data = smooth_data(is_ratio, 4)
# #     is_matrix.append(smoothed_data)

# # is_matrix = np.array(is_matrix)
# # is_matrix_reduced = is_matrix[:6, :]
# # is_matrix_reduced[np.isnan(is_matrix_reduced)] = 0

# # x = np.arange(1, is_matrix_reduced.shape[1] + 1)
# # y = np.arange(1, is_matrix_reduced.shape[0] + 1)
# # X, Y = np.meshgrid(x, y)

# # x2 = np.linspace(1, is_matrix_reduced.shape[1], num=int(is_matrix_reduced.shape[1] * 100))
# # y2 = np.linspace(1, is_matrix_reduced.shape[0], num=int(is_matrix_reduced.shape[0] * 100))
# # X2, Y2 = np.meshgrid(x2, y2)

# # out_data = griddata((X.flatten(), Y.flatten()), is_matrix_reduced.flatten(), (X2, Y2), method='linear')
# # plt.imshow(out_data, aspect='auto', extent=[1, is_matrix_reduced.shape[1], 1, is_matrix_reduced.shape[0]], origin='lower', cmap='coolwarm')
# # plt.clim(-0.15, 0.15)
# # #plt.savefig(Figure_folder+'/'+'Domainogram_HiM_rut.png',dpi=300)

