#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:50:20 2026

@author: olivier
"""

#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder

#%% Figure 1C 
# ================== CONFIG ==================
loom_path = '/home/olivier/Desktop/Marion_project/Paper_figures_marion/Source_data/Figure_2/Source_data//Davie_Janssens_Koldere_et_al_2018_AdultBrain.loom'
output_dir = '/home/olivier/Desktop/Marion_project/Paper_figures_marion/Source_data/Source_light/'
os.makedirs(output_dir, exist_ok=True)
# ===========================================

#%%
with h5py.File(loom_path, 'r') as f:
    matrix = f['matrix'][:]
    emb_x = f['col_attrs']['Embeddings_X'][:]
    emb_y = f['col_attrs']['Embeddings_Y'][:]
    leiden = f['col_attrs']['leiden'][:]
    cell_type = f['col_attrs']['cell_type'][:]
    gene_names = f['row_attrs']['Gene'][:]

# Clean data (your original logic)
if isinstance(emb_x[0], (tuple, np.void)):
    emb_x = np.array([item[0] for item in emb_x])
    emb_y = np.array([item[0] for item in emb_y])
    leiden = np.array([item[0] for item in leiden])

emb_x = emb_x.astype(np.float32)
emb_y = emb_y.astype(np.float32)
cell_type = np.array([str(ct, 'utf-8') if isinstance(ct, bytes) else str(ct) for ct in cell_type])

leiden_numeric = LabelEncoder().fit_transform(leiden)

# KC mask
kc_mask = np.array(['KC' in ct for ct in cell_type])

# Gene expression
rut_idx = np.where(gene_names == b'rut')[0][0]
act5c_idx = np.where(gene_names == b'Act5C')[0][0]  # if needed later

rut_expr = matrix[rut_idx, :].astype(np.float32)

# ================== SAVE SOURCE DATA ==================
# Main UMAP coordinates + metadata
df_umap = pd.DataFrame({
    'UMAP1': emb_x,
    'UMAP2': emb_y,
    'leiden': leiden,
    'cell_type': cell_type,
    'is_KC': kc_mask.astype(int),
    'rut_expression': rut_expr
})
df_umap.to_csv(os.path.join(output_dir, 'figure1c_umap_source.csv'), index=False)

# Optional: gene names for reference
pd.Series(gene_names).to_csv(os.path.join(output_dir, 'gene_names.csv'), index=False, header=False)

print(f"Source data saved to {output_dir}")
print(f"File size: {os.path.getsize(os.path.join(output_dir, 'figure1c_umap_source.csv')) / 1e6:.1f} MB")

#%%

INPUT_DATA = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Version_X_Lausanne/Source_data/Figure_1/Source_data/"

# ===========================================================

print("Loading raw matrices...")

# Load the original matrices
Matrix_Hi_M_brain1 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_G_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_Hi_M_brain2 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_AB_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_Hi_M_brain3 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_ABp_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_Hi_M_brain4 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_all_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_Hi_M_brain5 = np.load(INPUT_DATA + 'Traces_combined_all_traces_not_KC_split_imputed_LEAR_Matrix_PWDscMatrix.npy')

Matrix_Hi_M_brain = np.concatenate(
    [Matrix_Hi_M_brain1, Matrix_Hi_M_brain2, Matrix_Hi_M_brain3, Matrix_Hi_M_brain4, Matrix_Hi_M_brain5],
    axis=2
)

print(f"Combined matrix shape: {Matrix_Hi_M_brain.shape}")

# Compute median matrix (this is the key data for the plot)
Matrix_median = np.nanmedian(Matrix_Hi_M_brain, axis=2)
np.fill_diagonal(Matrix_median, 0)

# Genomic coordinates (hard-coded as in your script)
data = {
    'Locus_N': list(range(1, 26)),
    'Start': [14785864, 14789398, 14792433, 14795381, 14799003, 14802255, 14805412, 14809057, 14812113, 14814823,
              14824792, 14829662, 14833223, 14838104, 14841518, 14844805, 14848436, 14851801, 14853799, 14857068,
              14860470, 14864234, 14865932, 14869591, 14873097],
    'End': [14789298, 14792430, 14795380, 14798629, 14802202, 14805371, 14809056, 14812112, 14814817, 14818656,
            14829604, 14833222, 14835453, 14841356, 14844734, 14848435, 14851800, 14853798, 14856954, 14860469,
            14864233, 14865931, 14869590, 14873096, 14876400]
}

df_loci = pd.DataFrame(data)
df_loci['Center'] = (df_loci['Start'] + df_loci['End']) / 2

# Save the essential source data
np.save(output_dir + "figure1e_median_pwd_matrix.npy", Matrix_median)
df_loci.to_csv(output_dir + "figure1e_genomic_loci.csv", index=False)

print(f"✅ Light source data saved to: {output_dir}")
print(f"   - figure1b_median_pwd_matrix.npy")
print(f"   - figure1b_genomic_loci.csv")

#%%

print("Loading PWD matrix for contact conversion...")

# Load the combined PWD matrix (same as Figure 1b)
Matrix_Hi_M_brain = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_all_split_LEAR_Matrix_PWDscMatrix.npy')  # or use the full concat if needed

# Compute Contact matrix (same parameters as in your code)
Matrix_Hi_M_brain_contact, _ = PWD_to_Contact(Matrix_Hi_M_brain, 1000, 150)

# Load Hi-C matrix
Hi_C_Brain = np.load(INPUT_DATA + 'rut_Hi-C_Mohana_4kb_interpolated.npy')

print(f"Hi-M Contact matrix shape: {Matrix_Hi_M_brain_contact.shape}")
print(f"Hi-C matrix shape: {Hi_C_Brain.shape}")

# Save light source data
np.save(output_dir + "figure1e_HiM_contact_matrix.npy", Matrix_Hi_M_brain_contact)
np.save(output_dir + "figure1e_HiC_contact_matrix.npy", Hi_C_Brain)

print(f"✅ Light source data saved to: {output_dir}")
print("   - figure1e_HiM_contact_matrix.npy")
print("   - figure1e_HiC_contact_matrix.npy")


#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction script for Figure 1f - Log2 Observed vs Expected
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ====================== RELATIVE PATHS ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

INPUT_DATA = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Version_X_Lausanne/Source_data/Figure_1/Source_data/"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ===========================================================

# Import your custom functions
import sys
sys.path.append(str(SCRIPT_DIR))
from Function_Messina_et_al import PWD_to_Contact

print("Loading data for Figure 1f...")

# 1. Load and combine PWD matrix (same as 1b)
m1 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_G_split_LEAR_Matrix_PWDscMatrix.npy')
m2 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_AB_split_LEAR_Matrix_PWDscMatrix.npy')
m3 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_ABp_split_LEAR_Matrix_PWDscMatrix.npy')
m4 = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_all_split_LEAR_Matrix_PWDscMatrix.npy')
m5 = np.load(INPUT_DATA + 'Traces_combined_all_traces_not_KC_split_imputed_LEAR_Matrix_PWDscMatrix.npy')

Matrix_Hi_M_brain = np.concatenate([m1, m2, m3, m4, m5], axis=2)

# 2. Compute Contact matrix
Matrix_Hi_M_brain_contact, _ = PWD_to_Contact(Matrix_Hi_M_brain, 1000, 150)

# 3. Genomic loci (same as before)
data = {
    'Locus_N': list(range(1, 26)),
    'Start': [14785864,14789398,14792433,14795381,14799003,14802255,14805412,14809057,14812113,14814823,
              14824792,14829662,14833223,14838104,14841518,14844805,14848436,14851801,14853799,14857068,
              14860470,14864234,14865932,14869591,14873097],
    'End': [14789298,14792430,14795380,14798629,14802202,14805371,14809056,14812112,14814817,14818656,
            14829604,14833222,14835453,14841356,14844734,14848435,14851800,14853798,14856954,14860469,
            14864233,14865931,14869590,14873096,14876400]
}
df = pd.DataFrame(data)
df['Center'] = (df['Start'] + df['End']) / 2

# Create distance matrix
num_bins = len(df)
distance_matrix = np.zeros((num_bins, num_bins))
for i in range(num_bins):
    for j in range(num_bins):
        distance_matrix[i, j] = abs(df['Center'][i] - df['Center'][j])

genomic_distances = distance_matrix.flatten() / 1000

# 4. Compute power-law fit for contact (needed for Expected)
valid_idx = np.where(genomic_distances > 0)[0]
valid_genomic = genomic_distances[valid_idx]
valid_contact = Matrix_Hi_M_brain_contact.flatten()[valid_idx]

def power_law(x, a, b):
    return a * np.power(x, b)

popt, _ = curve_fit(power_law, valid_genomic, valid_contact, p0=[0.1, 0.3])
a_opt, b_opt = popt

x_values_contact = np.linspace(valid_genomic.min(), valid_genomic.max(), 500)
fitted_contact = power_law(x_values_contact, a_opt, b_opt)

print(f"Power-law parameters: a={a_opt:.4f}, b={b_opt:.4f}")

# Save everything needed for Figure 1f
np.save(OUTPUT_DIR / "figure1f_HiM_contact_matrix.npy", Matrix_Hi_M_brain_contact)
np.save(OUTPUT_DIR / "figure1f_distance_matrix.npy", distance_matrix)
np.save(OUTPUT_DIR / "figure1f_fitted_contact.npy", fitted_contact)
np.save(OUTPUT_DIR / "figure1f_x_values_contact.npy", x_values_contact)

df.to_csv(OUTPUT_DIR / "figure1f_genomic_loci.csv", index=False)

print(f"✅ Light source data saved in {OUTPUT_DIR}")

#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction script for Figure 2c, 2d, 2e - OK107 vs not_OK107
"""

from pathlib import Path
import numpy as np
import sys
from scipy.stats import gaussian_kde

# ====================== PATHS ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
INPUT_DATA = REPO_ROOT / "//home/olivier/Desktop/Marion_project/Paper_figures_marion/Version_X_Lausanne/Source_data/Figure_1_supp/Source_data//"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Building light source for S1B histogram...")

# Load all replicates

# Load and combine
files = [f for f in INPUT_DATA.glob("combined_all_labels*.npy")]
matrices = [np.load(f) for f in files]
Matrix = np.concatenate(matrices, axis=2)

n = Matrix.shape[0]
bins = np.arange(0, 2, 0.05)

# Pre-compute KDE for every (i,j) pair
kde_data = []

for i in range(n):
    row = []
    for j in range(n):
        data = Matrix[i, j, :]
        data = data[np.isfinite(data)]
        if len(data) > 5:  # minimum points for KDE
            try:
                kde = gaussian_kde(data, bw_method='silverman')
                kde_vals = kde(bins)
            except:
                kde_vals = np.zeros_like(bins)
        else:
            kde_vals = np.zeros_like(bins)
        row.append(kde_vals)
    kde_data.append(row)

kde_grid = np.array(kde_data)   # shape (n, n, len(bins))

# N-matrix
binary_matrix = np.where(np.isnan(Matrix), 0, 1)
normalized_matrix = (np.sum(binary_matrix, axis=2) / binary_matrix.shape[2]) * 100
np.fill_diagonal(normalized_matrix, 100)

# Save light data
np.save(OUTPUT_DIR / "S1b_N_matrix.npy", normalized_matrix.astype(np.float32))
np.save(OUTPUT_DIR / "S1b_kde_grid.npy", kde_grid.astype(np.float32))
np.save(OUTPUT_DIR / "S1b_bins.npy", bins.astype(np.float32))

print(f"✅ Ready-to-plot data saved!")
print(f"   N_matrix: {normalized_matrix.shape}")
print(f"   KDE grid: {kde_grid.shape} → much smaller")
#%% 


from pathlib import Path
import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
INPUT_DATA = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Version_X_Lausanne/Source_data/Figure_3/Source_data/"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating single-file light source for Violin Plot...")

# Load loom
loom_path = INPUT_DATA + 'Davie_Janssens_Koldere_et_al_2018_AdultBrain.loom'
with h5py.File(loom_path, 'r') as f:
    matrix_data = f['matrix'][:]
    cell_type = f['col_attrs']['cell_type'][:]
    gene_names = f['row_attrs']['Gene'][:]

cell_type_cleaned = np.array([str(ct, 'utf-8') if isinstance(ct, bytes) else str(ct) 
                              for ct in cell_type])

# Genes
rut_idx = np.where(gene_names == b'rut')[0][0]
act5c_idx = np.where(gene_names == b'Act5C')[0][0]

rut = matrix_data[rut_idx, :]
act5c = matrix_data[act5c_idx, :]

ratio = rut / act5c
ratio = np.where(np.isinf(ratio), np.nan, ratio)

# Masks
kc_G   = np.array(['G-KC' in ct for ct in cell_type_cleaned])
kc_AB  = np.array(['A/B-KC' in ct for ct in cell_type_cleaned])
kc_ABp = np.array(['A/B*-KC' in ct for ct in cell_type_cleaned])
not_kc = ~(kc_G | kc_AB | kc_ABp)

def remove_outliers(data, threshold=1.5):
    if len(data) == 0:
        return np.array([], dtype=np.float32)
    q1 = np.percentile(data, 20)
    q3 = np.percentile(data, 80)
    iqr = q3 - q1
    return data[(data >= q1 - threshold*iqr) & (data <= q3 + threshold*iqr)]

# Cleaned data
data_dict = {
    'not_kc': remove_outliers(ratio[not_kc][~np.isnan(ratio[not_kc])]),
    'kc_G':   remove_outliers(ratio[kc_G][~np.isnan(ratio[kc_G])]),
    'kc_AB':  remove_outliers(ratio[kc_AB][~np.isnan(ratio[kc_AB])]),
    'kc_ABp': remove_outliers(ratio[kc_ABp][~np.isnan(ratio[kc_ABp])])
}

# Save everything in ONE file
np.savez(OUTPUT_DIR / "figure4a_violin_data.npz", **data_dict)

print("✅ All violin data saved in one file:")
print(f"   {OUTPUT_DIR / 'figure4a_violin_data.npz'}")
for name, arr in data_dict.items():
    print(f"   {name:>8}: {len(arr)} values")
    
    
#%% 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light source for Figure 4 - KC subtypes contact maps
"""

from pathlib import Path
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
INPUT_DATA = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Version_X_Lausanne/Source_data/Figure_3/Source_data/"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(SCRIPT_DIR))
from Function_Messina_et_al import PWD_to_Contact

print("Creating light source for KC subtypes contact maps...")

# Load the relevant matrices
Matrix_KCs_G   = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_G_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_KCs_AB  = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_AB_split_LEAR_Matrix_PWDscMatrix.npy')
Matrix_KCs_ABp = np.load(INPUT_DATA + 'Traces_combined_all_traces_KC_ABp_split_LEAR_Matrix_PWDscMatrix.npy')

print(f"G-KC shape:   {Matrix_KCs_G.shape}")
print(f"AB-KC shape:  {Matrix_KCs_AB.shape}")
print(f"ABp-KC shape: {Matrix_KCs_ABp.shape}")

# Compute contact matrices
Matrix_KCs_G_contact,   _ = PWD_to_Contact(Matrix_KCs_G,   1000, 150)
Matrix_KCs_AB_contact,  _ = PWD_to_Contact(Matrix_KCs_AB,  1000, 150)
Matrix_KCs_ABp_contact, _ = PWD_to_Contact(Matrix_KCs_ABp, 1000, 150)

# Save in one file (clean)
np.savez(OUTPUT_DIR / "figure4_kc_contact_maps.npz",
         G_contact=Matrix_KCs_G_contact.astype(np.float32),
         AB_contact=Matrix_KCs_AB_contact.astype(np.float32),
         ABp_contact=Matrix_KCs_ABp_contact.astype(np.float32))

print(f"✅ Light source saved: figure4_kc_contact_maps.npz")


#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light source extraction for Figure 4F
"""

from pathlib import Path
import numpy as np
import json

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
INPUT_FOLDER = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Test_Bootstraping_RNA_seq/Export_Data_RNA/"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating light source for Figure 4F...")

# Load all data
all_xgrid = np.load(INPUT_FOLDER + 'all_xgrid.npy')
all_yhat = np.load(INPUT_FOLDER + 'all_yhat.npy')
all_ci = np.load(INPUT_FOLDER + 'all_ci.npy')

kc_xgrid = np.load(INPUT_FOLDER + 'kc_xgrid.npy')
kc_yhat = np.load(INPUT_FOLDER + 'kc_yhat.npy')
kc_ci = np.load(INPUT_FOLDER + 'kc_ci.npy')

with open(INPUT_FOLDER + 'plot_metadata.json', 'r') as f:
    metadata = json.load(f)

# Save everything in ONE clean file
np.savez(OUTPUT_DIR / "figure4f_correlation_data.npz",
         all_xgrid=all_xgrid,
         all_yhat=all_yhat,
         all_ci=all_ci,
         kc_xgrid=kc_xgrid,
         kc_yhat=kc_yhat,
         kc_ci=kc_ci)

# Save metadata separately (small json)
with open(OUTPUT_DIR / "figure4f_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Light source created:")
print(f"   {OUTPUT_DIR / 'figure4f_correlation_data.npz'}")
print(f"   {OUTPUT_DIR / 'figure4f_metadata.json'}")

#%% 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create light source for Figure 4a replot script
"""

from pathlib import Path
import numpy as np
import json

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXPORT_FOLDER = "/home/olivier/Desktop/Marion_project/Paper_figures_marion/Test_Bootstraping_RNA_seq/Export_Data_ATAC/"
OUTPUT_DIR = REPO_ROOT / "Source_light"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating light source from exported data...")

# Load all files from your export folder
all_xgrid = np.load(EXPORT_FOLDER + 'all_xgrid.npy')
all_yhat = np.load(EXPORT_FOLDER + 'all_yhat.npy')
all_ci = np.load(EXPORT_FOLDER + 'all_ci.npy')

kc_xgrid = np.load(EXPORT_FOLDER + 'kc_xgrid.npy')
kc_yhat = np.load(EXPORT_FOLDER + 'kc_yhat.npy')
kc_ci = np.load(EXPORT_FOLDER +'kc_ci.npy')

with open(EXPORT_FOLDER + 'plot_metadata.json', 'r') as f:
    metadata = json.load(f)

# Save everything in one clean .npz file
np.savez(OUTPUT_DIR / "figure4a_light_data.npz",
         all_xgrid=all_xgrid,
         all_yhat=all_yhat,
         all_ci=all_ci,
         kc_xgrid=kc_xgrid,
         kc_yhat=kc_yhat,
         kc_ci=kc_ci)

# Also save metadata
with open(OUTPUT_DIR / "figure4a_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Light source created in {OUTPUT_DIR}")
print("Files:")
print("   - figure4a_light_data.npz")
print("   - figure4a_metadata.json")