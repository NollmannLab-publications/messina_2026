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