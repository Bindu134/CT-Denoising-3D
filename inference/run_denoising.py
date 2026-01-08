import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import nibabel as nib
import numpy as np
from models.unet3d import UNet3D
import os

# ------------------
# CONFIG
# ------------------
MODEL_PATH = "unet3d_denoising.pth"
INPUT_CT = "data/raw/test_ct.nii.gz"
OUTPUT_CT = "data/raw/denoised_ct.nii.gz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# LOAD MODEL
# ------------------
model = UNet3D(in_channels=1, out_channels=1, base_filters=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded")

# ------------------
# LOAD CT
# ------------------
ct_nifti = nib.load(INPUT_CT)
ct_data = ct_nifti.get_fdata().astype(np.float32)

# Normalize (same as training)
ct_min, ct_max = ct_data.min(), ct_data.max()
ct_data = (ct_data - ct_min) / (ct_max - ct_min + 1e-8)

print("CT loaded:", ct_data.shape)

# ------------------
# PREPARE TENSOR
# Shape: [1, 1, D, H, W]
# ------------------
ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0)
ct_tensor = ct_tensor.to(DEVICE)

# ------------------
# INFERENCE
# ------------------
with torch.no_grad():
    denoised = model(ct_tensor)

denoised = denoised.squeeze().cpu().numpy()

# De-normalize
denoised = denoised * (ct_max - ct_min) + ct_min

# ------------------
# SAVE OUTPUT
# ------------------
denoised_nifti = nib.Nifti1Image(denoised, ct_nifti.affine)
nib.save(denoised_nifti, OUTPUT_CT)

print("Denoised CT saved to:", OUTPUT_CT)
