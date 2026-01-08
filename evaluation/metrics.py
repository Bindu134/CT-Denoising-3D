import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Load volumes
clean = nib.load("data/raw/sample_ct.nii.gz").get_fdata()
denoised = nib.load("data/raw/denoised_ct.nii.gz").get_fdata()

# Normalize (important for fair metrics)
clean = (clean - clean.min()) / (clean.max() - clean.min())
denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())

# ---- PSNR (volume-level) ----
psnr_value = peak_signal_noise_ratio(clean, denoised, data_range=1.0)

# ---- SSIM (slice-wise average, standard practice) ----
ssim_values = []
for z in range(clean.shape[2]):
    ssim = structural_similarity(
        clean[:, :, z],
        denoised[:, :, z],
        data_range=1.0
    )
    ssim_values.append(ssim)

mean_ssim = np.mean(ssim_values)

print(f"PSNR : {psnr_value:.2f} dB")
print(f"SSIM : {mean_ssim:.4f}")
