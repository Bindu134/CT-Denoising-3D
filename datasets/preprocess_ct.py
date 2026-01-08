import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load CT volume
ct_path = "data/raw/sample_ct.nii.gz"
ct_img = nib.load(ct_path)
ct = ct_img.get_fdata()

# -------- Normalize CT --------
# Clip extreme values (HU-like range)
ct = np.clip(ct, -1000, 1000)

# Normalize to [0, 1]
ct_norm = (ct - ct.min()) / (ct.max() - ct.min())

print("Normalized range:", ct_norm.min(), ct_norm.max())

# -------- Add Noise (simulate low-dose CT) --------
noise_std = 0.05
noise = np.random.normal(0, noise_std, ct_norm.shape)
ct_noisy = ct_norm + noise
ct_noisy = np.clip(ct_noisy, 0, 1)

# -------- Visualization --------
slice_idx = ct_norm.shape[2] // 2

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(ct_norm[:, :, slice_idx], cmap="gray")
plt.title("Clean CT")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ct_noisy[:, :, slice_idx], cmap="gray")
plt.title("Noisy CT")
plt.axis("off")

plt.tight_layout()
plt.show()
