import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

NOISY_PATH = "data/raw/test_ct.nii.gz"
DENOISED_PATH = "data/raw/denoised_ct.nii.gz"
OUT_DIR = "results"

os.makedirs(OUT_DIR, exist_ok=True)

# Load volumes
noisy = nib.load(NOISY_PATH).get_fdata()
denoised = nib.load(DENOISED_PATH).get_fdata()

# Choose middle slice
z = noisy.shape[2] // 2

noisy_slice = noisy[:, :, z]
denoised_slice = denoised[:, :, z]
diff_slice = np.abs(noisy_slice - denoised_slice)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(noisy_slice, cmap="gray")
plt.title("Noisy CT")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(denoised_slice, cmap="gray")
plt.title("Denoised CT")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(diff_slice, cmap="hot")
plt.title("Difference")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_slice.png", dpi=200)
plt.show()

print("Saved comparison image to results/comparison_slice.png")
