import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

noisy = nib.load("data/raw/test_ct.nii.gz").get_fdata()
denoised = nib.load("data/raw/denoised_ct.nii.gz").get_fdata()

z = noisy.shape[2] // 2

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Noisy CT")
plt.imshow(noisy[:, :, z], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Denoised CT")
plt.imshow(denoised[:, :, z], cmap="gray")
plt.axis("off")

plt.show()
