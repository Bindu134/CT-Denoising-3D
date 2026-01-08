import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Path to CT file
ct_path = "data/raw/sample_ct.nii.gz"

# Load CT volume
ct_img = nib.load(ct_path)
ct_data = ct_img.get_fdata()

print("CT volume shape:", ct_data.shape)
print("Intensity range:", ct_data.min(), ct_data.max())

# Show middle slice
slice_idx = ct_data.shape[2] // 2
plt.imshow(ct_data[:, :, slice_idx], cmap="gray")
plt.title("Middle CT Slice")
plt.axis("off")
plt.show()
