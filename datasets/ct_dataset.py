import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class CTDenoisingDataset(Dataset):
    def __init__(self, ct_path, patch_size=32, noise_std=0.05):
        self.patch_size = patch_size
        self.noise_std = noise_std

        ct_img = nib.load(ct_path)
        ct = ct_img.get_fdata()

        # Normalize
        ct = np.clip(ct, -1000, 1000)
        ct = (ct - ct.min()) / (ct.max() - ct.min())
        self.ct = ct.astype(np.float32)

        # Precompute patch indices
        self.indices = []
        D, H, W = self.ct.shape
        for z in range(0, D - patch_size, patch_size):
            for y in range(0, H - patch_size, patch_size):
                for x in range(0, W - patch_size, patch_size):
                    self.indices.append((z, y, x))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        z, y, x = self.indices[idx]

        clean = self.ct[
            z:z+self.patch_size,
            y:y+self.patch_size,
            x:x+self.patch_size
        ]

        noise = np.random.normal(0, self.noise_std, clean.shape)
        noisy = (clean + noise).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        noisy = torch.from_numpy(noisy).float().unsqueeze(0)
        clean = torch.from_numpy(clean).float().unsqueeze(0)

        return noisy, clean
