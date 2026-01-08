from ct_dataset import CTDenoisingDataset

dataset = CTDenoisingDataset("data/raw/sample_ct.nii.gz")

noisy, clean = dataset[0]

print("Noisy shape:", noisy.shape)
print("Clean shape:", clean.shape)
print("Value range:", noisy.min().item(), noisy.max().item())
