import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from datasets.ct_dataset import CTDenoisingDataset
from models.unet3d import UNet3D

# ------------------ CONFIG ------------------
CT_PATH = "data/raw/sample_ct.nii.gz"
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ DATA ------------------
dataset = CTDenoisingDataset(CT_PATH)
print("Number of patches:", len(dataset))
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------ MODEL ------------------
model = UNet3D().to(DEVICE)
model = model.float()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print("Testing tiny forward...")
with torch.no_grad():
    x = torch.randn(1, 1, 16, 16, 16)
    model(x)
print("Tiny forward passed")

# ------------------ TRAIN LOOP ------------------
print("Starting training loop...")

for epoch in range(EPOCHS):
    print(f"Epoch loop entered: {epoch+1}")
    model.train()
    epoch_loss = 0

    for noisy, clean in loader:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.6f}")


# ------------------ SAVE MODEL ------------------
torch.save(model.state_dict(), "unet3d_denoising.pth")
print("Model saved as unet3d_denoising.pth")
