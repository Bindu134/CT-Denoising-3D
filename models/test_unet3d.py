import torch
from unet3d import UNet3D

# Dummy 3D input [B, C, D, H, W]
x = torch.randn(1, 1, 64, 64, 64)

model = UNet3D()
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
