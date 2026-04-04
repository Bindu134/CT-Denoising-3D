# CT-Denoising-3D

**3D U-Net for Low-Dose CT Volume Denoising**

Low-dose CT protocols reduce patient radiation exposure but introduce noise that degrades diagnostic quality. This project trains a lightweight 3D U-Net to map noisy NIfTI volumes to denoised outputs, preserving anatomical structures without manual filtering heuristics.

---

## Architecture

Encoder–bottleneck–decoder structure built on 3D convolutions, batch normalization, transpose convolutions, and skip connections. Single-channel input and output (volumetric CT, one modality).

| Component | Detail |
|---|---|
| Model | 3D U-Net |
| Input | Noisy CT volume — 1 channel |
| Output | Denoised CT volume — 1 channel |
| Loss | Mean Squared Error (MSE) |
| Optimizer | Adam |

The encoder downsamples with 3D conv blocks; the decoder reconstructs spatial resolution via transpose convolutions. Skip connections carry encoder feature maps directly to the corresponding decoder stage to retain fine structural detail.

---

## Design Decisions

**Patch-based training** was chosen over spatial downsampling. Reducing volume resolution would discard the fine anatomical detail the model is trained to recover — patching preserves resolution while fitting GPU memory constraints on single-GPU hardware.

**NIfTI-native pipeline** — volumes are read, processed, and written back in `.nii.gz` format throughout, avoiding lossy intermediate conversions.

---

## Setup

```bash
conda create -n ct3d python=3.9
conda activate ct3d
pip install torch torchvision nibabel numpy matplotlib scikit-image tqdm
```

---

## Dataset

Place CT volumes in `data/raw/` in `.nii.gz` format:

```
data/raw/sample_ct.nii.gz
data/raw/test_ct.nii.gz
```

During training, volumes are split into 3D patches automatically.

---

## Training

```bash
python training/train_unet3d.py
```

Training log example:

```
Epoch [1/5] - Loss: 42.13
Epoch [5/5] - Loss: 9.01
Model saved: unet3d_denoising.pth
```

---

## Inference

```bash
python inference/run_denoising.py
```

Output written to `data/raw/denoised_ct.nii.gz`.

---

## Evaluation

Performance is measured using PSNR and SSIM against the noisy input baseline.

```
PSNR : 0.03 dB
SSIM : 0.0003
```

> These numbers reflect a demo-scale dataset. The evaluation framework is in place; metrics improve with multi-patient training data and extended epochs.

Slice-wise visual comparisons (noisy vs. denoised) are saved to `results/comparison_slice.png`.

---

## Limitations

Current metrics reflect a single-sample demo run, not a clinical benchmark. The model has not been evaluated against ground-truth clean CT volumes — a requirement for any downstream diagnostic use. DICOM input is not yet supported; volumes must be converted to NIfTI before ingestion.

---

## Project Structure

```
CT-Denoising-3D/
├── data/
│   └── raw/                  # Input and output CT volumes
├── training/
│   └── train_unet3d.py       # Training script
├── inference/
│   └── run_denoising.py      # Inference script
├── results/                  # Slice-wise visualizations
├── requirements.txt
└── README.md
```

---

## Author

**Bindu S Reddy**
M.Tech — Artificial Intelligence & Data Science
Research interests: Medical imaging, deep learning, applied AI

---

## License

Academic and research use only.
