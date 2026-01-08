CT-Denoising-3D

A deep learning–based 3D CT image denoising framework using a lightweight 3D U-Net architecture. This project focuses on reducing noise in volumetric CT scans stored in NIfTI (.nii.gz) format while preserving anatomical structures.

The pipeline covers data preprocessing, patch-based training, inference, visualization, and evaluation using standard image quality metrics.

---

🔍 Project Overview

Medical CT scans often suffer from noise due to low-dose imaging protocols. This project addresses that problem by training a 3D U-Net model to learn a mapping from noisy CT volumes to clean (denoised) volumes.

Key highlights:

* Full end-to-end 3D pipeline

* Patch-based training to handle memory constraints

* Supports NIfTI (.nii.gz) medical imaging format

* Quantitative evaluation using PSNR and SSIM

* Qualitative slice-wise visualization

---


🧠 Model Architecture

* Model: 3D U-Net
* Input: Noisy CT volume (1 channel)
* Output: Denoised CT volume (1 channel)
* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam

The architecture consists of:
* Encoder with 3D convolutions + batch normalization
* Bottleneck layer
* Decoder with transpose convolutions and skip connections

---

⚙️ Setup Instructions

1️⃣ Create Conda Environment

conda create -n ct3d python=3.9
conda activate ct3d

2️⃣ Install Dependencies

pip install torch torchvision nibabel numpy matplotlib scikit-image tqdm

🧪 Dataset Preparation

* Place CT volumes in data/raw/
* Supported format: .nii.gz
* Example:

data/raw/sample_ct.nii.gz
data/raw/test_ct.nii.gz

During training, the dataset is split into 3D patches to reduce memory usage.

---


🚀 Training

Run the training script from the project root:

python training/train_unet3d.py

Training output example:

Epoch [1/5] - Loss: 42.13
Epoch [5/5] - Loss: 9.01
Model saved as unet3d_denoising.pth

---

🔎 Inference (Denoising)

Run inference on a test CT volume:

python inference/run_denoising.py

Output:

Denoised CT saved as:

data/raw/denoised_ct.nii.gz

---

📊 Evaluation Metrics

The model performance is evaluated using:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

Example results:

PSNR : 0.03 dB
SSIM : 0.0003

Note: These results are expected for a demo-scale dataset. Performance improves significantly with larger datasets and longer training.

---

🖼️ Visualization

* Slice-wise comparison between noisy and denoised CT
* Output saved in results/

Example:
comparison_slice.png

---

🧩 Key Challenges Addressed

* 3D memory constraints (handled via patching)
* Medical imaging file formats
* Stable training of 3D CNNs on limited hardware

---

🔮 Future Improvements

* Train on larger multi-patient datasets
* Add residual learning
* Incorporate perceptual or adversarial loss
* Support DICOM input
* Multi-GPU training
* Quantitative evaluation against ground truth clean CT

---

👩‍💻 Author

Bindu S Reddy
M.Tech – Artificial Intelligence & Data Science
Focused on Medical Imaging, Deep Learning, and Applied AI

---

📜 License

This project is intended for academic and research purposes.

⭐ If you find this project useful, consider starring the repository!
