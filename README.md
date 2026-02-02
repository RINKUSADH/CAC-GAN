# CAC-GAN: High-Fidelity Synthesis of MRI Sequences using Contrast-Aware CycleGAN

Official PyTorch implementation of the paper:  
**"CAC-GAN: HIGH-FIDELITY SYNTHESIS OF MRI SEQUENCES USING CONTRAST-AWARE CYCLEGAN"**  
*Accepted at ISBI 2026.*

[![Paper](https://img.shields.io/badge/Paper-ISBI2026-blue)](https://github.com/RINKUSADH/CAC-GAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Abstract
Artificial synthesis of MRI sequences can reduce scan time, cost, and patient discomfort. However, existing GAN-based approaches struggle with structural inconsistencies and sequence-specific variations across different field strengths. 

**CAC-GAN** (Contrast-aware CycleGAN) addresses these challenges through three key innovations:
1.  **Single Cycle Consistency Loss (SCCL):** Enforces cycle constraints only in the target contrast direction to preserve contrast variations and prevent over-regularization.
2.  **Feature Matching Loss ($L_{FM}$):** Aligns intermediate discriminator activations to enhance anatomical accuracy and texture realism.
3.  **Contrast Loss ($L_c$):** Matches pixel intensity distributions (histograms) between real and synthetic images to stabilize global contrast.

Our model is computationally lightweight and achieves up to a **10% improvement** in SSIM and PSNR over state-of-the-art methods.

---

## 🏗️ Architecture
The framework consists of two generators ($G_{T1 \to T2}$, $G_{T2 \to T1}$) using an encoder-decoder structure with residual blocks, and two domain-specific discriminators ($D_{T1}$, $D_{T2}$) using a PatchGAN architecture.

<img width="987" height="451" alt="image" src="https://github.com/user-attachments/assets/a4093cb9-d885-45c5-b3c5-44b2fc3f698b" />

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RINKUSADH/CAC-GAN.git
cd CAC-GAN
```
### 2. Set Up Environment
We recommend using a Conda environment:

```bash
conda create -n cacgan python=3.10
conda activate cacgan
pip install -r requirements.txt
```
## 📂 Dataset Preparation
The code expects the 2D PMC Dataset structure. Ensure your images are named with suffixes _T1 and _T2 (e.g., subject01_T1.png, subject01_T2.png) so the matching logic in datasets.py can pair them.
Folder Structure:
```bash
/DATA/PMC_dataset/2D/
├── train/
│   └── 3T/ (or 1.5T)
│       ├── T1/ (contains *_T1.png)
│       └── T2/ (contains *_T2.png)
├── valid/
│   └── 3T/
│       ├── T1/
│       └── T2/
└── test/
    └── 3T/
        ├── T1/
        └── T2/
```
## 🚀 Usage
### 1. Training
The main.py script uses the best hyperparameters determined via Optuna. To train the model for 3T field strength:

```bash
python main.py \
    --field_strength 3T \
    --input_modality t1 \
    --target_modality t2 \
    --n_epochs 200 \
    --output_dir ./output
```
### 2. Inference & Evaluation
To generate synthetic images and calculate quantitative metrics (PSNR, SSIM, LPIPS) on the test set:

```bash
python test.py \
    --model_path ./output/best_trial_YYYY-MM-DD/best_model_ssim.pth \
    --field_strength 3T \
    --data_dir /DATA/PMC_dataset/2D/test \
    --output_dir ./inference_output
```
## 📊 Experimental Results
Quantitative Comparison (3T Field Strength)
```bash
Method	          SSIM	       PSNR (dB)
Pix2Pix	          0.77 ± 0.12	 25.05 ± 1.64
CycleGAN	        0.63 ± 0.05	 22.27 ± 0.80
TSIT	            0.81 ± 0.12	 25.83 ± 1.50
BBDM	            0.82 ± 0.12	 26.87 ± 1.87
Proposed CAC-GAN	0.91 ± 0.01	 29.66 ± 0.46
```
Qualitative Samples
Our method preserves sharp structural details and contrast profiles highly faithful to the target domain compared to blurred or artifact-heavy baseline results.
<img width="1358" height="755" alt="image" src="https://github.com/user-attachments/assets/4b1a3144-9bad-45bc-9627-6a8cb485a83f" />


## 📁 Repository Structure
```bash
generator.py / discriminator.py: Network architectures.
latent_space.py: Implementation of residual blocks.
losses.py: Implementation of the novel Contrast loss.
train.py / main.py: Core training logic and hyperparameter configuration.
metrics.py: Calculation of PSNR and SSIM.
test.py: Inference script with LPIPS evaluation.
visualization.py: Utilities for plotting training curves and saving image samples.
```
## 📜 Citation
If you use this code or our paper in your research, please cite:
```bash
Bibtex
@inproceedings{sadh2026cacgan,
  title={CAC-GAN: High-Fidelity Synthesis of MRI Sequences Using Contrast-Aware CycleGAN},
  author={Rinku Sadh and Prabhat Ranjan and Angshuman Paul},
  booktitle={Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```
