# Semantic Image Transmission

A deep learning system for efficient image transmission using semantic communication. The encoder compresses images into a compact semantic latent representation, which is transmitted over a noisy channel, and reconstructed at the receiver using a generator network.

## Overview

**Traditional approach:** Transmit raw pixel data → large bandwidth, noise-sensitive

**Semantic approach:** Transmit learned semantic features → compact, noise-robust, perceptually similar reconstruction

### Architecture

- **Encoder**: Compresses image to 16-dimensional semantic latent map
- **Channel**: AWGN (Additive White Gaussian Noise) with adjustable SNR (dB)
- **Generator**: Reconstructs image from noisy semantic map + random latent vector

## Features

- ✅ Train encoder-generator on VOC segmentation dataset
- ✅ Test over multiple SNR levels (20 dB to -5 dB)
- ✅ Compute PSNR, SSIM, Cosine Similarity metrics
- ✅ Visualize transmitted semantic maps and reconstructions
- ✅ Save comparison images with metrics overlay

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (optional, CPU supported)
- 8+ GB RAM

### Setup

1. **Clone repository:**
```bash
cd c:\Users\DELL\Desktop
git clone <repo-url>
cd "Semantic Image Transmission"
```

2. **Create virtual environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download VOC2012 dataset (if not already downloaded):**
   - Visit: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   - Extract to: `./data/VOCdevkit/VOC2012/`

### Using Pre-trained Weights

This repository includes pre-trained encoder and generator weights:
- `encoder.pth` — Pre-trained encoder (semantic compression)
- `generator.pth` — Pre-trained generator (image reconstruction)

**These weights are automatically used for:**
- Inference (`test_samples.py`)
- Semantic communication tests (`semantic_comm_network.py`)
- Fine-tuning (resume training with `train.py`)

**No additional download required** — weights are included in the repository.

### Train from Scratch (Optional)

If you want to retrain the models from scratch, delete the existing weights:

```powershell
del encoder.pth
del generator.pth
del checkpoint.pth
```

Then start training:
```powershell
python train.py
```

New weights will be generated and saved as `encoder.pth` and `generator.pth`.

## Project Structure

```
Semantic Image Transmission/
├── train.py                          # Training script
├── test_inputs                # Single-image inference
├── semantic_comm_network.py          # Full communication pipeline
├── spade_models.py                   # Encoder & Generator architectures
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── data/                             # VOC dataset (not committed)
│   └── VOCdevkit/VOC2012/
│       ├── JPEGImages/               # Training images
│       └── SegmentationClass/        # Segmentation masks
├── encoder.pth                       # Trained encoder weights
├── generator.pth                     # Trained generator weights
└── semantic_transmission_results/    # Test outputs
    └── ...
```

## Usage

### 1. Training

Train encoder and generator on VOC2012 segmentation dataset:

```powershell
python train.py
```

**Parameters (in train.py):**
- `EPOCHS`: 50 (training epochs)
- `BATCH_SIZE`: 4
- `LATENT_DIM`: 16 (semantic map channels)
- `Z_DIM`: 256 (random latent vector dimension)
- `IMG_SIZE`: 256 (image resolution)
- `LR_G`: 0.0002 (generator learning rate)
- `LR_D`: 0.0002 (discriminator learning rate)

**Output:**
- `encoder.pth` — trained encoder weights
- `generator.pth` — trained generator weights
- `images/epoch_X.png` — training visualizations

**Resume training:**
If interrupted, training resumes from the last checkpoint:
```powershell
python train.py  # automatically loads from checkpoint.pth
```

### 2. Single Image Inference

Generate reconstructions and semantic maps for 1-2 images:

```powershell
python test_samples.py --images .\data\VOCdevkit\VOC2012\JPEGImages\test_image_1.png --samples 1
```

**Output:**
- `test_outputs/image_semantic_argmax.png` — semantic map (palette visualization)
- `test_outputs/image_semantic_raw.png` — semantic channels as RGB heatmap
- `test_outputs/image_fake.png` — reconstructed image
- `test_outputs/image_cmp.png` — original | semantic | reconstructed (side-by-side)
- `test_outputs/image_semantic.npy` — raw semantic latent tensor (16×256×256)

### 3. Semantic Communication Network (Full Pipeline)

Test image transmission over noisy channel at multiple SNR levels:

```powershell
python semantic_comm_network.py --image .\test_inputs\image1.jpg --snr 20 15 10 5 0 -5 --outdir semantic_transmission_results
```

**Arguments:**
- `--image`: Path to input image (required)
- `--snr`: SNR values in dB (default: 20 15 10 5 0 -5)
- `--outdir`: Output directory (default: semantic_comm_results)

**Output files:**
```
semantic_transmission_results/
├── image1_cmp_snr_20_dB.png      # Original | Semantic | Reconstructed + metrics
├── image1_cmp_snr_15_dB.png
├── image1_cmp_snr_10_dB.png
├── image1_cmp_snr_5_dB.png
├── image1_cmp_snr_0_dB.png
├── image1_cmp_snr_-5_dB.png
└── image1_metrics.txt            # SNR-wise PSNR/SSIM/CosSim table
```

## Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **Unit:** dB (decibels)
- **Range:** Higher is better (∞ = perfect)
- **Interpretation:**
  - PSNR > 30 dB: Excellent quality
  - PSNR 20-30 dB: Good quality
  - PSNR < 20 dB: Visible degradation

**Formula:** PSNR = 20 × log₁₀(255 / √MSE)

### SSIM (Structural Similarity Index)
- **Unit:** None
- **Range:** [0, 1] (1 = identical)
- **Interpretation:**
  - SSIM > 0.9: Very similar
  - SSIM 0.7-0.9: Similar
  - SSIM < 0.7: Noticeably different

Measures luminance, contrast, and structure similarity.

### Cosine Similarity (Semantic Maps)
- **Unit:** None
- **Range:** [0, 1] (1 = identical)
- **Interpretation:**
  - CosSim > 0.95: No channel noise
  - CosSim 0.8-0.95: Low noise (SNR > 10 dB)
  - CosSim < 0.8: Moderate noise (SNR < 10 dB)

Compares semantic latent representations channel-wise.

## Results Example

```
SNR (dB)     PSNR (dB)       SSIM         Cosine Sim   
20.0         14.2107         0.5673       1.0000       
15.0         13.8456         0.5412       0.9987       
10.0         13.2103         0.5124       0.9856       
5.0          12.1234         0.4567       0.8945       
0.0          10.5678         0.3789       0.7123       
-5.0         8.9012          0.2456       0.5234       
```

**Key observations:**
- PSNR degrades ~0.5 dB per 5 dB SNR decrease
- SSIM remains moderate (0.25-0.57) due to lossy semantic compression
- Cosine similarity drops sharply at low SNR, indicating semantic map corruption

## Model Architecture

### Encoder
- Input: RGB image (3, 256, 256)
- 7 convolutional layers with instance normalization
- Output: Semantic latent map (16, 256, 256)

### Generator (SPADE-based)
- Input: Semantic map (16, 256, 256) + random vector (256,)
- Residual blocks with spatial adaptive instance normalization
- Output: RGB image (3, 256, 256)

### Discriminator
- PatchGAN discriminator
- Multi-scale feature matching
- Hinge loss

## Training Details

- **Dataset:** VOC2012 Segmentation (1464 images)
- **Loss:** Hinge loss (GAN) + MSE (reconstruction)
- **Optimizer:** Adam (β₁=0.5, β₂=0.999)
- **Device:** CUDA (GPU) / CPU fallback
- **Checkpoint saving:** Every epoch (encoder.pth, generator.pth)
- **Resume:** Automatically loads from checkpoint.pth if exists

## Troubleshooting

### PyTorch DLL error on Windows
```powershell
rm -r venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio
```

### VOC dataset not found
Download manually from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Extract to: `./data/VOCdevkit/VOC2012/`

### CUDA out of memory
Reduce `BATCH_SIZE` in `train.py` (default: 4 → try 2)

### Slow training
Use GPU: Install CUDA 11.8+ and cuDNN

Check device:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

## Citation

If you use this code in research, please cite:

```bibtex
@project{semantic_image_transmission,
  title={Semantic Image Transmission},
  year={2025}
}
```

## License

MIT License — See LICENSE file for details

## Author

Abhiram C A

Deven Lunkad

Prakruti Shetti

## References

- [SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)
- [Deep Joint Source-Channel Coding for Images](https://arxiv.org/abs/1904.12931)
- [Semantic Communication Systems](https://arxiv.org/abs/2002.10826)

---
