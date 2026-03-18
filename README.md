# Improving Diffusion-Based Virtual Try-On for Real-World Applications

> ECE285 Final Project — University of California, San Diego
> **Tiancheng Chen** (tic041@ucsd.edu) · **Haifan Zhao** (haz152@ucsd.edu)

---

## Overview

This project proposes an **unpaired diffusion-based virtual try-on framework** that allows users to visualize how a garment looks when worn, without requiring physical fittings or paired training data. Given a person image and an independently collected garment image, the system synthesizes a photorealistic try-on result that:

- Preserves the person's body structure and pose
- Accurately transfers fine-grained garment details (logos, textures, embroidery)
- Produces realistic clothing dynamics (natural wrinkles and shading)

Key innovations over prior work:

- **Dual-path garment encoder** — combines CLIP (global semantics) with a fine-grained vision encoder (local texture), fused via a learnable weighted sum
- **Unpaired training strategy** — geometric and appearance augmentation + attention total variation loss (L_ATV), enabling semantic correspondence without paired supervision
- **Mask-guided generation** — concatenates a clothing segmentation mask to the U-Net input to restrict synthesis to the garment region

---

## Architecture

```
Garment Image ──► CLIP Image Encoder ──────────────────► Feature Fusion (λ) ──►
               └─► Fine-grained CNN (ResNet) ──────────►                        │
                                                                                 ▼
Person Image ──► Pose Estimator ──► Pose Map ──────────────────────► U-Net Diffusion Model ──► Final Try-On Image
             ├─► Segmentation & Inpainting ──► Agnostic Map ──────►     (Cross-Attention)
             └─► Mask Generation ──► Segmentation Mask ───────────►
```

The U-Net receives a concatenated **13-channel** input:

| Channel(s) | Source |
|---|---|
| 4 | Noisy latent z_t |
| 4 | VAE-encoded agnostic map |
| 1 | Binary clothing mask |
| 4 | VAE-encoded DensePose map |

Zero cross-attention blocks in the U-Net decoder learn implicit garment-body alignment in latent space — no explicit warping network required.

---

## Results

### Quantitative Comparison (VITON-HD)

| Method | LPIPS ↓ | SSIM ↑ | FID ↓ |
|---|---|---|---|
| Pre-VTON (Baseline) | 0.153 | 0.826 | 9.98 |
| **Ours (Proposed)** | **0.110** | **0.923** | **6.89** |

### Ablation Study (Paired Setting)

| Configuration | LPIPS ↓ | SSIM ↑ | FID ↓ |
|---|---|---|---|
| Baseline (CLIP only) | 0.142 | 0.854 | 8.75 |
| + Dual-Path Encoder | 0.125 | 0.891 | 7.42 |
| + L_ATV (Full Model) | **0.110** | **0.923** | **6.89** |

---

## Dataset

We use the **[VITON-HD](https://github.com/shadow2496/VITON-HD)** benchmark (11,647 training pairs · 2,032 test pairs · 1024×768 resolution).

The dataset provides: clothing-agnostic maps, binary agnostic masks, DensePose surface maps, human parsing results, and garment masks — all used directly as conditioning inputs.

**Download our preprocessed dataset here:**
[Google Drive](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view)

After downloading, extract it to the `data/` directory:

```bash
unzip viton_hd_dataset.zip -d data/
```

Expected structure:

```
data/
├── train/
│   ├── image/
│   ├── cloth/
│   ├── agnostic-v3.2/
│   ├── agnostic-mask/
│   ├── densepose/
│   └── parse/
└── test/
    ├── image/
    ├── cloth/
    ├── agnostic-v3.2/
    ├── agnostic-mask/
    ├── densepose/
    └── parse/
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (tested on NVIDIA A100)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/Tiancheng0125/ece285-viton.git
cd ece285-viton

# Create a virtual environment
conda create -n viton python=3.9
conda activate viton

# Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
python train.py \
  --data_dir data/train \
  --output_dir checkpoints/ \
  --batch_size 4 \
  --lr 1e-5 \
  --max_steps 30000 \
  --lambda_atv 1.0 \
  --lambda_fine 0.0
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--lambda_atv` | `1.0` | Weight for attention total variation loss |
| `--lambda_fine` | `0.0` | Weight λ for fine-grained encoder (dual-path fusion) |
| `--batch_size` | `4` | Training batch size |
| `--max_steps` | `30000` | Total training steps |

---

## Inference

```bash
python inference.py \
  --checkpoint checkpoints/final.ckpt \
  --person_image path/to/person.jpg \
  --garment_image path/to/garment.jpg \
  --output_dir results/
```

---

## Evaluation

```bash
# Paired setting (SSIM, LPIPS)
python evaluate.py --mode paired --data_dir data/test --pred_dir results/

# Unpaired / No-GT setting (CLIP-I, FID, KID)
python evaluate.py --mode unpaired --data_dir data/test --pred_dir results/
```

---

## Method Details

### Dual-Path Garment Encoding

```
f_coarse  = CLIP_img(x_g)                          # global semantics
f_fine    = FineEncoder({p_1, ..., p_K, p_thumb})  # local texture details
c_garment = f_coarse + λ · f_fine                  # fused representation
```

### Zero Cross-Attention (Implicit Alignment)

```
O_h = SelfAttention(O_s) + λ · CrossAttention(O_s, c_garment)
```

### Training Objective

```
L_total = L_LDM + λ_ATV · L_ATV
```

where L_ATV penalises spatially dispersed attention activations to suppress colour bleeding at garment boundaries.

---


## Acknowledgements

This project builds on [StableVITON](https://github.com/rlawjdghek/StableVITON), [DH-VTON](https://github.com/jiaweiwei2/DH-VTON), and the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset. We thank the authors of Stable Diffusion, DINOv2, and DensePose for their open-source contributions.

---

## Citation

```bibtex
@misc{chen2026viton,
  title   = {Improving Diffusion-Based Virtual Try-On for Real-World Applications},
  author  = {Tiancheng Chen and Haifan Zhao},
  year    = {2026},
  school  = {University of California, San Diego},
  note    = {ECE285 Final Project}
}
```
