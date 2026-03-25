# Improving Diffusion-Based Virtual Try-On for Real-World Applications

> ECE285 Final Project — University of California, San Diego
> **Tiancheng Chen** (tic041@ucsd.edu) · **Haifan Zhao** (haz152@ucsd.edu)

---

# Improving Diffusion-Based Virtual Try-On for Real-World Applications

> ECE285 Final Project — University of California, San Diego  
> **Tiancheng Chen** · **Haifan Zhao**

---

## Overview

This repository contains the code, experiment scripts, and supplementary materials for our ECE285 final project on **unpaired diffusion-based virtual try-on**.

Given a person image and an independently collected garment image, the proposed framework synthesizes a photorealistic try-on result that aims to:

- preserve the person's body structure and pose,
- transfer fine-grained garment details such as logos, textures, and embroidery,
- maintain realistic clothing appearance with natural wrinkles and shading.

The project was designed to improve diffusion-based virtual try-on under real-world conditions, especially when **paired supervision is unavailable**.

The core design includes:

- **Dual-path garment encoding**: global semantic features from CLIP and local texture features from a fine-grained encoder are fused.
- **Unpaired training strategy**: semantic correspondence is encouraged through augmentation and attention regularization.
- **Mask-guided generation**: a clothing mask is concatenated into the diffusion U-Net input to better localize garment synthesis.

The repository includes not only inference scripts and notebooks, but also the core model implementation under `src/`, which contains the modified attention, transformer, U-Net, and try-on pipeline source code.

---

## Repository Structure
```text
.
├── README.md
├── environment.yaml
├── inference.py
├── inference_dc.py
├── ECE285_FINAL_ablation (1).ipynb
├── preprocess.tar.gz
├── viton_ablation4.tar.gz
└── src/
    ├── tryon_pipeline.py
    ├── attentionhacked_*.py
    ├── transformerhacked_*.py
    ├── unet_block_hacked_*.py
    ├── unet_hacked_garmnet.py
    └── unet_hacked_tryon.py
```

### Repository Contents

The repository includes the following main files:

- **`environment.yaml`**  
  Conda environment specification for dependency setup.
- **`inference.py`**  
  Main inference script for generating try-on results.
- **`inference_dc.py`**  
  Inference or variant script used for conditioning-related experiments and ablation-style testing.
- **`ECE285_FINAL_ablation (1).ipynb`**  
  Notebook containing the ablation pipeline and experiment procedures used for the project analysis.
- **`preprocess.tar.gz`**  
  Archived preprocessing-related source files and utilities.
- **`viton_ablation4.tar.gz`**  
  Archived ablation-related source files or experiment assets.
- **`src/`**  
  Core model implementation directory. This folder contains the internal source code of the diffusion-based virtual try-on system, including:
  - modified attention modules,
  - modified transformer blocks,
  - hacked U-Net blocks,
  - garment-conditioned U-Net modules,
  - try-on generation U-Net modules,
  - and the main try-on pipeline implementation.

---

## Method Summary

### Architecture

The framework combines garment features, pose information, segmentation-derived conditioning, and diffusion-based generation in a unified pipeline.

A high-level view is:
```
Garment Image ──► CLIP Image Encoder ──────────────────► Feature Fusion ──►
               └─► Fine-grained Encoder ───────────────►                 │
                                                                          ▼
Person Image ──► Pose / DensePose ─────────────────────────────────► Diffusion U-Net ──► Final Try-On Image
             ├─► Agnostic Representation ───────────────────────────►
             └─► Clothing Mask ─────────────────────────────────────►
```

The U-Net takes a concatenated multi-channel conditioning input consisting of:

- noisy latent,
- VAE-encoded agnostic representation,
- binary clothing mask,
- and VAE-encoded DensePose representation.

This design allows the model to preserve body structure while restricting synthesis to garment-relevant regions.

### Core Ideas

- **Dual-path garment encoder**  
  A coarse semantic representation and a fine-grained texture representation are fused for better garment fidelity.
- **Implicit alignment through attention**  
  Instead of relying on an explicit warping module, the model uses attention-based conditioning to align garment information with the target person.
- **Attention regularization**  
  An attention total variation loss is used to reduce overly dispersed attention and suppress artifacts such as boundary color bleeding.

---

## Quantitative Results

### Main Comparison (VITON-HD)

| Method              | LPIPS ↓ | SSIM ↑ | FID ↓ |
|---------------------|---------|--------|-------|
| Pre-VTON (Baseline) | 0.153   | 0.826  | 9.98  |
| Ours (Proposed)     | 0.110   | 0.923  | 6.89  |

### Paired Ablation Study

| Configuration              | LPIPS ↓ | SSIM ↑ | FID ↓ |
|----------------------------|---------|--------|-------|
| Baseline (CLIP only)       | 0.142   | 0.854  | 8.75  |
| + Dual-Path Encoder        | 0.125   | 0.891  | 7.42  |
| + L_ATV (Full Model)       | 0.110   | 0.923  | 6.89  |

These results indicate that the proposed conditioning and regularization strategies improve both structural similarity and perceptual realism.

---

## Dataset

We use the **VITON-HD** benchmark for evaluation and analysis.

The dataset provides conditioning signals commonly used in virtual try-on, including:

- person images,
- garment images,
- clothing-agnostic representations,
- agnostic masks,
- DensePose maps,
- parsing results.

If you use a preprocessed version of the dataset, place it under a `data/` directory with a structure similar to:
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

If preprocessing is needed, refer to the files included in `preprocess.tar.gz`.

---

## Environment Setup

This repository uses a Conda environment file:
```bash
conda env create -f environment.yaml
conda activate viton
```

If your environment name differs from `viton`, replace it with the name specified in `environment.yaml`.

---

## How to Run

### 1. Main Inference

Run the main inference script to generate try-on results:
```bash
python inference.py
```

### 2. Conditioning or Variant Inference

For ablation-style or conditioning-related experiments:
```bash
python inference_dc.py
```

### 3. Ablation Notebook

To reproduce or inspect the ablation pipeline used in the project:
```bash
jupyter notebook "ECE285_FINAL_ablation (1).ipynb"
```

### 4. Core Model Source

The internal model implementation is located in:
```
src/
```

This directory contains the modified attention, transformer, U-Net, and pipeline code used by the try-on framework.

---

## Archived Source Files

Two archived files are included:

- `preprocess.tar.gz`
- `viton_ablation4.tar.gz`

Extract them with:
```bash
tar -xzvf preprocess.tar.gz
tar -xzvf viton_ablation4.tar.gz
```

These archives contain additional source files, preprocessing utilities, or ablation-related experiment files used in the project.

---

## Code-to-Paper Mapping

For clarity, the repository files correspond to the project components as follows:

| Component                                     | File(s)                                  |
|-----------------------------------------------|------------------------------------------|
| Main inference / qualitative generation       | `inference.py`                           |
| Conditioning variants / ablation-style inference | `inference_dc.py`                     |
| Ablation workflow and experiment analysis     | `ECE285_FINAL_ablation (1).ipynb`        |
| Core diffusion model implementation           | `src/`                                   |
| Preprocessing utilities                       | `preprocess.tar.gz`                      |
| Additional ablation assets or source files    | `viton_ablation4.tar.gz`                 |

---

## Notes on Reproducibility

This repository contains the main scripts, notebook-based ablation workflow, environment file, archived supporting source files, and the core model implementation under `src/`.

Before running the code, users should check:

- local path settings in the Python scripts and notebook,
- expected dataset folder structure,
- checkpoint locations,
- output directories.

If needed, update paths in the scripts to match the local environment.

---

## Acknowledgements

This project builds on ideas and resources from prior virtual try-on and diffusion-based generation works, including datasets and open-source implementations related to VITON-HD, StableVITON, and related baselines.

---

## Citation
```bibtex
@misc{chen2026viton,
  title   = {Improving Diffusion-Based Virtual Try-On for Real-World Applications},
  author  = {Tiancheng Chen and Haifan Zhao},
  year    = {2026},
  note    = {ECE285 Final Project, University of California, San Diego}
}
```
```
