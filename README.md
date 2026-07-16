# CoVaL: Compact Commonality–Variation Learning for Remote Sensing Change Detection


<p align="center">
   <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python"> 
    <img src="https://img.shields.io/badge/Code-PyTorch-orange" alt="Code">
    <img src="https://img.shields.io/badge/Dataset-LEVIR--CD%20%7C%20SYSU--CD%20%7C%20WHU--CD%20%7C%20CDD--CD-yellow" alt="Dataset">
    <img src="https://img.shields.io/badge/Backbone-VMamba-cyan" alt="Backbone">
    </p>
    

This repository provides the official implementation of: **From Disentanglement to Localization: Compact Commonality–Variation Learning for Remote Sensing Change Detection.**

CoVaL is a remote sensing change detection framework that follows a two-stage paradigm:

> **What changes?** → **Where changes occur?**

CoVaL decouples bi-temporal features into commonality and variation representations, and then progressively localizes changed regions using variation-driven spatial reasoning.

<p align="center">
  <img src="assets/images/CoVaL_framework.jpg" width="95%">
</p>

Remote sensing change detection aims to identify changed regions between two images captured at different times. 
Existing methods often directly fuse or subtract bi-temporal features, which may entangle invariant background information with change-related variations and introduce redundant responses.

To address this issue, we propose **CoVaL**, a compact commonality–variation learning framework.

- **Stage I: LCVD**  
  Low-redundancy Commonality–Variation Decoupling answers **“what changes?”** by separating invariant commonality from change-sensitive variation.

- **Stage II: VPL**  
  Variation-guided Progressive Localization answers **“where changes occur?”** by progressively decoding variation features from deep semantic levels to shallow spatial details.

---

## News

- Code and pretrained models will be released soon.
- The paper is currently under review.

---



## Installation

```bash
git clone https://github.com/VisionVerse/CoVaL.git
cd CoVaL
conda create -n coval python=3.10 -y
conda activate coval
pip install -r requirements.txt
```

A typical environment includes:

```text
torch
torchvision
numpy
opencv-python
tqdm
einops
timm
scikit-learn
matplotlib
```



## Dataset Preparation

Each dataset follows a unified `A/B/label/list` structure:

```text
dataset/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

- `A/` — images at time T1
- `B/` — images at time T2
- `label/` — binary change masks (0 = unchanged, 255 = changed)
- `list/` — per-line filenames (no path prefix) for train/val/test splits

The data loader supports flexible matching: a filename `train_001` in the list matches `train_001.png`, `train_001.jpg`, `001.png`, etc.

Evaluated datasets:

| Dataset | `--dataset` | Format | Samples | Train/Val/Test |
|---------|------------|--------|---------|----------------|
| LEVIR-CD-256 | `LEVIR-CD-256` | `.png` | 10,192 | 7,120 / 1,024 / 2,048 |
| SYSU-CD-256 | `SYSU-CD-256` | `.png` | 20,000 | 12,000 / 4,000 / 4,000 |
| WHU-CD-256 | `WHU-CD-256` | `.png` | 7,434 | 5,947 / 743 / 744 |
| CDD-CD-256 | `CDD-CD-256` | `.jpg` | 15,998 | 10,000 / 2,998 / 3,000 |

> CDD-CD-256 uses `.jpg` format; the other three datasets use `.png`.

---

## Training

```bash
python train.py \
  --cfg configs/vssm_tiny_224.yaml \
  --dataset_path /path/to/dataset \
  --dataset LEVIR-CD-256 \
  --pretrained_weight_path pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cfg` | required | YAML config path |
| `--dataset_path` | required | Dataset root directory |
| `--dataset` | LEVIR-CD-256 | Dataset name |
| `--batch_size` | 12 | Batch size per GPU |
| `--max_iters` | 50000 | Training iterations |
| `--pretrained_weight_path` | '' | VMamba pretrained weight |


## Testing

```bash
python test.py \
  --cfg configs/vssm_tiny_224.yaml \
  --test_dataset_path /path/to/dataset \
  --test_data_list_path /path/to/dataset/list/test.txt \
  --resume saved_models/CoVaL_run/best_model_f1_xxxx.pth \
  --dataset LEVIR-CD-256 \
  --batch_size 1
```

With post-processing:

```bash
python test.py \
  --cfg configs/vssm_tiny_224.yaml \
  --test_dataset_path /path/to/dataset \
  --test_data_list_path /path/to/dataset/list/test.txt \
  --resume saved_models/CoVaL_run/best_model_f1_xxxx.pth \
  --dataset LEVIR-CD-256 \
  --use_post_processing \
  --post_min_area 50
```

The predicted change maps and evaluation metrics will be saved in:

```text
results/
└── CoVaL/
    ├── change_map/
    └── summary_metrics.txt
```

---

---

## Results

| Dataset | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|
| LEVIR-CD-256 | 99.20 | 92.08 | 85.33 | 91.66 |
| SYSU-CD-256 | 92.56 | 84.46 | 73.11 | 79.57 |
| WHU-CD-256 | 99.62 | 95.12 | 90.69 | 94.92 |
| CDD-CD-256 | 99.26 | 96.87 | 93.93 | 96.45 |

More detailed comparisons and ablation studies can be found in the paper.

---

## Visualization

<p align="center">
  <img src="assets/images/Visualization_Result_1.jpg" width="95%">
  <img src="assets/images/Visualization_Result_2.jpg" width="95%">
</p>

CoVaL produces accurate and structurally consistent change maps, especially in challenging cases with complex backgrounds, small changed regions, and blurred boundaries.



## Repository Structure

```text
CoVaL/
├── train.py                          # Training entry point
├── test.py                           # Inference entry point
├── requirements.txt
├── .gitignore
├── configs/
│   ├── config.py
│   └── vssm_tiny_224.yaml
├── datasets/
│   ├── imutils.py
│   └── make_data_loader.py
├── models/
│   ├── coval.py                      # CoVaLModel (main)
│   ├── lcvd.py                       # Stage I: CSP + FCD
│   ├── vpl.py                        # Stage II: CVA + CLR + ESE
│   └── backbone/
│       ├── coval_backbone.py         # CoVaLBackbone
│       ├── vmamba.py                 # VSSM / SS2D
│       └── csm_triton.py             # Triton cross-scan
├── losses/
│   ├── edge_loss.py
│   └── lovasz_loss.py
├── utils/
│   ├── metrics.py
│   ├── eval_segm.py
│   ├── post_processing.py
│   └── mcd_utils.py
├── assets/
│   └── images/
│       ├── CoVaL_framework.jpg
│       ├── Visualization_Result_1.jpg
│       └── Visualization_Result_2.jpg
├── kernels/
│   └── selective_scan/               # CUDA kernels
├── classification/                    # VMamba reference
├── pretrained_weight/
└── docs/
```


## Acknowledgement

This project is built upon several excellent open-source repositories and remote sensing change detection benchmarks. We sincerely thank the authors for their contributions.


