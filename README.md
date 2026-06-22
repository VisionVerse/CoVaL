# CCVL: Compact Commonality–Variation Learning for Remote Sensing Change Detection


<p align="center">
   <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python"> 
    <img src="https://img.shields.io/badge/Code-PyTorch-orange" alt="Code">
    <img src="https://img.shields.io/badge/Dataset-LEVIR--CD%20%7C%20LEVIR--CD+%20%7C%20WHU--CD%20%7C%20CDD--CD-yellow" alt="Dataset">
    <img src="https://img.shields.io/badge/Backbone-VMamba-cyan" alt="Backbone">
    </p>
    

This repository provides the official implementation of: **From Disentanglement to Localization: Compact Commonality–Variation Learning for Remote Sensing Change Detection.**

CCVL is a remote sensing change detection framework that follows a two-stage paradigm:

> **What changes?** → **Where changes occur?**

CCVL decouples bi-temporal features into commonality and variation representations, and then progressively localizes changed regions using variation-driven spatial reasoning.

<p align="center">
  <img src="CCVL_framework.jpg" width="95%">
</p>

Remote sensing change detection aims to identify changed regions between two images captured at different times. 
Existing methods often directly fuse or subtract bi-temporal features, which may entangle invariant background information with change-related variations and introduce redundant responses.

To address this issue, we propose **CCVL**, a compact commonality–variation learning framework.

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
git clone https://github.com/your-username/CCVL.git
cd CCVL
conda create -n ccvl python=3.8 -y
conda activate ccvl
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

Please organize each dataset as follows:

```text
dataset/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

where:

- `A/` contains images at time T1
- `B/` contains images at time T2
- `label/` contains binary change masks

Supported datasets may include:

- LEVIR-CD
- WHU-CD
- DSIFN-CD
- SYSU-CD
- CDD

Please update the dataset path in the corresponding configuration file before training or testing.

---

## Training

```bash
python train.py \
  --config configs/ccvl_levir.yaml \
  --dataset_path /path/to/dataset \
  --save_path checkpoints/ccvl_levir
```

Or run:

```bash
bash scripts/train_levir.sh
```


## Testing

```bash
python test.py \
  --config configs/ccvl_levir.yaml \
  --checkpoint checkpoints/ccvl_levir/best.pth \
  --dataset_path /path/to/dataset
```

Or run:

```bash
bash scripts/test_levir.sh
```

The predicted change maps will be saved in:

```text
results/
└── ccvl_levir/
```

---

## Pretrained Models

Pretrained weights will be released after publication.

| Dataset | Model | Download |
|---|---|---|
| LEVIR-CD | CCVL | Coming soon |
| WHU-CD | CCVL | Coming soon |
| DSIFN-CD | CCVL | Coming soon |

---

## Results

| Dataset | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|
| LEVIR-CD | - | - | - | - |
| WHU-CD | - | - | - | - |
| DSIFN-CD | - | - | - | - |

More detailed comparisons and ablation studies can be found in the paper.

---

## Visualization

<p align="center">
  <img src="figures/visualization.png" width="95%">
</p>

CCVL produces accurate and structurally consistent change maps, especially in challenging cases with complex backgrounds, small changed regions, and blurred boundaries.



## Repository Structure

```text
CCVL/
├── configs/
├── datasets/
├── models/
│   ├── ccvl.py
│   ├── lcvd.py
│   ├── vpl.py
│   └── backbone/
├── losses/
├── utils/
├── scripts/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```


## Acknowledgement

This project is built upon several excellent open-source repositories and remote sensing change detection benchmarks. We sincerely thank the authors for their contributions.


