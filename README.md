# Benchmarking Detectors for the RIVA Cervical Cytology Challenge

This repository contains the official code for **Team jht010312**'s solution to the ISBI 2026 RIVA Cervical Cytology Challenge. We fine-tuned the **Co-DETR** model with  `ViT-coco-instance` weights,  which achieved **1st place** in Track A (Multi-class Detection) and **3rd place** in Track B (Single-class Detection).

🔗 **[Download our fine-tuned Co-DETR weights here](https://drive.google.com/drive/folders/1qsezIvFR-f6I_FpvX_aiF3kLWxZJfypX?usp=drive_link)**

## 🏆 Performance Highlights

Our solution secured top placements across both evaluation tracks:

* **Track A (Multi-class Detection):**

  * **1st Place mAP: 0.2273 (Preliminary) / 0.1937 (Final)**
  * Focus: Fine-grained classification of 8 categories (NILM, ENDO, INFL, ASC-US, LSIL, HSIL, ASC-H, SCC).
* **Track B (Single-class Detection):**

  * **3rd Place mAP: 0.6044 (Preliminary) / 0.5906 (Final)**

## 🚀 How to reproduce our results

```bash
python infer.py --config [config path] --weights [model weights] --img-dir [your image path]
```

## ⚙️ Methodology

### Benchmark Models

We evaluated a diverse set of architectures:

* **Co-DETR:** Collaborative Hybrid Assignments Training (Best performing).
* **D-Fine:** Distribution refinement for regression.
* **EVA-02:** Next-generation transformer based on masked image modeling.
* **Mask R-CNN & Keypoint R-CNN:** Robust two-stage baselines.

### Training Details

* **Framework:** PyTorch 2.0.1.
* **Input Resolution:** 1024 × 1024 pixels.
* **Augmentation:** Random horizontal and vertical flipping.
* **Co-DETR Configuration:**
  * Epochs: 12
  * Batch Size: 2
  * Optimizer: AdamW (weight decay 0.0001)
  * Learning Rate: 0.0001 (with 500-step linear warmup).
  * Pre-training: Initialized with `ViT-coco-instance` weights.

### Inference Strategy

Predictions are generated at 1024×1024 resolution. To meet the challenge requirement of fixed 100×100 pixel bounding boxes, we apply post-processing: if a detected box is within a 5-pixel margin of the image boundary, we dynamically pad its truncated spatial axis inward to ensure the 100-pixel dimension.

## 🔗 Resources and Links

* **Challenge Website:** [RIVA Cervical Cytology Challenge](https://lia-ditella.github.io/rivachallenge/)
* **Model Weights:** [Download our fine-tuned Co-DETR weights](https://drive.google.com/drive/folders/1qsezIvFR-f6I_FpvX_aiF3kLWxZJfypX?usp=drive_link)
* **Codebase:** Built upon [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR).
