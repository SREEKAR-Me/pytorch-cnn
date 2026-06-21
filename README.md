Multi-label land-use classification on EuroSAT satellite imagery.
Fine-tuned ResNet-50 trained with 3-stage progressive unfreezing.

---

## Problem

Sentinel-2 satellite tiles frequently contain more than one land-use type —
a tile can be both Residential and River at the same time. Standard single-label
classifiers force the model to pick exactly one class, losing this information.

This project frames the task as **multi-label classification**:
the model outputs an independent probability for each of 10 land-use classes,
allowing multiple simultaneous predictions per tile.

---

## Dataset

- **EuroSAT** — 27,000 Sentinel-2 RGB images at 64×64 pixels
- 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
  Pasture, PermanentCrop, Residential, River, SeaLake
- Multi-label extension: ~15% of tiles assigned a secondary label
  based on geographic co-occurrence rules (see `src/create_labels.py`)

---

## Architecture
**Backbone**: ResNet-50 pretrained on ImageNet
- **Head**: Dropout(0.3) → Linear(2048 → 10) → *no activation*
- **Loss**: BCEWithLogitsLoss (sigmoid applied internally)
- **Why not softmax?** Softmax forces class probabilities to sum to 1,
  which is wrong for multi-label problems. Sigmoid treats each class
  independently, which is correct here.

## Training Strategy

| Stage | Layers Trained | Epochs | Learning Rate |
|-------|---------------|--------|---------------|
| 1 | New head only | 5 | 1e-3 |
| 2 | Head + layer3 + layer4 | 8 | 1e-4 |
| 3 | All layers | 15 | 1e-5 (cosine) |

## ABLATION STUDY RESULTS

| Condition | mAP | F1 Macro |
|-------|---------------|--------|
| Random init (no pretrained)| 0.8207 | 0.7119 |
| Frozen pretrained backbone | 0.8759 | 0.7478 |
| 3-stage fine-tuned (ours)  | 0.9688 | 0.9466 |
