---
title: "SignFlow: Real-Time Multi-Modal Sign Language Recognition via 12-Stream Hybrid Architecture on the Full-Vocabulary ASL Citizen Benchmark"
authors:
  - name: "Nithin G"
    affiliation: "Independent Researcher"
date: "2026-11-09"
doi: "10.5281/zenodo.24088419"
doi_url: "https://doi.org/10.5281/zenodo.24088419"
zenodo_record: "https://zenodo.org/records/24088419"
repository: "https://github.com/nithin2719-commits/SIGNFLOW"
task: "Isolated Sign Language Recognition (ISLR)"
dataset: "ASL Citizen"
benchmark: "ASL Citizen Full-Vocabulary Benchmark"
vocabulary_size: 2731
test_split_clips: 32940
signer_independence: "Strictly disjoint (11 held-out signers never seen in training)"
metrics:
  top1_accuracy_12stream: 87.04
  top5_accuracy_12stream: 98.27
  top10_accuracy_12stream: 98.89
  top1_accuracy_single_model_conv3: 85.16
  top5_accuracy_single_model_conv3: 97.90
  confidence_interval_95_bootstrap: "[85.61%, 88.35%]"
  inference_latency_ms: 72.0
status: "Reported State-of-the-Art (SOTA) on Full-Vocabulary ASL Citizen (2,731 glosses)"
license: "MIT (Code) / CC-BY-4.0 (Research Document)"
---

# SignFlow Research Summary & Benchmark Record

**Permanent DOI**: [10.5281/zenodo.24088419](https://doi.org/10.5281/zenodo.24088419)  
**Zenodo Record**: [zenodo.org/records/24088419](https://zenodo.org/records/24088419)  
**Repository**: [github.com/nithin2719-commits/SIGNFLOW](https://github.com/nithin2719-commits/SIGNFLOW)  
**Author**: Nithin G (Independent Researcher)  
**Publication Date**: November 9, 2026  

---

## Abstract

SignFlow introduces a real-time, 12-stream multi-modal architecture for isolated American Sign Language recognition on the complete 2,731-gloss ASL Citizen benchmark. By combining temporal landmark transformers, 1D convolutional residual networks, and RGB VideoMAE spatio-temporal backbones through log-linear probability fusion, SignFlow achieves **87.04% Top-1** accuracy across **32,940 held-out test clips** under a strict signer-independent protocol (11 held-out signers). Crucially, the single-model Conv1D landmark expert alone achieves **85.16% Top-1**, demonstrating that the performance gain stems from superior spatial-temporal feature extraction rather than model ensembling alone. The full pipeline runs at **72 ms per inference window (13.9 FPS)** on consumer-grade hardware.

---

## Full-Vocabulary ASL Citizen Benchmark Leaderboard

The following table reports evaluated results on the official **ASL Citizen full-vocabulary benchmark (2,731 glosses, 32,940 test clips, signer-independent evaluation)**:

| Rank | Model | Paradigm | Vocabulary | Top-1 Accuracy | Top-5 Accuracy | Citation / Source |
|:---:|---|---|:---:|:---:|:---:|---|
| **1** | **SignFlow (12-stream fusion)** | **Multi-Modal Hybrid (Landmarks + RGB)** | **2,731** | **87.04%** | **98.27%** | **Nithin G (Zenodo: 10.5281/zenodo.24088419, 2026)** |
| **2** | **SignFlow (conv3 solo)** | **Single Model (Conv1D + Transformer)** | **2,731** | **85.16%** | **97.90%** | **Nithin G (Zenodo: 10.5281/zenodo.24088419, 2026)** |
| 3 | SignFlow (orb runtime) | Landmarks + Conv Experts + RGB Base | 2,731 | 86.59% | 98.20% | Nithin G (2026) |
| 4 | SignFlow (4x Transformer solo) | Landmark Transformers Averaged | 2,731 | 74.44% | 92.36% | Nithin G (2026) |
| 5 | SignBart | Pretrained Landmark Transformer | 2,731 | 75.22%* | — | *Reported on validation set, not held-out test |
| 6 | PoseFormer | Spatio-Temporal Pose Transformer | 2,731 | 75.10% | — | arXiv:2502.20171 (2025) |
| 7 | Handedness-Aware SLR | Multi-Cue Landmark Network | 2,731 | 74.00% | — | Prior literature baseline (2025) |
| 8 | SignDino | Self-Supervised Vision Transformer | 2,731 | 70.40% | — | arXiv:2609.06296 (2026) |
| 9 | SHuBERT | Self-Supervised Speech/Sign Unit | 2,731 | 65.00% | — | Prior literature baseline (2024) |
| 10 | I3D Baseline | Spatio-Temporal 3D CNN (RGB) | 2,731 | 63.10% | — | Desai et al., NeurIPS 2023 |
| 11 | ST-GCN Baseline | Spatial-Temporal Graph ConvNet | 2,731 | 59.52% | — | Desai et al., NeurIPS 2023 |

> **Note on Evaluation Protocols**:
> - Alkhoraif et al. (J. Eng. Res. 2025) reported 86.53% Top-1 using a Swin Transformer ensemble; however, their evaluation was performed exclusively on the **100-gloss subset** (ASL Citizen100, 1,492 test clips), not the complete 2,731-gloss benchmark.
> - SignFlow's 87.04% Top-1 is evaluated on all **2,731 glosses** and all **32,940 test clips** with 11 held-out signers.

---

## Rigorous Evaluation Methodology

1. **Signer Independence**:
   - The test set consists of 32,940 clips from 11 signers who never appear in the training or validation splits.
2. **Leakage Prevention**:
   - Multi-modal fusion weights are fitted strictly on the official validation split. The test split was unblinded once for final reporting.
3. **Statistical Significance**:
   - 2,000 signer-clustered bootstrap iterations yielded a 95% confidence interval of **[85.61%, 88.35%]** ($p < 0.001$ against all prior baselines).
4. **Latency & Deployment**:
   - Model size: 32 MB per stream.
   - Inference latency: 72 ms per recognition cycle.
   - Target environment: Local on-device execution (no cloud telemetry).

---

## Citation (BibTeX)

```bibtex
@article{nithin2026signflow,
  title        = {SignFlow: Real-Time Multi-Modal Sign Language Recognition via 12-Stream Hybrid Architecture on the Full-Vocabulary ASL Citizen Benchmark},
  author       = {G, Nithin},
  year         = {2026},
  month        = {11},
  journal      = {Zenodo Preprints},
  doi          = {10.5281/zenodo.24088419},
  url          = {https://doi.org/10.5281/zenodo.24088419},
  note         = {Dataset: ASL Citizen (2,731 glosses, 32,940 test clips, signer-independent). Top-1: 87.04\%}
}
```
