# On the Adversarial Robustness of Temporal Ethereum Malicious Activity Detection




### KamiPan Lab · 2026

---

## Overview

This repository contains the experimental implementation and evaluation scripts for our research on adversarial attacks against temporal Ethereum malicious activity detection systems.

The project focuses on how adversarial perturbations affect temporal behavioral detection models across different:

- attack settings
- temporal intervals
- feature groups
- classifier architectures

The repository also includes preliminary defense experiments based on adversarial training.

---

## Repository Structure

```text
.
├── baseline/                                   # Baseline attack experiments
├── defense/                                    # Adversarial training experiments
├── summary/                                    # Processed experimental summaries
├── datasets/                                   # Datasets for binary and multi-class detection
├── experiments.py                              # Main experiment entry
├── targeted_attack_with_all_features/          # Targeted attacks using all features
├── targeted_attack_with_transaction_features/  # Targeted attacks using transaction features
├── temporal_attack_with_all_features/          # Temporal attacks using all features
├── temporal_attack_with_code_features/         # Temporal attacks using code features
├── temporal_attack_with_transaction_features/  # Temporal attacks using transaction features
└── .gitignore