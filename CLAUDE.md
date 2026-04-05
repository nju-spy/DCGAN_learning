# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DCGAN (Deep Convolutional Generative Adversarial Network) implementation in PyTorch, based on [arXiv:1511.06434](https://arxiv.org/abs/1511.06434). Single-file project (`main.py`) that trains a GAN on various image datasets.

## Running the Project

Train on a dataset:
```bash
python main.py --dataset cifar10 --dataroot /path/to/data --outf /path/to/output
python main.py --dataset lsun --dataroot /path/to/lsun --classes bedroom
python main.py --dataset folder --dataroot /path/to/images  # custom dataset
```

Quick single-iteration test:
```bash
python main.py --dataset fake --dry-run
```

There is no separate test suite or lint configuration.

## Architecture

All code lives in `main.py`. Key sections:

- **Lines 1–55**: Argument parsing and device setup (CUDA / Intel XPU via `--accel`)
- **Lines 57–110**: Dataset and DataLoader construction (cifar10, mnist, imagenet, folder, lfw, fake)
- **Lines 112–125**: Weight initialization (`weights_init`) applied to Conv and BatchNorm layers
- **Lines 126–160**: `Generator` — latent vector → 64×64 image via transposed convolutions, BatchNorm, ReLU, Tanh
- **Lines 169–200**: `Discriminator` — image → real/fake score via strided convolutions, BatchNorm, LeakyReLU, Sigmoid
- **Lines 202–220**: Model instantiation, optional checkpoint loading (`--netG`, `--netD`), multi-GPU wrapping
- **Lines 222–277**: Training loop — alternates D and G updates using BCELoss + Adam; saves sample images every 100 iters and `.pth` checkpoints every epoch

**Key hyperparameter flags:** `--nz` (latent dim, default 100), `--ngf`/`--ndf` (filter counts, default 64), `--lr` (default 0.0002), `--beta1` (Adam β₁, default 0.5), `--ngpu` (default 1).

**Outputs written to `--outf` directory:**
- `real_samples.png`
- `fake_samples_epoch_XXX.png`
- `netG_epoch_X.pth` / `netD_epoch_X.pth`
