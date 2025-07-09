# BayesTTA: Continual-Temporal Test-Time Adaptation for Vision-Language Models via Gaussian Discriminant Analysis

This repository contains the official implementation of **BayesTTA**, a principled Bayesian adaptation framework designed for continual-temporal test-time adaptation (CT-TTA) of vision-language models (VLMs). BayesTTA incrementally estimates class-conditional Gaussian distributions, adaptively determines covariance structures via hypothesis testing, and performs calibrated inference through Gaussian discriminant analysis (GDA).

> **Submission**  
> [Project Page (Coming Soon)]() | [Paper (arXiv)]() | [BibTeX](#citation)

<p align="center">
  <img src="assets/overview.png" width="80%" />
</p>

---

## 📦 Installation

Follow these steps to set up a conda environment and ensure all necessary packages are installed:

git clone https://github.com/cuishuang99/BayesTTA.git
cd BayesTTA

conda create -n bayestta python=3.7
conda activate bayestta

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt

---

## 🚀 Running

To execute BayesTTA, navigate to the `scripts` directory, where you'll find several bash scripts available. Each script applies BayesTTA to two benchmarks using either the ResNet50 or ViT-B/16 backbone. The datasets are processed sequentially as indicated by the order in each script.

Below are instructions for running BayesTTA on both CT-TTA and Cross-Domain benchmarks:

---

### 🌀 CT-TTA Benchmark

**ResNet50**  
Run BayesTTA on fMoW, Yearbook, RMNIST, and CIFAR10-Light using the ResNet50 model:

bash ./scripts/run_cttta_benchmark_rn50.sh
