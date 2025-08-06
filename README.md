# 🧬 RNA-Diffusion: A Diffusion Model for Inverse Design of Single-Cell Gene Expression

**RNA-Diffusion** is a generative framework for modeling, denoising, and inverse designing single-cell gene expression profiles using **score-based diffusion models**.

This project extends the ideas from scGen, CPA, and RNA-Diffusion by introducing a **perturbation-aware, property-conditioned diffusion model** for single-cell transcriptomics.

---

## 🚀 Project Goals

- **Denoise** and simulate scRNA-seq expression profiles using diffusion models
- **Condition generation** on biological covariates (e.g., drug, timepoint, cell type)
- Enable **inverse design** of gene expression: generate profiles that match desired phenotypes (e.g., marker up/down)
- Bridge transcriptional signatures with **potential perturbations (e.g., drugs, TFs)**

---

## 🧠 Background

Single-cell perturbation datasets (like Perturb-seq) capture how gene programs change in response to stimuli. But modeling these changes remains hard due to:

- High dimensionality, sparsity, and noise
- Nonlinear cell state transitions
- Complex interactions between covariates (e.g., drug + cell type)

We propose using **denoising diffusion probabilistic models (DDPMs)** to model the distribution of gene expression vectors and enable **conditional sampling** from desired outcomes.

---

## 🧰 Methods Overview

### 📊 Input
- Gene expression matrix (cells × genes)
- Metadata: perturbation, cell type, dose, timepoint

### 🌬️ Diffusion Model
- **Forward process**: Add Gaussian or biologically-informed noise to gene expression
- **Reverse process**: Train a neural network to denoise
- **Conditioning**: Perturbation/cell-type embeddings steer the generation
- **Guided sampling**: Inverse design via classifier-free guidance or Langevin dynamics

---

## 🧪 Applications

- **scRNA-seq denoising**
- **Simulation of unseen perturbation outcomes**
- **Perturbation-aware cell fate prediction**
- **Inverse design of transcriptional phenotypes**
- **Linking desired phenotypes to candidate TFs or small molecules**

---

## 📦 Code Structure

```bash
RNA-Diffusion/
├── data/                   # Preprocessed datasets (Perturb-seq, sciPlex, etc.)
├── models/                 # Score-based diffusion networks
├── conditioning/           # Embedding models for perturbation/cell covariates
├── sampling/               # Classifier-free and guided sampling methods
├── evaluation/             # Metrics, visualization, benchmarking
├── notebooks/              # Demos and experiments
└── README.md               # You are here
