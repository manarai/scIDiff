# 🧬 scIDiff: Single-cell Inverse Diffusion

**scIDiff** is a deep generative framework for modeling, denoising, and inverse-designing single-cell gene expression profiles using **score-based diffusion models**.

This project combines the strengths of **denoising diffusion probabilistic models (DDPMs)** with **perturbation-aware learning** to create a toolkit for perturbation prediction, trajectory simulation, and transcriptional phenotype design at single-cell resolution.

---
<img width="1536" height="1024" alt="ChatGPT Image Aug 7, 2025, 03_53_41 PM" src="https://github.com/user-attachments/assets/babacac3-fc89-4ddd-b9b2-8003d36dece3" />

---

## 🎯 Purpose

* Generate realistic single-cell expression profiles
* Denoise scRNA-seq data with learned diffusion-based score functions
* Predict gene expression changes across drug, CRISPR, or time-based perturbations
* Inverse design: generate gene expression programs from desired cell-state phenotypes
* Connect phenotypes to causal regulators (e.g., transcription factors, small molecules)

---

## 🧠 Background

Single-cell technologies allow high-resolution interrogation of cellular response to perturbations. While models like **scGen** and **CPA** approximate these perturbations via latent space shifts, they struggle with nonlinearity and sparse data.

**scIDiff** leverages **score-based diffusion modeling** to better model the full gene expression manifold and perform controlled generation of transcriptional states.

---

## 🧰 Core Components

### 🧬 Diffusion Model

* Forward noise process on gene expression vectors
* Reverse denoising learned via neural networks (score function)
* Score conditioning using biological covariates (e.g., drug, cell type)

### 🎯 Inverse Design Engine

* Accepts target phenotypes (e.g., marker genes ↑ or ↓)
* Performs guided generation toward those transcriptional profiles

### 🧪 Integration Modules

* SCENIC+ / TF-gene mapping
* Drug–gene effect priors (e.g., LINCS, DrugBank)
* Cell trajectory inference tools (e.g., scVelo)

---

## 🔬 Mathematical Framework

Let $x_0 \in \mathbb{R}^d$ be the clean gene expression vector (e.g., log-normalized counts), and $x_t$ be its noisy version at diffusion step $t$.

### Forward Process

We apply Gaussian noise incrementally:
$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$
with schedule $\{\beta_t\}_{t=1}^T$.

### Reverse Process (Learned)

We learn a parameterized score model $\nabla_{x_t} \log p(x_t) \approx s_\theta(x_t, t, c)$, conditioned on covariates $c$ (e.g., drug, cell type, TF module).

Sampling is done by solving the reverse SDE or ODE:
$dx = [f(x, t) - g(t)^2 \nabla_x \log p(x_t)]dt + g(t) d\bar{w}$

### Inverse Design Objective

Given a phenotype descriptor $y$ (e.g., "high IL2RA, low exhaustion"), we optimize sampling path to steer toward matching features:
$\min_{x_0} \mathcal{L}_{\text{target}}(f(x_0), y) \quad \text{while} \quad x_0 \sim p_\theta(x_0 | x_T)$

---

## 📊 Framework Schematic

```
Target Phenotype
   ↓
[Inverse Guidance]
   ↓
Sample x_T → x_0 (DDPM)
   ↓
Generated Gene Expression
   ↓
↓ SCENIC+ / Drug Mapping
   ↓
TFs / Molecules → Validation
```

---

## 📦 Code Structure

```bash
scIDiff/
├── data/               # Preprocessed datasets (Perturb-seq, sciPlex, etc.)
├── models/             # Diffusion networks and conditioning modules
├── training/           # Loss functions and optimization logic
├── sampling/           # Guided / inverse sampling routines
├── evaluation/         # Benchmarking and metrics
├── integration/        # Links to SCENIC+, drug priors, etc.
└── notebooks/          # Demos and experiments
```

---

## 🗓️ Roadmap

* ✅ **Prototype DDPM** for scRNA-seq denoising
* 🚧 **Add perturbation conditioning** (CPA-style architecture)
* 🚧 **Implement inverse design** from phenotype-level targets
* 🔬 **Publish benchmark** and submit to **NeurIPS / ICLR**

---

## 📚 References

* Lotfollahi et al. *scGen: Modeling single-cell perturbation response*, Nat. Methods (2019)
* Hetzel et al. *CPA: Compositional Perturbation Autoencoder*, bioRxiv (2021)
* Song et al. *Score-Based Generative Modeling through SDEs*, NeurIPS (2021)
* RNA-Diffusion (2024). [*https://arxiv.org/abs/2403.11247*](https://arxiv.org/abs/2403.11247)

---

## 🤝 Contributing

Contributions are welcome — open an issue to start a discussion. We're especially looking for help with:

* TF/pathway-guided inverse modeling
* Drug perturbation prior integration


---

## 📜 License

MIT License
