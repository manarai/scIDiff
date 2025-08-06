# 👬 scIDiff: Single-cell Inverse Diffusion

**scIDiff** is a deep generative framework for modeling, denoising, and inverse-designing single-cell gene expression profiles using **score-based diffusion models**.

This project combines the strengths of **denoising diffusion probabilistic models (DDPMs)** with **perturbation-aware learning** to create a toolkit for perturbation prediction, trajectory simulation, and transcriptional phenotype design at single-cell resolution.

---

## 🌟 Mathematical Framework

We model the distribution of scRNA-seq expression profiles $\mathbf{x} \in \mathbb{R}^G$, where $G$ is the number of genes, using a **score-based diffusion process**:

### 1. Forward Diffusion

We define a forward stochastic process $q(\mathbf{x}_t \,|\, \mathbf{x}_0)$ that progressively adds noise to gene expression $\mathbf{x}_0$:

$q(\mathbf{x}_t \,|\, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_0, (1 - \alpha_t)\mathbf{I})$

where $\alpha_t \in (0, 1]$ controls noise schedule and $t \in [0, T]$ is the diffusion timestep.

### 2. Reverse Diffusion

We learn a **score function** $s_\theta(\mathbf{x}_t, t, \mathbf{c}) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$, conditioned on biological context $\mathbf{c}$ (e.g., perturbation, cell type):

$\frac{d\mathbf{x}_t}{dt} = -\frac{1}{2} s_\theta(\mathbf{x}_t, t, \mathbf{c}) + \mathcal{N}(0, \mathbf{I})$

We simulate this reverse-time process using discretized denoising steps to sample from $p(\mathbf{x}_0 \,|\, \mathbf{c})$.

### 3. Conditional Sampling for Inverse Design

Given a **target phenotype** $\phi \in \mathbb{R}^K$ (e.g., marker gene activity, pathway scores), we guide the diffusion process toward satisfying:

$\mathbf{x}_0^* = \arg\max_{\mathbf{x}} \; \mathbb{E}_{p_\theta(\mathbf{x}_0)}[\text{sim}(f(\mathbf{x}), \phi)]$

where $f(\mathbf{x})$ maps gene expression to phenotype, and $\text{sim}(\cdot, \cdot)$ is a similarity or loss function (e.g., cosine similarity, MSE).

We implement **classifier-free guidance** or **gradient guidance** to steer generation toward $\phi$.

---

## 🎯 Purpose

* Generate realistic single-cell expression profiles
* Denoise scRNA-seq data with learned diffusion-based score functions
* Predict gene expression changes across drug, CRISPR, or time-based perturbations
* Inverse design: generate gene expression programs from desired cell-state phenotypes
* Connect phenotypes to causal regulators (e.g., transcription factors, small molecules)

---

## 🧐 Background

Single-cell technologies allow high-resolution interrogation of cellular response to perturbations. While models like **scGen** and **CPA** approximate these perturbations via latent space shifts, they struggle with nonlinearity and sparse data.

**scIDiff** leverages **score-based diffusion modeling** to better model the full gene expression manifold and perform controlled generation of transcriptional states.

---

## 🩰 Core Components

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
* 🔜 **Integrate TF/drug mapping** via SCENIC+ and knowledge graphs
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
* Multimodal (RNA + ATAC) extensions

---

## 📜 License

MIT License

