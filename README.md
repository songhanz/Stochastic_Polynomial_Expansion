# Stochastic Polynomial Expansion (PTPE) for Uncertainty Propagation through Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Published in TMLR](https://img.shields.io/badge/Published%20in-TMLR%20(06%2F2025)-blue)](https://openreview.net/forum?id=lyDRBhUjhv)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-orange.svg)](https://www.mathworks.com/)

This repository contains the official implementation of the **Pseudo-Taylor Polynomial Expansion (PTPE)** method, a generalized framework for propagating multivariate Gaussian distributions through neural networks with closed-form solutions.

> **Paper:** *A Stochastic Polynomial Expansion for Uncertainty Propagation through Networks*  
> **Authors:** Songhan Zhang, ShiNung Ching  
> **Published in:** Transactions on Machine Learning Research (TMLR), 06/2025  
> **OpenReview:** [https://openreview.net/forum?id=lyDRBhUjhv](https://openreview.net/forum?id=lyDRBhUjhv)

---

## Table of Contents

- [Overview](#overview)
- [Relationship to Prior Work: Stochastic Linearization](#relationship-to-prior-work-stochastic-linearization)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Experiments](#experiments)
- [How PTPE Works: A Practical Summary](#how-ptpe-works-a-practical-summary)
- [Choosing the Expansion Order](#choosing-the-expansion-order)
- [Scaling Factor Optimization](#scaling-factor-optimization)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)

---

## Overview

### The Problem

When neural networks are deployed in safety-critical systems — pedestrian detection, power management, autonomous vehicles — it is essential to understand how input uncertainty propagates through the network layers. Given a multivariate Gaussian input distribution, **how does the output distribution look after passing through nonlinear activation functions?**

Existing approaches either rely on expensive Monte Carlo sampling (prone to undersampling in high dimensions) or use Jacobian linearization which degrades dramatically under moderate-to-high input uncertainty because it ignores higher-order moments.

### Our Solution: PTPE

PTPE introduces a **stochastic polynomial expansion** as a surrogate for the nonlinear activation functions. The central construct is a modified Taylor polynomial expansion where the polynomial terms act on **independent, identically distributed** Gaussian random variables:

$$g(X) = \mathbb{E}[f(X)] + \sum_{s=1}^{S} \frac{\mathbb{E}[\nabla_x^{\circ s} f(X)]}{s!} \circ \left(\Xi_{(s)}^{\circ s} - \mathbb{E}[\Xi_{(s)}^{\circ s}]\right)$$

This yields **closed-form expressions** for the output mean, covariance, and cross-covariance, enabling efficient uncertainty propagation through deep networks with:

- **O(n²) complexity** for full covariance propagation (comparable to Jacobian linearization)
- **O(n) complexity** when propagating only marginal variance
- **Significantly higher accuracy** than Jacobian linearization, especially in moderate-to-high uncertainty regimes

When taking only the first-order term (S=1), this expansion reduces to stochastic linearization. Higher-order terms capture curvature effects that linearization misses entirely.

### Supported Nonlinearities

PTPE provides analytical solutions for **seven** commonly used activation functions:

| Activation | Key Technique | Variance Parameter σ́² |
|---|---|---|
| **Tanh** | Approximated via linear combination of error functions | 1/(2γⱼ²) |
| **Sigmoid** | Approximated via linear combination of Gaussian CDFs | 1/(2γⱼ²) |
| **Softplus** | Derived from sigmoid (derivative of softplus = sigmoid) | 1/(2γⱼ²β²) |
| **ReLU** | Limiting case of softplus as β → ∞ | 0 |
| **LeakyReLU** | Superposition of two ReLU functions | 0 |
| **GELU** | Product of input and Gaussian CDF; Hermite recurrence | 1 |
| **SiLU (Swish)** | Product of input and sigmoid; combines sigmoid + GELU | 1/(2γⱼ²) |

All pseudo-Taylor coefficients Aₛ can be expressed in terms of three fundamental operations — Gaussian likelihood **L**, cumulative likelihood **F**, and an excess integral **I** — which are computationally cheap and fully vectorized.

---

## Relationship to Prior Work: Stochastic Linearization

PTPE is a **higher-order generalization** of the stochastic linearization approach for neural networks introduced in our earlier work:

> **Prior Paper:** *Estimating Uncertainty from Feed-Forward Network Based Sensing Using Quasi-Linear Approximation*  
> **Authors:** Songhan Zhang, Matthew Singh, Delsin Menolascino, ShiNung Ching  
> **Published in:** Neural Networks, Vol. 188, 107376, 2025  
> **DOI:** [10.1016/j.neunet.2025.107376](https://doi.org/10.1016/j.neunet.2025.107376)  
> **Repository:** [https://github.com/songhanz/Stochastic-Linearization-of-Neural-Nets](https://github.com/songhanz/Stochastic-Linearization-of-Neural-Nets)

That work applied **stochastic linearization** (also known as quasi-linear approximation) — a classical technique from the field of feedback control systems (Booton, 1953; Kazakov, 1954) — to neural network uncertainty propagation. Stochastic linearization replaces a nonlinearity with an equivalent linear gain computed from the *expected value* of the derivative over the input distribution, rather than the derivative evaluated at a single operating point (as in Jacobian linearization). This effectively corresponds to taking only the **first-order term** (S=1) of the PTPE expansion.

**What PTPE adds:** Higher-order polynomial terms (2nd, 3rd order and beyond) that capture the curvature and non-Gaussian effects of the nonlinear transformation. The result is dramatically improved accuracy in moderate-to-high uncertainty settings, while retaining the same computational complexity class as the first-order approach. In particular:

| Method | Complexity | Accuracy (high uncertainty) |
|---|---|---|
| Jacobian Linearization | O(n²) | Poor — ignores stochastic effects entirely |
| Stochastic Linearization (1st-order PTPE) | O(n²) | Moderate — uses expected derivative |
| **PTPE (3rd-order)** | **O(n²)** | **Excellent — captures curvature effects** |

---

## Repository Structure

```
Stochastic_Polynomial_Expansion/
│
├── cifar10experiment/          # ResNet uncertainty propagation benchmarks
│                                 Trains ResNets (13, 33, 65 layers) × (Tanh, ReLU, GELU)
│                                 Compares PTPE to Jacobian linearization and MC ground truth
│                                 → Reproduces Figures 3, 9, 10 from the paper
│
├── DVI+PTPE/                   # Deterministic Variational Inference with PTPE
│                                 UCI regression benchmarks on 8 datasets
│                                 Supports ReLU, GELU, and Tanh activations
│                                 → Reproduces Tables 2 and 3 from the paper
│
├── DVI+PTPE_categorical/       # DVI+PTPE for classification
│                                 MNIST digit classification
│                                 Bayesian neural network training with PTPE
│
├── VAE+PTPE/                   # Variational Autoencoder with PTPE decoder
│                                 MNIST reconstruction experiments
│                                 O(n) complexity (diagonal covariance only)
│                                 → Reproduces Figure 5 from the paper
│
├── rotation_mnist_ood/         # Out-of-distribution detection
│                                 Rotated MNIST (0°–180°) evaluation
│                                 FashionMNIST as OOD data
│                                 → Reproduces Figure 4 from the paper
│
└── LICENSE                     # MIT License
```

### Language Breakdown

The repository uses **Python** (56.2%) for the core neural network experiments, **MATLAB** (26.4%) for analytical derivations and scaling factor optimization, and **Jupyter Notebooks** (17.4%) for visualization and interactive exploration.

---

## Getting Started

### Prerequisites

**Python environment:**
- Python 3.8+
- PyTorch (1.10+ recommended)
- torchvision
- NumPy, SciPy, Matplotlib

**MATLAB** (optional, for scaling factor optimization):
- MATLAB R2021a+ with Optimization Toolbox (for `fmincon`)

### Installation

```bash
git clone https://github.com/songhanz/Stochastic_Polynomial_Expansion.git
cd Stochastic_Polynomial_Expansion
pip install torch torchvision numpy scipy matplotlib
```

---

## Experiments

### 1. CIFAR-10 Uncertainty Propagation (`cifar10experiment/`)

**Goal:** Benchmark PTPE accuracy for propagating Gaussian noise through pretrained ResNets.

**Setup:** Nine ResNets with three depths (13, 33, 65 layers) and three activations (Tanh, ReLU, GELU) are trained on CIFAR-10. Input images are corrupted with additive Gaussian noise at four variance levels ([1, 10, 100, 1000] on the RGB scale [0, 255]). The PTPE-predicted output distribution is compared against ground truth obtained via 10⁷ Monte Carlo samples.

**Metrics:** Euclidean distance of means (L2 norm), Frobenius norm of covariance residuals, 2-Wasserstein distance between predicted and reference Gaussian distributions.

**Key Finding:** Up to 3rd-order PTPE typically outperforms Jacobian linearization and stochastic linearization by a large margin, especially at moderate-to-high input variance.

```bash
cd cifar10experiment
# 1. Train ResNets on CIFAR-10 (or use provided pretrained weights)
# 2. Run PTPE propagation at different orders and input variances
# 3. Compare against 10⁷ MC sampling ground truth
```

### 2. Deterministic Variational Inference (`DVI+PTPE/`)

**Goal:** Replace the forward-passing functions in Deterministic Variational Inference (DVI) with PTPE, enabling non-piecewise-linear activations (Tanh, GELU) for BNN training. Previous DVI work (Wu et al., 2019) only supported piecewise-linear activations like ReLU.

**Setup:** Regression experiments on eight UCI datasets: Concrete Strength, Energy Efficiency, Kin8nm, Naval Propulsion, Power Plant, Protein Structure, Wine Quality, and Yacht Hydrodynamics. MLPs with up to four layers of 50 hidden units (100 for Protein Structure). 10% held-out test data with 20 random splits.

**Baselines:** MCVI, Probabilistic Backpropagation (PBP), Dropout, Deep Ensembles, DVI (ReLU only), Linearized Laplace.

**Key Finding:** PTPE-DVI achieves competitive or superior RMSE and log-likelihood across datasets, successfully extending DVI to smooth activations.

```bash
cd DVI+PTPE
# Run UCI regression across 8 datasets with 20 random train/test splits
# Reports RMSE and average log-likelihood
```

### 3. Categorical Classification (`DVI+PTPE_categorical/`)

**Goal:** Evaluate PTPE-based BNNs on classification tasks with MNIST digit classification.

```bash
cd DVI+PTPE_categorical
# Train BNN with PTPE on MNIST classification
```

### 4. VAE with PTPE Decoder (`VAE+PTPE/`)

**Goal:** Replace Monte Carlo sampling in the VAE decoder with deterministic PTPE moment propagation.

**Setup:** VAE trained on MNIST with latent dimensions of 2, 5, 10, and 20. The PTPE-VAE propagates only diagonal covariance (O(n) complexity) while keeping the architecture and trainable parameters identical to the vanilla VAE.

**Key Finding:** The PTPE-enabled VAE achieves higher ELBO and improved reconstruction accuracy compared to the vanilla VAE across all tested latent dimensions.

```bash
cd VAE+PTPE
# Train PTPE-VAE and vanilla VAE on MNIST
# Compare ELBO and reconstruction loss across latent dimensions (2, 5, 10, 20)
```

### 5. Out-of-Distribution Detection (`rotation_mnist_ood/`)

**Goal:** Test how DVI+PTPE models handle distribution shift and OOD inputs.

**Setup:** Models trained on MNIST are evaluated on progressively rotated MNIST images (0°–180°) and on FashionMNIST as entirely OOD data. Evaluated using Brier Score, Log Likelihood, confidence calibration, and entropy distributions.

**Key Finding:** DVI+PTPE achieves the highest accuracy on shifted images and is the least overconfident model among all tested approaches (Vanilla, Dropout, Ensemble), while maintaining comparable OOD detection capability.

```bash
cd rotation_mnist_ood
# Evaluate DVI+PTPE vs. Vanilla, Dropout, Ensemble on:
#   - Rotated MNIST (Brier Score, Log Likelihood)
#   - FashionMNIST OOD (Entropy, Confidence)
```

---

## How PTPE Works: A Practical Summary

### Core Workflow

For each nonlinear layer with activation function f(·) and Gaussian input X ~ N(μ̃, Σ̃):

**Step 1 — Compute Pseudo-Taylor Coefficients** *(O(n) per layer):*

Calculate A₀, A₁, A₂, A₃ using the closed-form expressions for your activation function (see Table 1 in the paper). These depend only on the marginal mean μ̃ and variance σ̃² = diag(Σ̃), not the full covariance matrix, so computation is O(n).

All coefficients are expressed in terms of three elementary building blocks:

- **L(μ̃; σ̂ⱼ)** = (1/σ̂ⱼ) · φ(μ̃/σ̂ⱼ) — Gaussian likelihood evaluated at the input mean
- **F(μ̃; σ̂ⱼ)** = Φ(μ̃/σ̂ⱼ) — Gaussian CDF evaluated at the input mean
- **I(μ̃; σ̂ⱼ)** = μ̃·F + σ̂²ⱼ·L — Integrated CDF (expected excess)

where σ̂²ⱼ = σ̃² + σ́²ⱼ combines the input variance with an activation-specific constant.

**Step 2 — Propagate Mean:**

The output mean is simply: **μ_out = A₀**

**Step 3 — Propagate Covariance** *(O(n²) per layer):*

For a 3rd-order expansion:

```
Σ_out = A₁ ∘ Σ̃ ∘ A₁ᵀ
      + A₂ ∘ (2·Σ̃∘²) ∘ A₂ᵀ
      + A₃ ∘ (6·Σ̃∘³ + 9·diag(Σ̃)·Σ̃·diag(Σ̃)ᵀ) ∘ A₃ᵀ
```

where ∘ denotes element-wise (Hadamard) operations.

**Step 4 — Handle Residual Connections:**

When combining X + g(Y) in a residual layer, compute the cross-covariance:

```
Σ_YZ = A₁ᵀ ∘ Σ_YX + 3·A₃ᵀ ∘ Σ_YX ∘ diag(Σ_X)ᵀ
```

Then the total covariance after addition: Σ_X + Σ_g(Y) + Σ_cross + Σ_crossᵀ

**Step 5 — Linear Layers:**

For X_out = Wᵀ·X + b:  μ̃_out = Wᵀ·μ + b  and  Σ̃_out = Wᵀ·Σ·W

**Repeat** Steps 1–5 for each layer in the network. See **Algorithm 1** in Appendix A.10 of the paper for complete pseudocode covering ResNet architectures with residual connections.

---

## Choosing the Expansion Order

| Input Uncertainty | Recommended Order | Notes |
|---|---|---|
| Low (σ² < 0.01 after z-scoring) | 1st or 2nd | Higher orders may introduce numerical instability |
| Moderate (0.01 ≤ σ² ≤ 0.1) | 2nd or 3rd | Best accuracy-stability tradeoff |
| High (σ² > 0.1) | 3rd | Large gains over lower-order methods |

**Practical recommendations:**

- When **batch normalization** is present, layer-wise input variances are typically small. Use 2nd-order PTPE to avoid numerical instability from higher-order terms.
- For **high input variance** (e.g., simulating noisy sensor inputs), 3rd-order PTPE provides significant accuracy improvements.
- There is **no significant difference** between using 4 vs. 8 scaling factors for the tanh/sigmoid approximation, so the default of 4 is recommended for efficiency.

---

## Scaling Factor Optimization

For tanh and sigmoid activations, PTPE approximates the activation using a linear combination of error functions (or Gaussian CDFs) with optimized scaling factors. The optimization is performed in MATLAB using `fmincon` and is available in the [Stochastic-Linearization-of-Neural-Nets](https://github.com/songhanz/Stochastic-Linearization-of-Neural-Nets) repository.

```matlab
% Default scaling factors used in this work:
% Tanh:    γ = [0.5583, 0.8596, 0.8596, 1.2612]   (4 terms)
% Sigmoid: γ = [0.2791, 0.4298, 0.4298, 0.6306]   (4 terms)
```

With 4 scaling factors, the maximum approximation error for tanh is on the order of 10⁻⁴. Additional terms can be added for higher accuracy without loss of parallelism, since each term in the linear combination is independent.

---

## Potential Applications

PTPE is a general-purpose tool for uncertainty propagation. Beyond the experiments demonstrated here, it can be integrated into:

- **Kalman Filtering** — propagate state estimates through learned neural network dynamics
- **Adversarial Training** — characterize output uncertainty under adversarial input perturbations
- **Variational Learning** — replace MC sampling with deterministic moment propagation in any variational framework
- **Safety-Critical Systems** — provide uncertainty certificates for neural network predictions in real-time
- **Sensor Fusion** — quantify how sensor noise affects downstream neural network decisions

---

## Citation

If you use this code in your research, please cite both works:

### PTPE (this repository):

```bibtex
@article{zhang2025stochastic_polynomial,
  title={A Stochastic Polynomial Expansion for Uncertainty Propagation through Networks},
  author={Zhang, Songhan and Ching, ShiNung},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=lyDRBhUjhv}
}
```

### Stochastic Linearization (foundational first-order method):

```bibtex
@article{zhang2025estimating,
  title={Estimating Uncertainty from Feed-Forward Network Based Sensing Using Quasi-Linear Approximation},
  author={Zhang, Songhan and Singh, Matthew and Menolascino, Delsin and Ching, ShiNung},
  journal={Neural Networks},
  volume={188},
  pages={107376},
  year={2025},
  doi={10.1016/j.neunet.2025.107376},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

