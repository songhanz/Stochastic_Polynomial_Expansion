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

| Activation | Key Technique |
|---|---|
| **Tanh** | Approximated via linear combination of error functions |
| **Sigmoid** | Approximated via linear combination of Gaussian CDFs |
| **Softplus** | Derived from sigmoid (derivative of softplus = sigmoid) |
| **ReLU** | Limiting case of softplus as β → ∞ |
| **LeakyReLU** | Superposition of two ReLU functions |
| **GELU** | Product of input and Gaussian CDF; Hermite recurrence |
| **SiLU (Swish)** | Product of input and sigmoid; combines sigmoid + GELU |

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

**For DVI+PTPE experiments (TensorFlow-based):**
- Python 3.6+
- TensorFlow 1.15 (with `tensorflow.compat.v1`)
- TensorFlow Probability 0.5.0
- scikit-learn, pandas, numpy, scipy, matplotlib, tqdm, openpyxl, xlrd

A full list of pinned dependencies is provided in `DVI+PTPE/requirements.txt`.

**For VAE+PTPE experiments (PyTorch-based):**
- Python 3.8+
- PyTorch 1.10+
- torchvision, numpy, scipy, matplotlib, pandas

**For CIFAR-10 experiments (MATLAB):**
- MATLAB R2021a+ with Deep Learning Toolbox and GPU support
- Pretrained ResNet `.mat` files (see instructions below)

### Installation

```bash
git clone https://github.com/songhanz/Stochastic_Polynomial_Expansion.git
cd Stochastic_Polynomial_Expansion
```

For the DVI+PTPE experiments:
```bash
cd DVI+PTPE
pip install -r requirements.txt
```

For the VAE+PTPE experiments:
```bash
pip install torch torchvision numpy scipy matplotlib pandas
```

---

## Experiments

### 1. CIFAR-10 Uncertainty Propagation (`cifar10experiment/`)

**Goal:** Benchmark PTPE accuracy for propagating Gaussian noise through pretrained ResNets.

**Setup:** Nine ResNets with three depths (ResNet-13, ResNet-33, ResNet-65, specified as stack configurations `[3,2,1]`, `[3,10,3]`, `[4,25,3]`) and three activations (Tanh, ReLU, GELU) are trained on CIFAR-10. Input images are corrupted with additive i.i.d. Gaussian noise at four variance levels ([1, 10, 100, 1000] on the RGB scale [0, 255]).

**Requires:** MATLAB with Deep Learning Toolbox and a GPU. Pretrained ResNet models must be saved as `.mat` files in a `models/` directory (e.g., `resnet321_tanh.mat`). Convolution weights must be pre-flattened into matrix form.

**Step 1 — Flatten convolution weights** (run once per model):
```matlab
% In MATLAB:
flatten_resnet   % Loads each ResNet .mat file and saves reshaped convolution weights
                 % Output: convW/<netname><nonlin>_convW.mat
```

**Step 2 — Compute MC ground truth** (10⁶ samples per configuration):
```matlab
% Run one script per activation:
true_output_dist_Tanh   % MC sampling through Tanh ResNets at all depths & variances
true_output_dist_ReLU   % MC sampling through ReLU ResNets
true_output_dist_GELU   % MC sampling through GELU ResNets
```

**Step 3 — Run PTPE estimation** (1st, 2nd, 3rd order):
```matlab
% Run one script per activation:
est_output_PTPE_3orders_resnet_tanh   % PTPE propagation through Tanh ResNets
est_output_PTPE_3orders_resnet_relu   % PTPE propagation through ReLU ResNets
est_output_PTPE_3orders_resnet_gelu   % PTPE propagation through GELU ResNets

% Optional: test with 8 scaling factors instead of 4
est_output_PTPE_3orders_resnet_tanh_8gamma
```

**Step 4 — Generate comparison figures** (Figures 3, 9, 10):
```matlab
compare_PTPE_JL              % Side-by-side comparison plots
compare_PTPE_JL_werrorbar    % Same, with error bars from repeated trials
```

**Key files:**
| File | Purpose |
|---|---|
| `flatten_resnet.m` | Reshapes convolution weights into matrix form for PTPE |
| `true_output_dist_*.m` | Monte Carlo ground truth (10⁶ samples) |
| `est_output_PTPE_3orders_resnet_*.m` | PTPE estimation at orders 1–3 |
| `compare_PTPE_JL.m` | Generates comparison figures (W2, L2, Frobenius) |
| `nearestSPD.m` | Utility: projects matrices to nearest symmetric positive definite |
| `hex2rgb.m` | Utility: color conversion for plots |

### 2. Deterministic Variational Inference (`DVI+PTPE/`)

**Goal:** Replace the moment propagation in DVI (Wu et al., 2019) with PTPE, enabling non-piecewise-linear activations (Tanh, GELU) for Bayesian neural network training.

**Setup:** Regression on eight UCI datasets (Concrete, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht). MLPs with up to 4 hidden layers of 50 units (100 for Protein Structure). Trains for 2000 epochs with early stopping (patience=20). Uses 10% test split with 20 random seeds. Datasets are downloaded automatically from UCI/OpenML.

**Run experiments** (one script per activation):
```bash
cd DVI+PTPE
python UCI_relu.py    # DVI+PTPE with ReLU activation
python UCI_gelu.py    # DVI+PTPE with GELU activation
python UCI_tanh.py    # DVI+PTPE with Tanh activation
```

Each script iterates over all 8 datasets and saves results to `UCI_results_sgd/<dataset>_<activation>_<layers>layer_adapter/`. Per-run checkpoints and training histories are stored in numbered subdirectories.

**Print aggregated results** (Tables 2 & 3):
```bash
python print_result.py    # Reads all result folders and prints RMSE ± stderr
                          # and log-likelihood ± stderr for each dataset/activation/depth
```

**Interactive exploration:**

Open `ToyData.ipynb` in Jupyter to visualize PTPE-DVI on a 1D toy regression problem.

**Key files:**
| File | Purpose |
|---|---|
| `UCI_relu.py`, `UCI_gelu.py`, `UCI_tanh.py` | Main training scripts (one per activation) |
| `print_result.py` | Aggregates and prints test RMSE & log-likelihood |
| `ToyData.ipynb` | Interactive 1D toy data demo |
| `bayes_layers.py` | PTPE moment propagation for ReLU, GELU, Tanh |
| `bayes_models.py` | BNN model definitions (MLP, AdaptedMLP) |
| `gaussian_variables.py` | Gaussian distribution utilities |
| `utils.py` | Training loop, optimizer, data utilities |

### 3. Categorical Classification & OOD Preparation (`DVI+PTPE_categorical/`)

**Goal:** Train PTPE-based BNNs on MNIST digit classification and generate predictions for OOD experiments.

**Step 1 — Train DVI+PTPE on MNIST:**
```bash
cd DVI+PTPE_categorical
python mnist.py                # Trains BNN with PTPE on MNIST
                               # Saves model to mnist_dvi_model/
                               # Saves training history to mnist_dvi_results/
```

**Step 2 — (Optional) Hyperparameter search and cyclic annealing:**
```bash
python mnist_search_lambda.py  # Grid search over KL weighting parameter λ
python mnist_cyclic_anneal.py  # Train with cyclic KL annealing schedule
```

**Step 3 — Run OOD evaluation with the trained DVI model:**
```bash
python rotation_ood.py         # Evaluates DVI+PTPE on rotated MNIST & FashionMNIST OOD
                               # Saves results as .pkl files in mnist_dvi_results/
```

**Key files:**
| File | Purpose |
|---|---|
| `mnist.py` | Main MNIST training script for DVI+PTPE |
| `mnist_search_lambda.py` | Hyperparameter search over λ |
| `mnist_cyclic_anneal.py` | Cyclic KL annealing variant |
| `rotation_ood.py` | Generates DVI rotation/OOD prediction results (.pkl) |

### 4. VAE with PTPE Decoder (`VAE+PTPE/`)

**Goal:** Replace Monte Carlo sampling in the VAE decoder with deterministic PTPE moment propagation.

**Setup:** VAE trained on MNIST with latent dimensions of 2, 5, 10, and 20. The script trains both a vanilla VAE and a PTPE-enabled VAE (`VAE_EP`) side by side for 200 epochs, then saves ELBO curves as CSV and PNG. Architecture: encoder and decoder with 500 hidden units, batch size 100.

**Note:** MNIST data must be available at `./data/MNIST/`. Set `download=True` in `train_mnist.py` on first run.

**Step 1 — Train both vanilla VAE and PTPE-VAE:**
```bash
cd VAE+PTPE
python train_mnist.py    # Trains VAE (vanilla) and VAE_EP (PTPE) for each latent dim
                         # Outputs:
                         #   trained_parameters/vanilla_v2/mnist_zdim{N}.pkl
                         #   trained_parameters/EP_v2/mnist_zdim{N}.pkl
                         #   elbo_data_{N}D_vanilla.csv   (ELBO curves)
                         #   elbocurve-{N}D.png           (ELBO plots)
```

To train across all latent dimensions, edit the `for latent_size in [2]:` loop in `train_mnist.py` to `[2, 5, 10, 20]`.

**Step 2 — Evaluate reconstruction error on test set:**
```bash
python reconst_err_vanilla.py   # Reconstruction MSE for vanilla VAE (all latent dims)
python reconst_err_PTPE.py      # Reconstruction MSE for PTPE-VAE (all latent dims)
```

**Step 3 — Generate paper figures** (Figure 5):
```matlab
% In MATLAB:
make_figures    % Reads the CSV files and produces publication-quality ELBO plots
```

**Key files:**
| File | Purpose |
|---|---|
| `train_mnist.py` | Trains vanilla VAE and PTPE-VAE side by side |
| `reconst_err_vanilla.py` | Evaluates reconstruction error for vanilla VAE |
| `reconst_err_PTPE.py` | Evaluates reconstruction error for PTPE-VAE |
| `models.py` | Model definitions: `VAE`, `VAE_EP`, `VAE_EP_fullcov` |
| `make_figures.m` | MATLAB script for publication-quality ELBO plots |

### 5. Out-of-Distribution Detection (`rotation_mnist_ood/`)

**Goal:** Compare DVI+PTPE against Vanilla, Dropout, and Ensemble models on distribution-shifted and OOD data.

**Prerequisites:** You must first train the DVI+PTPE model on MNIST and generate its prediction results:
```bash
# First, run Steps 1 and 3 in DVI+PTPE_categorical/ above:
cd DVI+PTPE_categorical
python mnist.py          # Train the DVI+PTPE model
python rotation_ood.py   # Generate DVI rotation & OOD predictions (.pkl files)

# Then copy the results folder into rotation_mnist_ood/:
cp -r mnist_dvi_results/ ../rotation_mnist_ood/mnist_dvi_results/
```

**Run the full comparison:**
```bash
cd rotation_mnist_ood
python main.py
```

This script will:
1. Train (or load previously saved) Vanilla, Dropout (p=0.5), and Ensemble (10 models) baselines on MNIST
2. Evaluate all models on rotated MNIST images at 12 angles (0°, 15°, 30°, ... 180°)
3. Evaluate all models on FashionMNIST as OOD data
4. Load the DVI+PTPE results from `mnist_dvi_results/`
5. Generate Figure 4: a 2×3 panel with Brier Score, Log Likelihood, Accuracy vs. Confidence, Count vs. Confidence, Entropy Distribution, and OOD Confidence

**Output:** `results/uncertainty_estimation_results.pdf` and `.png`

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

## Acknowledgement

This research was, in part, funded by the U.S. Government (award no. HR00112290113). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government.
