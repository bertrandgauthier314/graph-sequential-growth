# Code and data for *"Information-geometry-driven graph sequential growth"*

This repository contains the Python implementation, reproducible scripts and plotting routines for the experiments presented in the manuscript:

> ***Information-geometry-driven graph sequential growth***
>
> Harry T. Bond, Bertrand Gauthier, and Kirstin Strokorb
>
> Preprint: [arXiv:2601.22106](https://arxiv.org/abs/2601.22106)

## Table of contents

* [Overview](#overview)

* [Repository structure](#repository-structure)

* [Requirements and installation](#requirements-and-installation)

  * [Prerequisites](#prerequisites)

  * [Key dependencies](#key-dependencies)

* [Coordinate-descent solvers](#coordinate-descent-solvers)

  * [Input and output structure](#input-and-output-structure)

  * [Approximate full correction](#approximate-full-correction)

* [Thresholding and graphical lasso](#thresholding-and-graphical-lasso)

* [Diagonal ridge penalty](#diagonal-ridge-penalty)

* [Coordinate-descent solver usage example](#coordinate-descent-solver-usage-example)

## Overview

This repository implements a class of forward-search strategies for Gaussian graphical inference (structure learning) based on convex information-geometry-driven coordinate descents. Building on the correspondence between likelihood-optimal forward search and fully corrective coordinate descent, we implement numerically efficient relaxed search strategies guided by the local geometry of the likelihood landscape.

The ability of the proposed procedures to reliably extract sparse graphical models while limiting the number of false detections is demonstrated in a series of experiments. In line with the idea of stability selection, we also illustrate how the edge activation-rank distributions induced by Monte Carlo subsampling can support the identification of relevant edge sets.

## Repository structure

```
.
├── all_solvers.py           # Coordinate solvers, thresholding & GLasso wrappers
├── all_experiments.py       # Experimental workflow & figure generation routines
├── Nb1_494_bus.ipynb        # Setup and execution of the experiments on the 494_bus matrix (Figures 1-5)
├── Nb2_Full_Synthetic.ipynb # Setup and execution of the fully synthetic experiments (Figures 6 & 7)
├── Nb3_Riboflavin.ipynb     # Setup and execution of the experiments on the riboflavin data (Figures 8-10) 
├── Data/                    # Input data files (494_bus.mtx and riboflavin.csv) 
├── Figures/                 # Output folder for manuscript figures (PDFs)
└── README.md

```

### File breakdown

* **`all_solvers.py`**: Functions for coordinate-descent-driven graph recovery (`GS_solver`, `GSL_solver`, `BBI_solver`, `BFCI_solver`), naive thresholding extractions (`magnitude_xtr`), graphical lasso regularisation-path evaluation (`Glasso_ROC`), partial correlation conversions (`prec_to_pcorr`), and ROC/precision-recall metric calculations (`compute_ROC`).

* **`all_experiments.py`**: Functions performing experiments (`run_experiment_ROC`, `run_experiment_fig3`, etc.) and matplotlib rendering scripts for Figures 1–10.

* **Jupyter Notebooks**: Step-by-step reproduction notebooks pre-configured with manuscript seeds and parameters:

  * **`Nb1_494_bus.ipynb`**: experiments on the `494_bus` matrix (Section 6.1; Figures 1-5).

  * **`Nb2_Full_Synthetic.ipynb`**: evaluation across fully synthetic graph topologies (Section 6.2; Figures 6 & 7).

  * **`Nb3_Riboflavin.ipynb`**: experiments on the riboflavin data (Section 6.3; Figures 8-10).

## Requirements and installation

The code is developed for Python 3.8+ and relies on the standard scientific Python stack.

### Prerequisites

```bash
pip install numpy numba scipy matplotlib pandas networkx joblib tqdm gglasso cvxpy

```

### Key dependencies

| Package | Usage | 
| ----- | ----- | 
| `numba` | Just-In-Time (JIT) compilation (`@nb.njit(fastmath=True)`) for coordinate-solver loops. | 
| `joblib` | Multi-processing parallelisation across independent Monte Carlo trials. | 
| `gglasso` | ADMM solver for graphical lasso regularisation-path calculations. | 
| `cvxpy` | Semidefinite programming for matrix completion in the fully synthetic examples. | 
| `networkx` | Network layout computation and graph visualisation. | 

## Coordinate-descent solvers

The following coordinate-descent-based graph-recovery strategies are implemented.

* **`GS_solver`**: Gauss–Southwell selection rule (selects candidate edge with maximum absolute gradient).

* **`GSL_solver`**: Gauss–Southwell–Lipschitz selection rule (selects candidate edge using gradient and curvature information).

* **`BBI_solver`**: Best-Block-Improvement selection rule (selects candidate edge with maximum 2-block improvement).

* **`BFCI_solver`**: Best-Fully-Corrective-Improvement selection rule (evaluates full correction across all candidate edges to select an edge).

### Input and output structure

All the coordinate-descent solvers have the same input-output structure.

#### Inputs

* **`S`** (*ndarray*): Sample covariance matrix (optionally ridge-penalised).

* **`max_iter`** (*int*, default=`10`): Maximum number of outer iterations, i.e. edge activations ($k \le M=d(d-1)/2$).

* **`true_P`** (*ndarray*, optional, default=`None`): Ground-truth precision matrix. *Optional parameter used for evaluation only*; when provided, the function checks whether each selected coordinate $(i, j)$ corresponds to a true non-zero entry in the target precision matrix (`true_P[i, j] != 0`).

* **`activ_mul`** (*float*, default=`0.3`): Inner stopping multiplicative constant $\alpha$.

* **`activ_add`** (*int* or *float*, default=`10`): Inner stopping additive constant $\beta$.

* **`tau_improv`** (*float*, default=`1e-3`): Inner stopping improvement threshold $\tau$.

#### Outputs

Returns a **dictionary** containing the edge activation ranks, the number of inner iterations at each step, and, if `true_P` is provided, a list indicating whether activated edges are true or false.

* **`'evo_edge'`** (*list of tuples*): Ordered sequence of activated edge coordinate pairs `[(i_1, j_1), (i_2, j_2), ...]`.

* **`'edge_is_true'`** (*list of bool* or *None*): List indicating whether each activated edge exists in `true_P`. Evaluates to `None` if `true_P` is not provided.

* **`'evo_inner_up'`** (*list of int*): Number of inner refinement steps performed at each outer activation step (tracker for Figure 5).

### Approximate full correction

Inside each coordinate-descent solver, approximate full correction is performed via support-restricted GSL with $\{1,2\}$-block updates. The inner loop iteration budget and stopping conditions are controlled by:

* **`activ_mul` (**$\alpha$**, default=`0.3`)**: Multiplicative scaling factor for max inner-loop iterations ($l_{\text{max}} = \lceil \alpha k + \beta \rceil$, with $k$ the current number of active edges).

* **`activ_add` (**$\beta$**, default=`10`)**: Additive constant for max inner-loop iterations.

* **`tau_improv` (**$\tau$**, default=`1e-3`)**: Relative improvement threshold for early stopping during inner correction.

> **Computational note:** Numerically, and for similar approximate-full-correction parameters, computing a BFCI growth is significantly more demanding than computing a GS, GSL or BBI growth (GS, GSL and BBI involve one approximate full correction per activation step, while BFCI performs an approximate full correction for each candidate edge). Solver routines are single-threaded and C-compiled via Numba (`@nb.njit(fastmath=True)`).

## Thresholding and graphical lasso

The coordinate-descent solvers are compared to sample precision and partial correlation thresholding, and to graphical lasso.

* **`magnitude_xtr` (`Prec` / `PCorr`)**: Performs naive thresholding extractions by sorting the off-diagonal entries of sample precision ($\mathbf{S}^{-1}$) or sample partial correlation matrices in descending order of absolute magnitude.

* **`Glasso_ROC`**: Computes precision-recall and ROC curves across a discretised graphical lasso regularisation path $\lambda \in [0, \max_{i \neq j} \vert{}\mathbf{S}_{i,j}\vert{}]$ using ADMM ([Python package GGlasso](https://joss.theoj.org/papers/10.21105/joss.03865)).

  * ADMM optimisation convergence tolerance is set to $\text{tol} = 10^{-6}$.

  * Threshold for edge detection is set to $\vert{}\mathbf{Q}_{i,j}\vert{} > 10^{-5}$.

## Diagonal ridge penalty

To guarantee positive definiteness across all sample sizes (including $n \le d$), a diagonal ridge penalty is systematically applied to the empirical covariance matrix $\hat{\boldsymbol{\Sigma}}$; specifically:

$$
\mathbf{S} = \hat{\boldsymbol{\Sigma}} + \rho \cdot \text{mean}(\text{diag}(\hat{\boldsymbol{\Sigma}})) \, \mathbf{I}_d.
$$

**Default parameter**: `RP_coeff` ($\rho$) = $10^{-6}$.

## Coordinate-descent solver usage example

Run a quick graph recovery trial using `GSL_solver`:

```python
import numpy as np
import all_solvers as sol

# 1. Define precision and covariance matrices (d=5)
dim = 5
max_edges = dim * (dim - 1) // 2 # maximum edge capacity
true_P = np.eye(dim)
true_P[0, 1] = true_P[1, 0] = 0.4
true_P[0, 2] = true_P[2, 0] = 0.3
true_P[2, 3] = true_P[3, 2] = -0.4
true_cov = np.linalg.inv(true_P)

# 2. Sample data and compute ridge-penalised sample covariance
rng = np.random.default_rng(seed=42)
n_samp = 50
X = rng.multivariate_normal(mean=np.zeros(dim), cov=true_cov, size=n_samp)
Sigma_hat = (X.T @ X) / n_samp
S_test = Sigma_hat + 1e-6 * np.mean(np.diag(Sigma_hat)) * np.eye(dim)

# 3. Execute GSL solver (full growth, i.e. max_edges iterations)
results = sol.GSL_solver(
    S=S_test, 
    max_iter=max_edges, 
    true_P=true_P, 
    activ_mul=0.3, 
    activ_add=10, 
    tau_improv=1e-3
)

print("Activated Edge Indices:", results['evo_edge'])
print("True Edge Flags:", results['edge_is_true'])

```