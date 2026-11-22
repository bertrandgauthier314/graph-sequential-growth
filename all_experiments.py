"""
Experimental Workflows and Visualization Functions for Manuscript Figures

This module gathers the parallelized Monte Carlo experimental pipelines and  
plotting routines used to generate Figures 1 through 10 of the manuscript.

Key Components
--------------
1. Core Benchmarking Infrastructure:
   - full_ROC_comp        : Evaluates single-realization ROC and precision-recall (PR) curves.
   - run_experiment_ROC   : Parallelised Monte Carlo benchmark across various sample sizes.

2. Synthetic Benchmark Experiments & Figures:
   - Figure 1             : Sparsity pattern panel alongside 2x3 ROC/PR grids for GSL, BBI, and BFCI.
   - Figure 2             : 2x3 ROC/PR grid comparing GSL, GLasso, PCorr, and Prec.
   - Figure 3 (run/plot)  : True edge detection frequency and GSL false positive heatmap.
   - Figure 4 (run/plot)  : Activation rank boxplots for foremost and remaining true edges under full growth.
   - Figure 5 (run/plot)  : Inner iteration dynamics and recall evolution across stopping thresholds tau.
   - Figures 6 & 7        : 4x4 PR curve grid across 4 network topologies (Random m=200, Random m=40, 
                            Clique m=225, Hub m=45) and various sample sizes.

3. Empirical Data Experiments (Riboflavin / Gene Expression):
   - Figure 8 (run/plot)  : Rank stability boxplots for repeated subsample-based extractions.
   - Figures 9 & 10       : Concentric "Galaxy" layout network visualizations for various edge counts,  
                            presenting subsampling median activation ranks (Fig 9) with full dataset growth (Fig 10).
"""



from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import networkx as nx

import all_solvers as sol

import warnings

## Suppress harmless intermediate warnings from Matplotlib's compressed layout
## when fitting square-aspect subplots with LaTeX labels (the final layout renders properly).
warnings.filterwarnings(
    "ignore",
    message="constrained_layout not applied",
    category=UserWarning,
)

## Shared colour theme for plots
color_map_glob = {
    'GSL': 'firebrick',
    'Glasso': 'darkorange',
    'PCorr': 'forestgreen',
    'Prec': 'royalblue',
    'BBI': 'hotpink',
    'BFCI': 'darkgoldenrod', 
    'GS': 'darkgrey'
}

## Figure plot style (use LaTeX amsmath or stix)
def _init_plot_style(use_stix=False):

    if use_stix:
        mpl.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
    else:
        mpl.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })


## Compute full ROC and precision-recall curve for various methods
def full_ROC_comp(S,                  # sample covariance 
                  true_P,             # true precision 
                  methods,            # list of methods: GS, GSL, BBI, BFCI, Prec, PCorr, Glasso 
                  activ_mul=0.3,      # multiplicative cst for inner loop (FC)
                  activ_add=10,       # additive cst for inner loop (FC)
                  tau_improv=1e-3,    # treshold for inner loop (FC)):
                  n_glasso=500):      # number of glasso eval. 

    dim = S.shape[0]
    M_edge = int((dim-1) * dim / 2)
    nb_true_edge, true_edge = sol.compute_number_edges(true_P)
    true_ED = np.array(true_edge)
    
    prec_test = np.linalg.inv(S)                  # sample precision, i.e. inverse of S            
    pcorr_test = sol.prec_to_pcorr(prec_test)     # sample pcorr

    GS_ROC = None ; GSL_ROC = None ; BBI_ROC = None ; BFCI_ROC = None
    Prec_ROC = None ; PCorr_ROC = None ; Glasso_ROC = None
    
    # GS
    if 'GS' in methods: 
        GS = sol.GS_solver(S, M_edge, true_P, activ_mul, activ_add, tau_improv)
        GS_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=GS)
        GS_ROC = (np.concatenate(([100], GS_ROC[0])),
                   np.concatenate(([0], GS_ROC[1])),
                   np.concatenate(([0], GS_ROC[2])) )
    
    # GSL
    if 'GSL' in methods: 
        GSL = sol.GSL_solver(S, M_edge, true_P, activ_mul, activ_add, tau_improv)
        GSL_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=GSL)
        GSL_ROC = (np.concatenate(([100], GSL_ROC[0])),
                   np.concatenate(([0], GSL_ROC[1])),
                   np.concatenate(([0], GSL_ROC[2])) )
    # BBI
    if 'BBI' in methods: 
        BBI = sol.BBI_solver(S, M_edge, true_P, activ_mul, activ_add, tau_improv)
        BBI_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=BBI)
        BBI_ROC = (np.concatenate(([100], BBI_ROC[0])),
                   np.concatenate(([0], BBI_ROC[1])),
                   np.concatenate(([0], BBI_ROC[2])) )
    # BFCI
    if 'BFCI' in methods: 
        BFCI = sol.BFCI_solver(S, M_edge, true_P, activ_mul, activ_add, tau_improv)
        BFCI_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=BFCI) 
        BFCI_ROC = (np.concatenate(([100], BFCI_ROC[0])),
                    np.concatenate(([0], BFCI_ROC[1])),
                    np.concatenate(([0], BFCI_ROC[2])) )
    # Prec
    if 'Prec' in methods: 
        Prec = sol.magnitude_xtr(prec_test, M_edge, true_P) 
        Prec_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=Prec) 
        Prec_ROC = (np.concatenate(([100], Prec_ROC[0])),
                    np.concatenate(([0], Prec_ROC[1])),
                    np.concatenate(([0], Prec_ROC[2])) )
    # PCorr
    if 'PCorr' in methods: 
        PCorr = sol.magnitude_xtr(pcorr_test, M_edge, true_P) 
        PCorr_ROC = sol.compute_ROC(nb_true_edge, M_edge, Run=PCorr)
        PCorr_ROC = (np.concatenate(([100], PCorr_ROC[0])),
                     np.concatenate(([0], PCorr_ROC[1])),
                     np.concatenate(([0], PCorr_ROC[2])) )
    # Glasso
    if 'Glasso' in methods: 
        Glasso_ROC = sol.Glasso_ROC(S, true_P, n_glasso)
        Glasso_ROC = (np.concatenate((Glasso_ROC[0],[(nb_true_edge/M_edge)*100])),
                      np.concatenate((Glasso_ROC[1],[100])),
                      np.concatenate((Glasso_ROC[2],[100])) )

    return {'GS' : GS_ROC, 
            'GSL' : GSL_ROC, 
            'BBI' : BBI_ROC, 
            'BFCI' : BFCI_ROC, 
            'Prec' : Prec_ROC, 
            'PCorr' : PCorr_ROC, 
            'Glasso' : Glasso_ROC }




#######################
#######################
## General ROC-based Monte Carlo experiment (parallelised)
def _worker_ROC_rep(r, n_val, true_cov, true_P, methods, n_reps_BFCI, 
                    activ_mul, activ_add, tau_improv, n_glasso, 
                    RP_coeff, seed, dim, mean_vector):
    """Worker process executing a single Monte Carlo repetition for ROC/PR evaluation."""
    # Unique deterministic seed for (n_val, r)
    rep_seed = seed + 1000 * n_val + r
    rng = np.random.default_rng(seed=rep_seed)

    # 1. Generate dataset realization
    realisations = rng.multivariate_normal(mean_vector, true_cov, size=n_val, method='cholesky')

    # 2. Compute sample covariance with ridge penalty
    Sigma_hat = (realisations.T @ realisations) / n_val
    RP_val = np.mean(np.diag(Sigma_hat)) * RP_coeff
    S_test = Sigma_hat + RP_val * np.eye(dim)

    # 3. Restrict BFCI execution to the first n_reps_BFCI repetitions
    current_methods = [m for m in methods if m != 'BFCI' or r < n_reps_BFCI]

    # 4. Execute active solvers
    run_out = full_ROC_comp(
        S=S_test, 
        true_P=true_P, 
        methods=current_methods, 
        activ_mul=activ_mul,
        activ_add=activ_add,
        tau_improv=tau_improv,
        n_glasso=n_glasso
    )
    
    return current_methods, run_out

def run_experiment_ROC(true_cov, 
                       true_P, 
                       n_samples_list=[30, 90, 160], 
                       n_reps=100, 
                       n_reps_BFCI=10,
                       methods=['Prec', 'PCorr', 'Glasso', 'BFCI', 'BBI', 'GSL'], 
                       activ_mul=0.3,
                       activ_add=10,
                       tau_improv=1e-3,
                       n_glasso=200, 
                       RP_coeff=1e-6, 
                       n_jobs=-1,
                       seed=1):
    """
    Executes a unified ROC and Precision-Recall experiment across multiple solvers,
    sample sizes, and Monte Carlo repetitions in parallel using joblib.

    Parameters
    ----------
    true_cov : ndarray of shape (d, d)
        True population covariance matrix.
    true_P : ndarray of shape (d, d)
        True underlying precision matrix.
    n_samples_list : list of int, default=[30, 90, 160]
        Sample sizes to evaluate.
    n_reps : int, default=100
        Total Monte Carlo repetitions per sample size.
    n_reps_BFCI : int, default=10
        Restricted repetition count for the computationally heavy BFCI solver.
    methods : list of str, default=['Prec', 'PCorr', 'Glasso', 'BFCI', 'BBI', 'GSL']
        Solvers to evaluate. Any subset of these 6 solvers is supported.
    activ_mul : float, default=0.3
        Multiplicative constant for FC inner loop active iterations.
    activ_add : int, default=10
        Additive constant for FC inner loop active iterations.
    tau_improv : float, default=1e-3
        Objective improvement threshold for FC inner loop termination.
    n_glasso : int, default=200
        Discretization parameter (number of lambda steps) for Graphical Lasso path.
    RP_coeff : float, default=1e-6
        Ridge penalty coefficient applied to sample covariance.
    n_jobs : int, default=-1
        Number of parallel CPU workers (-1 uses all available logical cores).
    seed : int, default=1
        Base random seed for reproducible sampling.

    Returns
    -------
    results : dict
        Nested dictionary of the form:
        results[n_val][method] = {'prec': [...], 'rec': [...], 'fpr': [...]}
    """
    dim = true_cov.shape[0]
    mean_vector = np.zeros(dim)

    results = {
        n_val: {m: {'prec': [], 'rec': [], 'fpr': []} for m in methods}
        for n_val in n_samples_list
    }

    # Include BFCI repetition count in print output only if BFCI is in methods
    bfci_str = f" (BFCI reps={n_reps_BFCI})" if 'BFCI' in methods else ""
    print(f"Starting parallel ROC experiment: d={dim}, reps={n_reps}{bfci_str}, \n"
          f"methods={methods}, n_jobs={n_jobs}")

    for n_val in n_samples_list:
        rep_outputs = Parallel(n_jobs=n_jobs)(
            delayed(_worker_ROC_rep)(
                r, n_val, true_cov, true_P, methods, n_reps_BFCI, 
                activ_mul, activ_add, tau_improv, n_glasso, RP_coeff, seed, dim, mean_vector
            )
            for r in tqdm(range(n_reps), desc=f"Sample size n = {n_val}", mininterval=2.0)
        )

        for active_methods, run_out in rep_outputs:
            for m in active_methods:
                if run_out[m] is not None:
                    prec, rec, fpr = run_out[m]
                    results[n_val][m]['prec'].append(prec)
                    results[n_val][m]['rec'].append(rec)
                    results[n_val][m]['fpr'].append(fpr)

    return results
    
#######################
#######################
## Figure 1
def plot_figure_1(results, 
                  true_P,
                  n_samples_list=[30, 90, 160], 
                  font_scale=1.2, 
                  scale_x=1.0, 
                  use_stix=False,
                  color_map=None,
                  save_path=None):
    """
    Plots Figure 1: Sparsity pattern panel on the left (vertically centered across rows),
    alongside a 2x3 grid of ROC/PR curves comparing GSL, BBI, and BFCI across sample sizes.

    Parameters
    ----------
    results : dict
        Output dictionary from run_experiment_fig1.
    true_P : ndarray of shape (d, d)
        True precision matrix for rendering the sparsity pattern.
    n_samples_list : list of int, default=[30, 90, 160]
        Sample sizes corresponding to columns.
    font_scale : float, default=1.2
        Font scaling factor.
    scale_x : float, default=1.0
        Horizontal figure scaling factor.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath fonts.
    color_map : dict, optional
        Custom method color mappings.
    save_path : str or Path, optional
        Path to save vector PDF output.
    """
    if color_map is None:
        color_map = color_map_glob

    # Typography configuration
    _init_plot_style(use_stix=use_stix)

    title_fs  = 13 * font_scale
    label_fs  = 11 * font_scale
    tick_fs   = 9.5 * font_scale
    legend_fs = 9.5 * font_scale

    num_cols = len(n_samples_list)
    fig = plt.figure(figsize=(10.5 * scale_x, 5.6), layout='compressed')
    
    # GridSpec with equal width ratios for all 4 columns (1 sparsity + 3 curve columns)
    gs = GridSpec(2, num_cols + 1, figure=fig, width_ratios=[1.0] * (num_cols + 1))

    # --- LEFT PANEL: Precision Matrix Sparsity Pattern ---
    # Spans gs[:, 0] (both rows) to center vertically alongside the 2x3 curve grid
    ax_sparse = fig.add_subplot(gs[:, 0])
    ax_sparse.spy(true_P, markersize=1.2, color='black')
    ax_sparse.set_title(
        r"$\mathbf{Sparsity\ pattern\ of\ \Theta}$" if use_stix else r"$\bm{\text{Sparsity pattern of }\Theta}$", 
        fontsize=title_fs, pad=8
    )
    ax_sparse.set_xticks([])
    ax_sparse.set_yticks([])
    ax_sparse.set_box_aspect(1)

    ticks = [0, 25, 50, 75, 100]

    # --- RIGHT PANELS: ROC and PR Curves ---
    axes_roc = []
    axes_pr = []

    for col_idx, n_val in enumerate(n_samples_list):
        ax_roc = fig.add_subplot(gs[0, col_idx + 1])
        ax_pr  = fig.add_subplot(gs[1, col_idx + 1])
        
        axes_roc.append(ax_roc)
        axes_pr.append(ax_pr)

        ax_roc.set_title(
            rf"$\mathbf{{n = {n_val}}}$" if use_stix else rf"$\bm{{n = {n_val}}}$", 
            fontsize=title_fs, pad=6
        )
        ax_roc.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1, zorder=1)
        
        roc_handles = {}

        for method, data in results[n_val].items():
            if len(data['prec']) == 0:
                continue
                
            color = color_map.get(method, 'black')
            
            prec_arr = np.array(data['prec'])
            rec_arr  = np.array(data['rec'])
            fpr_arr  = np.array(data['fpr'])
            
            med_prec = np.median(prec_arr, axis=0)
            med_rec  = np.median(rec_arr, axis=0)
            med_fpr  = np.median(fpr_arr, axis=0)
            
            p10_rec  = np.percentile(rec_arr, 10, axis=0)
            p90_rec  = np.percentile(rec_arr, 90, axis=0)

            # TOP ROW: ROC
            line_roc, = ax_roc.plot(
                med_fpr, med_rec, color=color, linewidth=2, 
                clip_on=False, zorder=3
            )
            ax_roc.fill_between(
                med_fpr, p10_rec, p90_rec, color=color, alpha=0.18, 
                clip_on=False, zorder=2
            )
            roc_handles[method] = line_roc

            # BOTTOM ROW: PR
            ax_pr.plot(
                med_prec, med_rec, color=color, linewidth=2, 
                clip_on=False, zorder=3
            )
            ax_pr.fill_between(
                med_prec, p10_rec, p90_rec, color=color, alpha=0.18, 
                clip_on=False, zorder=2
            )

        # Formatting Subplots
        for ax in (ax_roc, ax_pr):
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_box_aspect(1)
            ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        ax_roc.set_xlabel(r"FP rate ($\%$)", fontsize=label_fs, labelpad=2)
        ax_pr.set_xlabel(r"Precision ($\%$)", fontsize=label_fs, labelpad=2)

        if col_idx == 0:
            ax_roc.set_ylabel(r"Recall ($\%$)", fontsize=label_fs)
            ax_pr.set_ylabel(r"Recall ($\%$)", fontsize=label_fs)
        else:
            ax_roc.tick_params(labelleft=False)
            ax_pr.tick_params(labelleft=False)

        # Legend on the first ROC plot showing all active methods
        if col_idx == 0 and roc_handles:
            ax_roc.legend(
                list(roc_handles.values()), list(roc_handles.keys()), 
                loc="lower right", frameon=True, edgecolor='black', 
                facecolor='white', framealpha=1.0, fontsize=legend_fs
            )

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 1 PDF to: {save_path}")

    plt.show()
    return fig

#######################
#######################
## Figure 2
def plot_figure_2(results, 
                  n_samples_list=[30, 90, 160], 
                  font_scale=1.2, 
                  scale_x=1.0, 
                  use_stix=False,
                  color_map=None,
                  save_path=None):
    r"""
    Plots Figure 2 (2x3 grid: ROC top row, Precision-Recall bottom row) using
    the output dictionary from run_experiment_fig2, formatted for publication.

    Parameters
    ----------
    results : dict
        Nested dictionary of the form:
        results[n_val][method] = {'prec': [...], 'rec': [...], 'fpr': [...]}
    n_samples_list : list of int, default=[30, 90, 160]
        Sample sizes corresponding to figure columns.
    font_scale : float, default=1.2
        Scaling multiplier to tune overall font sizes.
    scale_x : float, default=1.0
        Horizontal scaling factor for figure window width.
    use_stix : bool, default=True
        If True, uses built-in STIX typography. 
        If False, uses full LaTeX rendering with \usepackage{amsmath} and Computer Modern fonts.
    color_map : dict, optional
        Mapping of method names to Matplotlib color strings.
        Defaults to paper palette: {'GSL': 'firebrick', 'Glasso': 'darkorange', 
                                    'PCorr': 'forestgreen', 'Prec': 'royalblue'}.
    save_path : str or Path, optional
        Destination filepath to save the high-resolution vector PDF (e.g. 'Figure_2.pdf').
    """
    # Default color palette if none provided
    if color_map is None:
        color_map = color_map_glob

    # Configure Matplotlib typography based on use_stix option
    _init_plot_style(use_stix=use_stix)
    
    # Dynamic typography hierarchy
    title_fs  = 13 * font_scale
    label_fs  = 11 * font_scale
    tick_fs   = 9.5 * font_scale
    legend_fs = 9.5 * font_scale

    # Compressed layout automatically eliminates empty margins between square subplots
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=len(n_samples_list), 
        figsize=(8.2 * scale_x, 5.6),
        layout='compressed'
    )
    
    # Ultra-compact padding between subplots
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.03)

    # Ticks at exact 0-25-50-75-100 positions
    ticks = [0, 25, 50, 75, 100]

    for col_idx, n_val in enumerate(n_samples_list):
        ax_roc = axes[0, col_idx]
        ax_pr  = axes[1, col_idx]
        
        # Bold sample size titles in LaTeX
        ax_roc.set_title(rf"$\bm{{n = {n_val}}}$" if not use_stix else rf"$\mathbf{{n = {n_val}}}$", fontsize=title_fs, pad=6)
        
        # Reference diagonal line for ROC
        ax_roc.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1, zorder=1)
        
        roc_handles = {}

        for method, data in results[n_val].items():
            if len(data['prec']) == 0:
                continue
                
            color = color_map.get(method, 'black')
            
            # Convert metric lists to 2D numpy arrays
            prec_arr = np.array(data['prec'])
            rec_arr  = np.array(data['rec'])
            fpr_arr  = np.array(data['fpr'])
            
            # Pointwise Medians across repetitions
            med_prec = np.median(prec_arr, axis=0)
            med_rec  = np.median(rec_arr, axis=0)
            med_fpr  = np.median(fpr_arr, axis=0)
            
            # Pointwise 10th and 90th percentiles (Interdecile range)
            p10_rec  = np.percentile(rec_arr, 10, axis=0)
            p90_rec  = np.percentile(rec_arr, 90, axis=0)

            # --- TOP ROW: ROC CURVE ---
            line_roc, = ax_roc.plot(
                med_fpr, med_rec, color=color, linewidth=2, 
                clip_on=False, zorder=3
            )
            ax_roc.fill_between(
                med_fpr, p10_rec, p90_rec, color=color, alpha=0.18, 
                clip_on=False, zorder=2
            )
            roc_handles[method] = line_roc

            # --- BOTTOM ROW: PRECISION-RECALL CURVE ---
            ax_pr.plot(
                med_prec, med_rec, color=color, linewidth=2, 
                clip_on=False, zorder=3
            )
            ax_pr.fill_between(
                med_prec, p10_rec, p90_rec, color=color, alpha=0.18, 
                clip_on=False, zorder=2
            )

        # Formatting Subplots & Ticks
        for ax in (ax_roc, ax_pr):
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_box_aspect(1)
            ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        # X-Axis Labels in LaTeX
        ax_roc.set_xlabel(r"FP rate ($\%$)", fontsize=label_fs, labelpad=2)
        ax_pr.set_xlabel(r"Precision ($\%$)", fontsize=label_fs, labelpad=2)

        # Shared Y-Axis Logic: Y-label and numbers appear ONLY on Column 0
        if col_idx == 0:
            ax_roc.set_ylabel(r"Recall ($\%$)", fontsize=label_fs)
            ax_pr.set_ylabel(r"Recall ($\%$)", fontsize=label_fs)
        else:
            ax_roc.tick_params(labelleft=False)
            ax_pr.tick_params(labelleft=False)

        # Split Legends
        if col_idx == 0 and 'GSL' in roc_handles:
            ax_roc.legend(
                [roc_handles['GSL']], ['GSL'], 
                loc="lower right", frameon=True, edgecolor='black', 
                facecolor='white', framealpha=1.0, fontsize=legend_fs
            )
        elif col_idx == 1 and 'Glasso' in roc_handles:
            ax_roc.legend(
                [roc_handles['Glasso']], ['GLasso'], 
                loc="lower right", frameon=True, edgecolor='black', 
                facecolor='white', framealpha=1.0, fontsize=legend_fs
            )
        elif col_idx == 2:
            handles, labels = [], []
            if 'PCorr' in roc_handles:
                handles.append(roc_handles['PCorr'])
                labels.append('PCorr')
            if 'Prec' in roc_handles:
                handles.append(roc_handles['Prec'])
                labels.append('Prec')
            if handles:
                ax_roc.legend(
                    handles, labels, 
                    loc="lower right", frameon=True, edgecolor='black', 
                    facecolor='white', framealpha=1.0, fontsize=legend_fs
                )

    # Save vector PDF directly before calling plt.show()
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 2 PDF to: {save_path}")

    plt.show()
    return fig


#######################
#######################
## Figure 3
def run_experiment_fig3(true_cov, 
                        true_P, 
                        n_samples=30, 
                        k_edges=30, 
                        n_reps=100, 
                        methods=['GSL', 'PCorr', 'Prec'], 
                        activ_mul=0.3, 
                        activ_add=10, 
                        tau_improv=1e-3, 
                        RP_coeff=1e-6, 
                        seed=1):
    """
    Executes Figure 3 benchmark experiment evaluating true edge detection frequency
    and false positive detection heatmaps across repetitions.

    Parameters
    ----------
    true_cov : ndarray of shape (d, d)
        True population covariance matrix.
    true_P : ndarray of shape (d, d)
        True underlying precision matrix.
    n_samples : int, default=30
        Sample size for multivariate normal realizations.
    k_edges : int, default=30
        Number of edges extracted per run.
    n_reps : int, default=100
        Number of random realizations.
    methods : list of str, default=['GSL', 'PCorr', 'Prec']
        Solvers to evaluate.
    activ_mul : float, default=0.3
        Multiplicative constant for GSL inner loop (FC).
    activ_add : int, default=10
        Additive constant for GSL inner loop (FC).
    tau_improv : float, default=1e-3
        Objective improvement threshold for GSL inner loop (FC).
    RP_coeff : float, default=1e-6
        Ridge penalty coefficient applied to sample covariance.
    seed : int, default=1
        Base random seed.
    """
    d = true_cov.shape[0]
    mean_vector = np.zeros(d)
    rng = np.random.default_rng(seed=seed)

    # 1. Calculate true partial correlation matrix
    true_pcorr = sol.prec_to_pcorr(true_P)

    # 2. Extract and sort true edges by precision magnitude (descending)
    r_true, c_true = np.triu_indices(d, k=1)
    true_mask = (true_P[r_true, c_true] != 0)
    raw_true_edges = list(zip(r_true[true_mask].tolist(), c_true[true_mask].tolist()))

    true_edge_info = []
    for i, j in raw_true_edges:
        true_edge_info.append({
            'edge': (i, j),
            'label': f"({i+1},{j+1})",
            'prec_mag': abs(true_P[i, j]),
            'pcorr_mag': abs(true_pcorr[i, j])
        })

    true_edge_info.sort(key=lambda x: x['prec_mag'], reverse=True)
    sorted_true_edges = [x['edge'] for x in true_edge_info]

    # 3. Create boolean mask for true edges
    true_edge_mask_2d = np.zeros((d, d), dtype=bool)
    for i, j in sorted_true_edges:
        true_edge_mask_2d[i, j] = True
        true_edge_mask_2d[j, i] = True

    # 4. Initialize counts
    true_counts = {m: {e: 0 for e in sorted_true_edges} for m in methods}
    fp_counts = {m: np.zeros((d, d), dtype=int) for m in methods}

    print(f"Starting experiment: d={d}, reps={n_reps}, RP_coeff={RP_coeff}")

    # Clean progress bar matching Figure 2 format
    for r in tqdm(range(n_reps), desc=f"Sample size n = {n_samples}"):
        realisations = rng.multivariate_normal(mean_vector, true_cov, size=n_samples, method='cholesky')
        Sigma_hat = (realisations.T @ realisations) / n_samples
        RP_val = np.mean(np.diag(Sigma_hat)) * RP_coeff
        S_test = Sigma_hat + RP_val * np.eye(d)

        P_sample = np.linalg.inv(S_test)

        for m in methods:
            if m == 'GSL':
                out = sol.GSL_solver(
                    S=S_test,
                    max_iter=k_edges,
                    true_P=true_P,
                    activ_mul=activ_mul,
                    activ_add=activ_add,
                    tau_improv=tau_improv
                )
                selected = out['evo_edge']
            elif m == 'Prec':
                out = sol.magnitude_xtr(P_sample, k=k_edges, true_P=true_P)
                selected = out['evo_edge']
            elif m == 'PCorr':
                P_pcorr = sol.prec_to_pcorr(P_sample)
                out = sol.magnitude_xtr(P_pcorr, k=k_edges, true_P=true_P)
                selected = out['evo_edge']
            else:
                continue

            seen = set()
            for u, v in selected:
                edge = (min(u, v), max(u, v))
                if edge in seen:
                    continue
                seen.add(edge)

                if true_edge_mask_2d[edge[0], edge[1]]:
                    if edge in true_counts[m]:
                        true_counts[m][edge] += 1
                else:
                    fp_counts[m][edge[0], edge[1]] += 1
                    fp_counts[m][edge[1], edge[0]] += 1

    # Convert counts to percentages (%)
    detection_freq = {
        m: [(true_counts[m][e] / n_reps) * 100.0 for e in sorted_true_edges]
        for m in methods
    }
    fp_freq = {m: (fp_counts[m] / n_reps) * 100.0 for m in methods}

    return {
        'true_edge_info': true_edge_info,
        'detection_freq': detection_freq,
        'fp_freq': fp_freq,
        'true_edge_mask_2d': true_edge_mask_2d,
        'n_samples': n_samples,
        'k_edges': k_edges,
        'n_reps': n_reps,
        'dim': d
    }

#################
def plot_figure_3(results, 
                  font_scale=1.0, 
                  use_stix=False, 
                  color_map=None, 
                  save_path=None):
    r"""
    Plots Figure 3: True edge properties & detection frequency (left) 
    and GSL False Positive heatmap with black diagonal (right).

    Parameters
    ----------
    results : dict
        Output dictionary from run_experiment_fig3.
    font_scale : float, default=1.0
        Font scaling multiplier.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath LaTeX fonts.
    color_map : dict, optional
        Custom solver color palette.
    save_path : str or Path, optional
        Destination filepath to save the high-resolution vector PDF.
    """
    if color_map is None:
        color_map = color_map_glob

    # Configure typography
    _init_plot_style(use_stix=use_stix)

    title_fs  = 12 * font_scale
    label_fs  = 11 * font_scale
    tick_fs   = 9.5 * font_scale
    legend_fs = 9.5 * font_scale

    info = results['true_edge_info']
    ka = len(info)
    lbls = [e['label'] for e in info]
    qv = [e['prec_mag'] for e in info]
    pc_vals = [e['pcorr_mag'] for e in info]
    xp = np.arange(ka)

    n_samples = results['n_samples']
    k_edges = results['k_edges']
    p = results['dim']
    tm = results['true_edge_mask_2d']

    # --- LAYOUT GEOMETRY MATH (Heatmap size reduced by 10% from 1.725 -> 1.5525) ---
    cell_size = 2.5 * font_scale
    fw = max(5.0, ka * 0.12)
    fhs, fpv = 1.6, 0.45
    
    # Heatmap panel size scaled down by 10%
    mw, mhs = cell_size * 1.5525, cell_size * 1.5525
    
    hsl = 2 * fhs + fpv
    cbg, cbt = 0.1, 0.15
    hsrt = mhs + cbg + cbt
    tmr, bmr = 0.12, 0.12
    fhc = max(hsl, hsrt)
    fh = fhc + tmr + bmr

    fig = plt.figure(figsize=(fw + mw, fh))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[fw, mw], wspace=0.15,
                  left=0.08, right=0.96, top=1.0 - tmr/fh, bottom=bmr/fh)

    ol, or_ = (fhc - hsl) / 2, (fhc - hsrt) / 2
    lhr = [ol, fhs, fpv, fhs, ol]
    gsl = gs[0].subgridspec(len(lhr), 1, height_ratios=lhr, hspace=0)

    rhr = [or_ + 0.2, mhs, cbg + cbt, or_]
    gsr = gs[1].subgridspec(len(rhr), 1, height_ratios=rhr, hspace=0)

    # --- TOP SUBPLOT: True Edges (Partial Correlation vs Precision) ---
    ax_top = fig.add_subplot(gsl[1])
    ax_top.set_title(
        r"$\mathbf{True\ edges}$" if use_stix else r"\textbf{True edges}", 
        fontsize=title_fs
    )
    ax_top.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_top.xaxis.grid(False)
    ax_top2 = ax_top.twinx()
    ax_top2.grid(False)

    shift = 0.15
    # Partial correlation (Teal)
    ax_top.vlines(xp - shift, 0, pc_vals, color='teal', lw=1.2, alpha=1.0, zorder=10, clip_on=False)
    ax_top.scatter(xp - shift, pc_vals, color='teal', s=14, alpha=1.0, label='Partial correlation', zorder=10, clip_on=False)
    ax_top.set_ylim(0, 1.0)
    ax_top.set_xlim(-0.5, ka - 0.5)
    ax_top.set_xticks(range(ka))
    ax_top.set_xticklabels([])

    # Precision magnitude (Dark Orange)
    ax_top2.vlines(xp + shift, 0, qv, color='darkorange', lw=1.2, alpha=1.0, zorder=10, clip_on=False)
    ax_top2.scatter(xp + shift, qv, color='darkorange', s=14, label='Precision (abs. value)', zorder=10, clip_on=False)
    ax_top2.set_ylim(0, 10)
    ax_top2.set_yticks([0, 2, 4, 6, 8, 10])

    ax_top.tick_params(axis='y', colors='teal', labelsize=tick_fs)
    ax_top2.tick_params(axis='y', colors='darkorange', labelsize=tick_fs)

    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_top2.get_legend_handles_labels()
    leg1 = ax_top.legend(
        h1 + h2, l1 + l2, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
        borderaxespad=0.35, handlelength=1.2, frameon=True, fontsize=legend_fs
    )
    leg1.get_frame().update({'facecolor': 'white', 'alpha': 1.0, 'edgecolor': 'black'})
    leg1.set_zorder(100)

    # --- BOTTOM SUBPLOT: Frequency of Detection (%) ---
    ax_bot = fig.add_subplot(gsl[3])
    
    title_text = (
        rf"$\mathbf{{Frequency\ of\ detection\ (\%)\ for\ n = {n_samples},\ k = {k_edges}}}$"
        if use_stix else
        rf"\textbf{{Frequency of detection (\%) for}} {{\boldmath $n = {n_samples}$}}, {{\boldmath $k = {k_edges}$}}"
    )
    ax_bot.set_title(title_text, fontsize=title_fs, pad=10)
    ax_bot.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_bot.xaxis.grid(False)

    solvers = [m for m in ['GSL', 'PCorr', 'Prec'] if m in results['detection_freq']]
    bw = 0.8 / max(1, len(solvers))

    for sn_i, sn in enumerate(solvers):
        clr = color_map.get(sn, 'black')
        frq = np.array(results['detection_freq'][sn])
        x_pos = np.arange(ka) + (sn_i - (len(solvers) - 1) / 2) * bw
        ax_bot.vlines(x_pos, 0, frq, color=clr, lw=1.2, alpha=1.0, zorder=10, clip_on=False)
        ax_bot.scatter(x_pos, frq, color=clr, s=14, alpha=1.0, zorder=10, clip_on=False)

    ax_bot.set_xlim(-0.5, ka - 0.5)
    ax_bot.set_ylim(0, 100)
    ax_bot.set_yticks([0, 20, 40, 60, 80, 100])
    ax_bot.set_xticks(range(ka))
    ax_bot.set_xticklabels(lbls, rotation=90, fontsize=tick_fs)
    ax_bot.tick_params(axis='y', labelsize=tick_fs)

    leg_handles = [
        Line2D([0], [0], color=color_map.get(sn, 'black'), marker='o', markersize=5, 
               linestyle='None', label=sn) for sn in solvers
    ]
    leg2 = ax_bot.legend(
        handles=leg_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
        borderaxespad=0.35, ncol=len(leg_handles), handlelength=1.2, frameon=True, fontsize=legend_fs
    )
    leg2.get_frame().update({'facecolor': 'white', 'alpha': 1.0, 'edgecolor': 'black'})
    leg2.set_zorder(100)

    # --- RIGHT PANEL: GSL False Positive Heatmap ---
    ax_map = fig.add_subplot(gsr[1])
    ax_map.set_aspect('equal')

    gsl_fp = results['fp_freq'].get('GSL', np.zeros((p, p)))
    gm = np.max(gsl_fp[~tm]) if np.any(~tm) else 1.0
    vmax = max(1, int(np.ceil(gm / 4.0))) * 4

    gsl_clr = color_map.get('GSL', (0.7, 0.13, 0.13))
    cmap = mcolors.LinearSegmentedColormap.from_list("wg", ["white", gsl_clr])

    # Combine true edges and matrix diagonal into black overlay
    black_mask = tm | np.eye(p, dtype=bool)

    fmf = gsl_fp.copy()
    tb = np.full((p, p), np.nan)
    fmf[black_mask] = np.nan
    tb[black_mask] = 1.0

    ax_map.pcolormesh(fmf, cmap=cmap, vmin=0, vmax=vmax)
    ax_map.pcolormesh(tb, cmap=mcolors.ListedColormap(['black']), vmin=0, vmax=1)
    ax_map.invert_yaxis()
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    ax_map.set_title(
        r"$\mathbf{GSL\ -\ False\ positive\ (\%)}$" if use_stix else r"\textbf{GSL - False positive (\%)}", 
        fontsize=title_fs, pad=6
    )

    # Colorbar aligned under heatmap
    pos = ax_map.get_position()
    ch, gap = cbt / fh, cbg / fh
    cax = fig.add_axes([pos.x0, pos.y0 - gap - ch, pos.width, ch])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=vmax), cmap=cmap), 
        cax=cax, orientation='horizontal'
    )
    cb.set_ticks(np.linspace(0, vmax, 5, dtype=int))
    cb.ax.tick_params(labelsize=tick_fs)

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 3 PDF to: {save_path}")

    plt.show()
    return fig

#######################
#######################
## Figure 4
def _worker_fig4_rep(r, n_samples, true_cov, true_P, max_iter, 
                     activ_mul, activ_add, tau_improv, RP_coeff, seed, dim, mean_vector):
    """Worker process executing full GSL growth for a single Monte Carlo repetition."""
    rep_seed = seed + 1000 * n_samples + r
    rng = np.random.default_rng(seed=rep_seed)

    # 1. Sample multivariate normal realizations
    realisations = rng.multivariate_normal(mean_vector, true_cov, size=n_samples, method='cholesky')

    # 2. Compute sample covariance S_test with ridge penalty
    Sigma_hat = (realisations.T @ realisations) / n_samples
    RP_val = np.mean(np.diag(Sigma_hat)) * RP_coeff
    S_test = Sigma_hat + RP_val * np.eye(dim)

    # 3. Execute full GSL growth trajectory
    gsl_out = sol.GSL_solver(
        S=S_test,
        max_iter=max_iter,
        true_P=true_P,
        activ_mul=activ_mul,
        activ_add=activ_add,
        tau_improv=tau_improv
    )
    
    return gsl_out['evo_edge']


def run_experiment_fig4(true_cov, 
                        true_P, 
                        n_samples=90, 
                        max_iter=None, 
                        n_reps=500, 
                        activ_mul=0.3,
                        activ_add=10,
                        tau_improv=1e-3,
                        RP_coeff=1e-6, 
                        n_jobs=-1,
                        seed=1):
    """
    Executes the Figure 4 benchmark evaluating GSL edge activation ranks 
    under full growth across Monte Carlo repetitions in parallel.

    Parameters
    ----------
    true_cov : ndarray of shape (d, d)
        True population covariance matrix.
    true_P : ndarray of shape (d, d)
        True underlying precision matrix.
    n_samples : int, default=90
        Sample size per dataset realization.
    max_iter : int, optional
        Total outer iterations for full growth (defaults to d * (d - 1) // 2).
    n_reps : int, default=500
        Number of Monte Carlo repetitions.
    activ_mul : float, default=0.3
        Multiplicative constant for GSL inner loop (FC).
    activ_add : int, default=10
        Additive constant for GSL inner loop (FC).
    tau_improv : float, default=1e-3
        Objective improvement threshold for GSL inner loop (FC).
    RP_coeff : float, default=1e-6
        Ridge penalty coefficient applied to sample covariance.
    n_jobs : int, default=-1
        Number of parallel CPU workers (-1 uses all logical cores).
    seed : int, default=1
        Base random seed for reproducible sampling.
    """
    dim = true_cov.shape[0]
    mean_vector = np.zeros(dim)
    
    if max_iter is None:
        max_iter = (dim * (dim - 1)) // 2

    print(f"Starting Figure 4 experiment: d={dim}, n={n_samples}, max_iter={max_iter}, reps={n_reps}, n_jobs={n_jobs}")

    # Parallel execution over Monte Carlo repetitions
    rep_trajectories = Parallel(n_jobs=n_jobs)(
        delayed(_worker_fig4_rep)(
            r, n_samples, true_cov, true_P, max_iter, 
            activ_mul, activ_add, tau_improv, RP_coeff, seed, dim, mean_vector
        )
        for r in tqdm(range(n_reps), desc=f"Sample size n = {n_samples}", mininterval=2.0)
    )

    # Aggregate 1-based activation rank for each edge across repetitions
    edge_positions = defaultdict(list)
    for evo_edge in rep_trajectories:
        seen = set()
        for pos, (u, v) in enumerate(evo_edge, start=1):
            edge = (min(u, v), max(u, v))
            if edge not in seen:
                seen.add(edge)
                edge_positions[edge].append(pos)

    return {
        'edge_positions': edge_positions,
        'true_P': true_P,
        'n_samples': n_samples,
        'n_reps': n_reps,
        'max_iter': max_iter,
        'dim': dim
    }

#######################
def plot_figure_4(results, 
                  top_k=55, 
                  font_scale=1.0, 
                  use_stix=False, 
                  save_path=None):
    r"""
    Plots Figure 4: Rank boxplots of the top_k foremost edges (left subpanel) 
    and remaining true edges (right subpanel) ordered by median activation rank.

    Parameters
    ----------
    results : dict
        Output dictionary from run_experiment_fig4.
    top_k : int, default=55
        Number of foremost activated edges to display in the primary panel.
    font_scale : float, default=1.0
        Font scaling factor.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath fonts.
    save_path : str or Path, optional
        Destination filepath to save vector PDF.
    """
    # Plot style 
    _init_plot_style(use_stix=use_stix)

    title_fs  = 12 * font_scale
    tick_fs   = 9.5 * font_scale
    legend_fs = 9.5 * font_scale

    edge_positions = results['edge_positions']
    true_P = results['true_P']
    n_samples = results['n_samples']
    n_reps = results['n_reps']
    d = results['dim']

    # Extract set of true edges (upper triangular)
    r_idx, c_idx = np.triu_indices(d, k=1)
    true_mask = (true_P[r_idx, c_idx] != 0)
    true_edges = set(zip(r_idx[true_mask].tolist(), c_idx[true_mask].tolist()))

    # Calculate median rank for every edge observed
    edge_stats = []
    for edge, positions in edge_positions.items():
        med = float(np.median(positions))
        is_true = edge in true_edges
        edge_stats.append({
            'edge': edge,
            'positions': positions,
            'median': med,
            'is_true': is_true
        })

    # Order all edges by median activation rank ascending
    edge_stats.sort(key=lambda x: x['median'])

    # Separate foremost edges and remaining true edges
    foremost = edge_stats[:top_k]
    foremost_set = {x['edge'] for x in foremost}
    
    remaining_true = [x for x in edge_stats if x['is_true'] and x['edge'] not in foremost_set]
    remaining_true.sort(key=lambda x: x['median'])

    # Geometry setup (Spaced out right panel boxplots so top indices don't overlap)
    n_main = len(foremost)
    n_rem = len(remaining_true)
    step_rem = 1.6  # Horizontal spacing factor for side panel boxplots
    
    rem_span = max(1.0, (n_rem - 1) * step_rem + 1.0)
    
    fig_width = max(9.975, (n_main + rem_span + 3.5) * 0.171)
    fig = plt.figure(figsize=(fig_width, 3.04))

    gs = GridSpec(1, 2, figure=fig, width_ratios=[n_main + 0.3, rem_span + 0.3], wspace=0.03)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_rem  = fig.add_subplot(gs[0, 1])

    color_true  = (0.13, 0.55, 0.13)  # Forest green
    color_false = (0.80, 0.20, 0.20)  # Red

    # Helper function to render boxplots on a panel
    def draw_boxplots(ax, edge_list, x_coords=None):
        for idx, item in enumerate(edge_list):
            ps = item['positions']
            clr = color_true if item['is_true'] else color_false
            
            q1, q3, med = np.percentile(ps, [25, 75, 50])
            iqr = q3 - q1
            lo, hi = max(min(ps), q1 - 1.5 * iqr), min(max(ps), q3 + 1.5 * iqr)
            
            bw = 0.6
            x = (idx + 1) if x_coords is None else x_coords[idx]

            # Box patch
            ax.add_patch(
                plt.Rectangle((x - bw/2, q1), bw, q3 - q1, 
                              facecolor=(*clr, 0.4), edgecolor=clr, lw=1.0, zorder=2)
            )
            # Median line
            ax.plot([x - bw/2, x + bw/2], [med, med], color='black', lw=1.5, zorder=3)
            # Whiskers & Caps
            ax.plot([x, x], [lo, q1], color=clr, lw=0.8, zorder=2)
            ax.plot([x, x], [q3, hi], color=clr, lw=0.8, zorder=2)
            ax.plot([x - bw/4, x + bw/4], [lo, lo], color=clr, lw=0.8, zorder=2)
            ax.plot([x - bw/4, x + bw/4], [hi, hi], color=clr, lw=0.8, zorder=2)
            
            # Outliers
            outs = [pv for pv in ps if pv < lo or pv > hi]
            if outs:
                ax.scatter([x] * len(outs), outs, color=(*clr, 0.6), s=6, zorder=4)

    # 1. Draw Main Panel Boxplots
    draw_boxplots(ax_main, foremost)
    
    # 2. Draw Side Panel Boxplots with increased horizontal spacing
    rem_x_coords = [1.0 + i * step_rem for i in range(n_rem)]
    if n_rem > 0:
        draw_boxplots(ax_rem, remaining_true, x_coords=rem_x_coords)

    # Formatting Limits and Axis Grid
    y_max = 1400
    for ax in (ax_main, ax_rem):
        ax.set_ylim(0, y_max)
        ax.set_yticks([0, 400, 800, 1200])
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    ax_main.set_xlim(0.5, n_main + 0.5)
    if n_rem > 0:
        ax_rem.set_xlim(0.5, rem_x_coords[-1] + 0.5)
        ax_rem.set_yticklabels([])

    # Vertical markers at every 10th position in main panel
    for k in range(10, n_main + 1, 10):
        ax_main.axvline(x=k, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax_main.text(k, y_max * 1.02, str(k), ha='center', va='bottom', 
                     fontsize=tick_fs, color='black', clip_on=False)

    # Display median activation ranks above side panel
    if n_rem > 0:
        for idx, item in enumerate(remaining_true):
            x_pos = rem_x_coords[idx]
            ax_rem.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
            ax_rem.text(x_pos, y_max * 1.02, str(int(round(item['median']))), 
                        ha='center', va='bottom', fontsize=tick_fs, color='black', clip_on=False)

    # X-Axis Labels 1-indexed (u+1, v+1)
    main_labels = [f"({u+1},{v+1})" for (u, v) in [x['edge'] for x in foremost]]
    ax_main.set_xticks(range(1, n_main + 1))
    ax_main.set_xticklabels(main_labels, rotation=90, fontsize=tick_fs)

    for tick_idx, tick_label in enumerate(ax_main.get_xticklabels()):
        lbl_color = color_true if foremost[tick_idx]['is_true'] else color_false
        tick_label.set_color(lbl_color)

    if n_rem > 0:
        rem_labels = [f"({u+1},{v+1})" for (u, v) in [x['edge'] for x in remaining_true]]
        ax_rem.set_xticks(rem_x_coords)
        ax_rem.set_xticklabels(rem_labels, rotation=90, fontsize=tick_fs)
        for tick_label in ax_rem.get_xticklabels():
            tick_label.set_color(color_true)

    # Title
    title_str = (
        rf"$\mathbf{{GSL\ (full\ growth),\ n = {n_samples}\ and\ {n_reps}\ repetitions}}$"
        if use_stix else
        rf"\textbf{{GSL (full growth), }} {{\boldmath $n={n_samples}$}} \textbf{{ and {n_reps} repetitions}}"
    )
    ax_main.text(0.01, 0.96, title_str, transform=ax_main.transAxes, va='top', fontsize=title_fs)

    # Legend in Bottom-Left
    legend_elements = [
        Patch(facecolor=(*color_true, 0.4), edgecolor=color_true, label='True'),
        Patch(facecolor=(*color_false, 0.4), edgecolor=color_false, label='False')
    ]
    leg = ax_main.legend(
        handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.05),
        frameon=True, facecolor='white', framealpha=1.0, edgecolor='black', fontsize=legend_fs
    )
    leg.set_zorder(100)

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 4 PDF to: {save_path}")

    plt.show()
    return fig



##########################
##########################
## Figure 5 worker process
def _worker_fig5_rep(r, true_cov, true_P, n_samples, max_iter, taus, 
                     activ_mul, activ_add, RP_coeff, seed, dim, mean_vector, n_true_edges):
    """Worker process executing a single Monte Carlo repetition for Figure 5."""
    rep_seed = seed + 1000 * n_samples + r
    rng = np.random.default_rng(seed=rep_seed)

    # 1. Generate sample covariance realization with ridge regularisation
    realisations = rng.multivariate_normal(mean_vector, true_cov, size=n_samples, method='cholesky')
    Sigma_hat = (realisations.T @ realisations) / n_samples
    RP_val = np.mean(np.diag(Sigma_hat)) * RP_coeff
    S_test = Sigma_hat + RP_val * np.eye(dim)

    rep_results = {}

    # 2. Sequential solver evaluation across tau thresholds
    for tau in taus:
        gsl_out = sol.GSL_solver(
            S=S_test,
            max_iter=max_iter,
            true_P=true_P,
            activ_mul=activ_mul,
            activ_add=activ_add,
            tau_improv=tau
        )

        inner_counts = np.array(gsl_out['evo_inner_up'][:max_iter])
        edge_is_true = np.array(gsl_out['edge_is_true'][:max_iter])
        recall_pct = (np.cumsum(edge_is_true) / n_true_edges) * 100.0

        rep_results[tau] = {
            'inner_counts': inner_counts,
            'recall_pct': recall_pct
        }

    return rep_results


###############################
## Figure 5 experiment function
def run_experiment_fig5(true_cov, 
                        true_P, 
                        n_samples=90, 
                        max_iter=300, 
                        n_reps=100,
                        taus=[1e-1, 1e-3, 1e-5], 
                        activ_mul=1.0, 
                        activ_add=10, 
                        RP_coeff=1e-6, 
                        n_jobs=-1,
                        seed=555):
    """
    Executes Figure 5 Monte Carlo benchmark evaluating inner iteration counts and recall % 
    trajectories across stopping thresholds tau, returning medians and 10th-90th percentiles.
    """
    d = true_cov.shape[0]
    mean_vector = np.zeros(d)

    r_idx, c_idx = np.triu_indices(d, k=1)
    n_true_edges = int(np.sum(true_P[r_idx, c_idx] != 0))

    print(f"Starting Figure 5 Monte Carlo experiment: d={d}, n={n_samples}, max_iter={max_iter}, reps={n_reps}, n_jobs={n_jobs}")

    # Parallel execution over Monte Carlo repetitions
    rep_outputs = Parallel(n_jobs=n_jobs)(
        delayed(_worker_fig5_rep)(
            r, true_cov, true_P, n_samples, max_iter, taus, 
            activ_mul, activ_add, RP_coeff, seed, d, mean_vector, n_true_edges
        )
        for r in tqdm(range(n_reps), desc=f"Monte Carlo reps (n={n_samples})", mininterval=2.0)
    )

    outer_steps = np.arange(1, max_iter + 1)
    results_by_tau = {}

    # Aggregate repetitions and compute median & interdecile ranges
    for tau in taus:
        inner_matrix = np.array([rep[tau]['inner_counts'] for rep in rep_outputs])  # (n_reps, max_iter)
        recall_matrix = np.array([rep[tau]['recall_pct'] for rep in rep_outputs])   # (n_reps, max_iter)

        results_by_tau[tau] = {
            'outer_steps': outer_steps,
            # Inner iterations summary statistics
            'med_inner': np.median(inner_matrix, axis=0),
            'p10_inner': np.percentile(inner_matrix, 10, axis=0),
            'p90_inner': np.percentile(inner_matrix, 90, axis=0),
            # Recall % summary statistics
            'med_recall': np.median(recall_matrix, axis=0),
            'p10_recall': np.percentile(recall_matrix, 10, axis=0),
            'p90_recall': np.percentile(recall_matrix, 90, axis=0)
        }

    return {
        'results_by_tau': results_by_tau,
        'n_samples': n_samples,
        'max_iter': max_iter,
        'n_reps': n_reps,
        'dim': d
    }


#############################
## Figure 5 plotting function
def plot_figure_5(results, 
                  font_scale=1.5, 
                  use_stix=False, 
                  save_path=None):
    r"""
    Plots Figure 5: 3-panel grid showing median GSL inner iterations (left y-axis, red) 
    and median Recall % (right y-axis, green) with 10th-90th percentile interdecile shaded bands
    versus outer iterations for tau in [10^-1, 10^-3, 10^-5].
    """
    # Plot style 
    _init_plot_style(use_stix=use_stix)

    title_fs = 12 * font_scale
    label_fs = 11 * font_scale
    tick_fs  = 9.5 * font_scale

    results_by_tau = results['results_by_tau']
    max_iter = results['max_iter']
    taus = [1e-1, 1e-3, 1e-5]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout='compressed')

    color_inner = (0.70, 0.13, 0.13)   # Red
    color_recall = (0.13, 0.55, 0.13)  # Forest Green

    y_limits = {
        1e-1: (0, 12, [0, 3, 6, 9, 12]),
        1e-3: (0, 80, [0, 20, 40, 60, 80]),
        1e-5: (0, 360, [0, 90, 180, 270, 360])
    }

    title_labels = {
        1e-1: r"$\boldsymbol{\tau} = \mathbf{10^{-1}}$" if use_stix else r"{\boldmath $\tau = 10^{-1}$}",
        1e-3: r"$\boldsymbol{\tau} = \mathbf{10^{-3}}$" if use_stix else r"{\boldmath $\tau = 10^{-3}$}",
        1e-5: r"$\boldsymbol{\tau} = \mathbf{10^{-5}}$" if use_stix else r"{\boldmath $\tau = 10^{-5}$}"
    }

    for idx, tau in enumerate(taus):
        ax_left = axes[idx]
        ax_right = ax_left.twinx()

        data = results_by_tau[tau]
        x_vals = data['outer_steps']

        # Extract Medians & Interdecile Ranges
        med_inner, p10_inner, p90_inner = data['med_inner'], data['p10_inner'], data['p90_inner']
        med_rec, p10_rec, p90_rec = data['med_recall'], data['p10_recall'], data['p90_recall']

        # --- Plot Inner Iterations (Red) ---
        ax_left.plot(x_vals, med_inner, color=color_inner, lw=1.5, zorder=3)
        ax_left.fill_between(x_vals, p10_inner, p90_inner, color=color_inner, alpha=0.20, zorder=2)
        ax_left.set_xlim(0, max_iter)
        ax_left.set_xticks([0, 100, 200, 300])
        ax_left.set_xlabel("Outer iteration", fontsize=label_fs, labelpad=6)

        ymin, ymax, yticks = y_limits[tau]
        ax_left.set_ylim(ymin, ymax)
        ax_left.set_yticks(yticks)
        ax_left.tick_params(axis='y', colors=color_inner, labelsize=tick_fs)
        ax_left.tick_params(axis='x', labelsize=tick_fs)
        ax_left.grid(True, linestyle='--', alpha=0.4, zorder=0)

        if idx == 0:
            ax_left.set_ylabel("Nb of inner iterations", color=color_inner, fontsize=label_fs, labelpad=6)

        # --- Plot Recall % (Green) ---
        ax_right.plot(x_vals, med_rec, color=color_recall, lw=1.6, zorder=4)
        ax_right.fill_between(x_vals, p10_rec, p90_rec, color=color_recall, alpha=0.20, zorder=2)
        ax_right.set_ylim(0, 100)
        ax_right.set_yticks([0, 25, 50, 75, 100])
        ax_right.grid(False)

        if idx == 2:
            ax_right.tick_params(axis='y', colors=color_recall, labelsize=tick_fs)
            ax_right.set_ylabel(r"Recall (\%)", color=color_recall, fontsize=label_fs, labelpad=6)
        else:
            ax_right.set_yticklabels([])
            ax_right.tick_params(axis='y', colors=color_recall, length=0)

        ax_left.set_title(title_labels[tau], fontsize=title_fs, pad=8)

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 5 PDF to: {save_path}")

    plt.show()
    return fig


#######################
#######################
## Figures 6 and 7  
def plot_figure_6(experiments_data, 
                  n_samples_list=[30, 90, 160], 
                  font_scale=1.2, 
                  scale_x=1.0, 
                  use_stix=True,
                  inter_dec=True,
                  glasso_legend='top',
                  color_map=None,
                  save_path=None):
    r"""
    Plots Figure 6: A 4x4 grid displaying precision matrix sparsity patterns (Column 1)
    alongside Precision-Recall curves across sample sizes n in [30, 90, 160] (Columns 2-4)
    for four network settings (Random m=200, Random m=40, Clique m=225, Hub m=45).

    Parameters
    ----------
    experiments_data : list of dict
        List of 4 experiment dictionaries in order [Random m=200, Random m=40, Clique m=225, Hub m=45].
        Each dict must contain:
          - 'name': str (e.g., r"Random ($m=200$)")
          - 'true_P' or 'Q_true': ndarray (d, d) true precision matrix
          - 'results' or 'results_map': dict returned by run_experiment_ROC
    n_samples_list : list of int, default=[30, 90, 160]
        Sample sizes corresponding to columns 2, 3, and 4.
    font_scale : float, default=1.2
        Typography scaling factor.
    scale_x : float, default=1.0
        Horizontal scaling factor for figure width.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath LaTeX fonts.
    inter_dec : bool, default=True
        If True, plots 10th-90th percentile interdecile shaded bands.
    glasso_legend : str, default='top'
        Position for the GLasso legend. If 'top', uses 'upper right', 
        otherwise uses 'lower left'.
    color_map : dict, optional
        Custom solver color map (defaults to color_map_glob).
    save_path : str or Path, optional
        Destination filepath to save the vector PDF.
    """
    if color_map is None:
        color_map = color_map_glob

    # Configure typography
    _init_plot_style(use_stix=use_stix)

    title_fs  = 13 * font_scale
    label_fs  = 11 * font_scale
    tick_fs   = 9.5 * font_scale
    legend_fs = 9.5 * font_scale

    n_rows = len(experiments_data)
    n_cols = len(n_samples_list) + 1  # 1 sparsity + 3 PR curve columns

    # Layout geometry: 1:1:1:1 width ratios ensuring exact same window sizes
    fig = plt.figure(figsize=(10.5 * scale_x, 10.0), layout='compressed')
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=[1.0] * n_cols)

    ticks = [0, 25, 50, 75, 100]

    for row_idx, exp in enumerate(experiments_data):
        exp_name = exp.get('name', f"Setting {row_idx + 1}")
        true_P   = exp.get('true_P', exp.get('Q_true'))
        results  = exp.get('results', exp.get('results_map'))

        # --- COLUMN 1: Sparsity Pattern ---
        ax_sparse = fig.add_subplot(gs[row_idx, 0])
        ax_sparse.spy(true_P, markersize=1.2, color='black')
        ax_sparse.set_box_aspect(1)
        ax_sparse.set_xticks([])
        ax_sparse.set_yticks([])
        ax_sparse.set_ylabel(exp_name, fontsize=title_fs, labelpad=8)

        if row_idx == 0:
            ax_sparse.set_title(
                r"$\mathbf{Sparsity\ pattern\ of\ \Theta}$" if use_stix else r"$\bm{\text{Sparsity pattern of }\Theta}$", 
                fontsize=title_fs, pad=8
            )

        # --- COLUMNS 2-4: Precision-Recall Curves ---
        for col_idx, n_val in enumerate(n_samples_list):
            ax_pr = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax_pr.set_box_aspect(1)

            # Top Column Titles
            if row_idx == 0:
                ax_pr.set_title(
                    rf"$\mathbf{{n = {n_val}}}$" if use_stix else rf"$\bm{{n = {n_val}}}$", 
                    fontsize=title_fs, pad=6
                )

            pr_handles = {}

            if n_val in results:
                for method, data in results[n_val].items():
                    if len(data['prec']) == 0:
                        continue

                    color = color_map.get(method, 'black')

                    prec_arr = np.array(data['prec'])
                    rec_arr  = np.array(data['rec'])

                    med_prec = np.median(prec_arr, axis=0)
                    med_rec  = np.median(rec_arr, axis=0)

                    # Median PR Curve
                    line_pr, = ax_pr.plot(
                        med_prec, med_rec, color=color, linewidth=2, 
                        clip_on=False, zorder=3
                    )
                    pr_handles[method] = line_pr

                    # Optional Interdecile Ranges (10th to 90th percentile)
                    if inter_dec:
                        p10_rec = np.percentile(rec_arr, 10, axis=0)
                        p90_rec = np.percentile(rec_arr, 90, axis=0)
                        ax_pr.fill_between(
                            med_prec, p10_rec, p90_rec, color=color, alpha=0.18, 
                            clip_on=False, zorder=2
                        )

            # Subplot formatting
            ax_pr.set_xlim([0, 100])
            ax_pr.set_ylim([0, 100])
            ax_pr.set_xticks(ticks)
            ax_pr.set_yticks(ticks)
            ax_pr.grid(True, linestyle='--', alpha=0.5)
            ax_pr.tick_params(axis='both', which='major', labelsize=tick_fs)

            # Shared X-axis label (Bottom row only)
            if row_idx == n_rows - 1:
                ax_pr.set_xlabel(r"Precision ($\%$)", fontsize=label_fs, labelpad=2)
            else:
                ax_pr.tick_params(labelbottom=False)

            # Shared Y-axis label (First curve column only)
            if col_idx == 0:
                ax_pr.set_ylabel(r"Recall ($\%$)", fontsize=label_fs)
            else:
                ax_pr.tick_params(labelleft=False)

            # Corrected Distributed Legends on Top Row
            if row_idx == 0:
                if col_idx == 0:
                    handles = [pr_handles[m] for m in ['GSL', 'BBI', 'BFCI'] if m in pr_handles]
                    labels  = [m for m in ['GSL', 'BBI', 'BFCI'] if m in pr_handles]
                    if handles:
                        ax_pr.legend(
                            handles, labels, loc="upper right", frameon=True, 
                            edgecolor='black', facecolor='white', framealpha=1.0, fontsize=legend_fs
                        )
                elif col_idx == 1:
                    if 'Glasso' in pr_handles:
                        glasso_loc = "upper right" if glasso_legend == 'top' else "lower left"
                        ax_pr.legend(
                            [pr_handles['Glasso']], ['GLasso'], loc=glasso_loc, frameon=True, 
                            edgecolor='black', facecolor='white', framealpha=1.0, fontsize=legend_fs
                        )
                elif col_idx == 2:
                    handles = [pr_handles[m] for m in ['PCorr', 'Prec'] if m in pr_handles]
                    labels  = [m for m in ['PCorr', 'Prec'] if m in pr_handles]
                    if handles:
                        ax_pr.legend(
                            handles, labels, loc="lower left", frameon=True, 
                            edgecolor='black', facecolor='white', framealpha=1.0, fontsize=legend_fs
                        )

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 6 PDF to: {save_path}")

    plt.show()
    return fig

#######################
#######################
## Figure 8
def _worker_fig8_rep(r, data, subsample_size, max_iter, 
                    activ_mul, activ_add, tau_improv, RP_coeff, seed, dim):
    """Worker process executing GSL growth for a single empirical data subsample."""
    rep_seed = seed + 1000 * subsample_size + r
    rng = np.random.default_rng(seed=rep_seed)

    # 1. Uniform random subsampling without replacement across samples (rows)
    n_samples_total = data.shape[0]
    subsample_idx = rng.choice(n_samples_total, size=subsample_size, replace=False)
    X_sub = data[subsample_idx, :]

    # 2. Compute sample covariance matrix S_test with ridge penalty (uncentered)
    Sigma_hat = (X_sub.T @ X_sub) / subsample_size
    RP_val = np.mean(np.diag(Sigma_hat)) * RP_coeff
    S_test = Sigma_hat + RP_val * np.eye(dim)

    # 3. Execute full GSL growth trajectory (true_P is unknown)
    gsl_out = sol.GSL_solver(
        S=S_test,
        max_iter=max_iter,
        true_P=None,
        activ_mul=activ_mul,
        activ_add=activ_add,
        tau_improv=tau_improv
    )
    
    return gsl_out['evo_edge']


def run_experiment_fig8(data, 
                        frac_sample=0.5,
                        max_iter=None, 
                        n_reps=500, 
                        activ_mul=0.3,
                        activ_add=10,
                        tau_improv=1e-3,
                        RP_coeff=1e-6, 
                        n_jobs=-1,
                        seed=1):
    """
    Executes the Figure 8 benchmark evaluating GSL edge activation ranks 
    under empirical data subsampling without replacement across parallel repetitions.

    Parameters
    ----------
    data : ndarray of shape (n_samples, d)
        Preprocessed dataset (e.g., X_npn post Nonparanormal transformation).
    frac_sample : float, default=0.5
        Fraction of total samples to draw per subsample realization.
    max_iter : int, optional
        Total outer iterations for GSL growth (defaults to d * (d - 1) // 2).
    n_reps : int, default=500
        Number of subsampling repetitions.
    activ_mul : float, default=0.3
        Multiplicative constant for GSL inner loop (FC).
    activ_add : int, default=10
        Additive constant for GSL inner loop (FC).
    tau_improv : float, default=1e-3
        Objective improvement threshold for GSL inner loop (FC).
    RP_coeff : float, default=1e-6
        Ridge penalty coefficient applied to sample covariance.
    n_jobs : int, default=-1
        Number of parallel CPU workers (-1 uses all logical cores).
    seed : int, default=1
        Base random seed for reproducible subsampling.

    Returns
    -------
    dict
        Dictionary containing edge activation positions across all repetitions,
        subsample size, total repetitions, and dimensions.
    """
    n_samples_total, dim = data.shape
    subsample_size = int(np.ceil(n_samples_total * frac_sample))
    
    if max_iter is None:
        max_iter = (dim * (dim - 1)) // 2

    print(f"Starting Figure 8 Subsampling Benchmark:")
    print(f"  Total Data Shape : {data.shape}")
    print(f"  Subsample Size   : {subsample_size} (frac={frac_sample})")
    print(f"  Max Iterations   : {max_iter}")
    print(f"  Repetitions      : {n_reps}")
    print(f"  Parallel Jobs    : {n_jobs}")

    # Parallel execution over subsamples
    rep_trajectories = Parallel(n_jobs=n_jobs)(
        delayed(_worker_fig8_rep)(
            r, data, subsample_size, max_iter, 
            activ_mul, activ_add, tau_improv, RP_coeff, seed, dim
        )
        for r in tqdm(range(n_reps), desc=f"Subsampling (n={subsample_size})", mininterval=2.0)
    )

    # Aggregate 1-based activation rank for each edge across repetitions
    edge_positions = defaultdict(list)
    for evo_edge in rep_trajectories:
        seen = set()
        for pos, (u, v) in enumerate(evo_edge, start=1):
            edge = (min(u, v), max(u, v))
            if edge not in seen:
                seen.add(edge)
                edge_positions[edge].append(pos)

    return {
        'edge_positions': edge_positions,
        'n_samples_total': n_samples_total,
        'subsample_size': subsample_size,
        'n_reps': n_reps,
        'max_iter': max_iter,
        'dim': dim
    }

#######################
def plot_figure_8(results, 
                  top_k=300, 
                  num_splits=3, 
                  max_y=5000, 
                  font_scale=1.0, 
                  use_stix=False, 
                  save_path=None):
    r"""
    Plots Figure 8: Distribution of edge activation ranks under full GSL growth
    for the Riboflavin dataset across subsamples ordered by median activation rank.

    Parameters
    ----------
    results : dict
        Output dictionary from run_experiment_fig8.
    top_k : int, default=300
        Total number of foremost edges to display across panels.
    num_splits : int, default=3
        Number of vertically stacked subplots (panels).
    max_y : int, default=5000
        Upper limit for the y-axis (activation rank).
    font_scale : float, default=1.0
        Font scaling factor.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath fonts.
    save_path : str or Path, optional
        Destination filepath to save vector PDF.
    """
    # Plot style 
    _init_plot_style(use_stix=use_stix)

    tick_fs = 9.5 * font_scale

    edge_positions = results['edge_positions']

    # 1. Compute median activation rank for every observed edge
    edge_stats = []
    for edge, positions in edge_positions.items():
        if len(positions) > 0:
            med = float(np.median(positions))
            edge_stats.append({
                'edge': edge,
                'positions': positions,
                'median': med
            })

    # 2. Order edges by median activation rank ascending
    edge_stats.sort(key=lambda x: x['median'])
    foremost = edge_stats[:top_k]
    
    total_edges = len(foremost)
    edges_per_split = int(np.ceil(total_edges / num_splits))

    # 3. Geometry & Figure Canvas Setup
    fig_width = max(8.5, edges_per_split * 0.12)
    fig_height = 1.8 * num_splits + 0.6
    
    fig, axes = plt.subplots(num_splits, 1, figsize=(fig_width, fig_height), sharex=False)
    if num_splits == 1:
        axes = [axes]

    color_purple = (0.50, 0.0, 0.50)  # Purple color for Riboflavin boxplots

    # 4. Render Boxplots Row by Row
    for s_idx in range(num_splits):
        ax = axes[s_idx]
        start_idx = s_idx * edges_per_split
        end_idx = min((s_idx + 1) * edges_per_split, total_edges)
        panel_edges = foremost[start_idx:end_idx]
        n_panel = len(panel_edges)
        
        start_rank = start_idx + 1
        
        # Draw Boxplots
        for local_x, item in enumerate(panel_edges, start=1):
            ps = item['positions']
            q1, q3, med = np.percentile(ps, [25, 75, 50])
            iqr = q3 - q1
            lo, hi = max(min(ps), q1 - 1.5 * iqr), min(max(ps), q3 + 1.5 * iqr)
            bw = 0.6

            # Box patch
            ax.add_patch(
                Rectangle((local_x - bw/2, q1), bw, q3 - q1, 
                          facecolor=(*color_purple, 0.4), edgecolor=color_purple, lw=0.8, zorder=2)
            )
            # Median bar
            ax.plot([local_x - bw/2, local_x + bw/2], [med, med], color='black', lw=1.2, zorder=3)
            # Whiskers & Caps
            ax.plot([local_x, local_x], [lo, q1], color=color_purple, lw=0.8, zorder=2)
            ax.plot([local_x, local_x], [q3, hi], color=color_purple, lw=0.8, zorder=2)
            ax.plot([local_x - bw/4, local_x + bw/4], [lo, lo], color=color_purple, lw=0.8, zorder=2)
            ax.plot([local_x - bw/4, local_x + bw/4], [hi, hi], color=color_purple, lw=0.8, zorder=2)
            
            # Outliers
            outs = [pv for pv in ps if pv < lo or pv > hi]
            if outs:
                ax.scatter([local_x] * len(outs), outs, color=(*color_purple, 0.6), s=5, zorder=4)

        # Y-Axis Formatting
        ax.set_ylim(0, max_y)
        ax.set_yticks([0, 1250, 2500, 3750, max_y])
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        # X-Axis Formatting & 10-Edge Rank Markers
        ax.set_xlim(0.5, edges_per_split + 0.5)
        ax.set_xticks([])
        
        for local_x in range(10, n_panel + 1, 10):
            global_rank = start_rank + local_x - 1
            ax.axvline(x=local_x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
            ax.text(local_x, -max_y * 0.05, str(global_rank), ha='center', va='top', 
                    fontsize=tick_fs, color='black', clip_on=False)

        # Spine Styling
        for spine_name in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color('black')
            ax.spines[spine_name].set_linewidth(0.8)

    plt.tight_layout(pad=0.5, h_pad=0.8)

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported Figure 8 PDF to: {save_path}")

    plt.show()
    return fig

#######################
#######################
## Figures 9 and 10
def _compute_galaxy_layout(G: nx.Graph, density_factor: float = 0.55, cluster_padding: float = 0.4):
    """Computes Galaxy layout positioning graph components in concentric circles."""
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not components:
        return {}
    
    comp_data = []
    for comp in components:
        n_nodes = len(comp)
        subgraph = G.subgraph(comp)
        if n_nodes < 3: 
            l_pos = nx.spring_layout(subgraph)
        else:
            try: 
                l_pos = nx.kamada_kawai_layout(subgraph, weight=None)
            except Exception: 
                l_pos = nx.spring_layout(subgraph, k=2.0 / np.sqrt(n_nodes), iterations=200)
        
        pts = np.array(list(l_pos.values()))
        if len(pts) > 0:
            center = pts.mean(axis=0)
            pts = pts - center
            max_dist = np.max(np.sqrt(np.sum(pts**2, axis=1))) if len(pts) > 1 else 1.0
            if max_dist < 1e-9: 
                max_dist = 1.0
            target_radius = np.sqrt(n_nodes) * density_factor
            comp_data.append(({node: (pts[i] / max_dist) * target_radius for i, node in enumerate(subgraph.nodes())}, target_radius))

    final_global_pos, placed_circles = {}, []
    spiral_step_angle, spiral_growth = 0.15, 0.35
    
    for i, (pos_data, radius) in enumerate(comp_data):
        if i == 0: 
            cx, cy = 0.0, 0.0
            placed_circles.append((cx, cy, radius))
        else:
            search_theta, found = 0.0, False
            while not found:
                dist = spiral_growth * search_theta
                px, py = dist * np.cos(search_theta), dist * np.sin(search_theta)
                if not any(np.sqrt((px - ex)**2 + (py - ey)**2) < radius + er + cluster_padding for ex, ey, er in placed_circles):
                    cx, cy = px, py
                    placed_circles.append((cx, cy, radius))
                    found = True
                search_theta += spiral_step_angle
                if search_theta > 1000: 
                    cx, cy = (radius + i * 1.5, 0)
                    placed_circles.append((cx, cy, radius))
                    found = True
        
        for node, (nx_val, ny_val) in pos_data.items():
            final_global_pos[node] = np.array([cx + nx_val, cy + ny_val])
            
    return final_global_pos

def plot_graphical_model_grid(activ_rank, 
                               k_vals=[40, 80, 120], 
                               isolated_vrtx=False,
                               p=None,
                               thickness_range=(0.6, 5.0), 
                               node_size=40,
                               edge_color='#D62828',
                               font_scale=1.0,
                               use_stix=False, 
                               save_path=None):
    r"""
    Plots a grid of gene expression cluster networks across different edge counts k_vals.

    Parameters
    ----------
    activ_rank : list of tuple (int, int)
        Edges ordered by activation rank (ascending).
    k_vals : list of int, default=[40, 80, 120]
        Edge counts to plot per panel.
    isolated_vrtx : bool, default=False
        If True, includes isolated nodes (degree 0) in the galaxy layout.
        If False, displays only vertices with at least one active edge.
    p : int, optional
        Total number of features/vertices in the dataset (e.g., p=100).
        If None and isolated_vrtx=True, inferred as max(node_idx) + 1 from activ_rank.
    thickness_range : tuple of (float, float), default=(0.6, 5.0)
        (min_width, max_width) line thickness for edges based on rank.
    node_size : int, default=40
        Diameter of black node markers.
    edge_color : str, default='#D62828'
        Color of graph edges.
    font_scale : float, default=1.0
        Font scaling multiplier.
    use_stix : bool, default=True
        Toggles STIX vs Computer Modern/amsmath fonts.
    save_path : str or Path, optional
        Destination filepath to save vector PDF.
    """
    # Plot style 
    _init_plot_style(use_stix=use_stix)

    title_fs = 14 * font_scale
    n_panels = len(k_vals)
    w_min, w_max = thickness_range

    # Infer total number of vertices p if isolated_vrtx=True and p is not given
    if isolated_vrtx and p is None:
        p = max(max(u, v) for u, v in activ_rank) + 1 if activ_rank else 100

    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.2))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, k in enumerate(k_vals):
        ax = axes[panel_idx]
        sub_edges = activ_rank[:k]
        
        G = nx.Graph()
        
        if isolated_vrtx:
            # Include all p total vertices (including isolated ones)
            G.add_nodes_from(range(p))
        else:
            # Include only active vertices in the current edge set
            sub_nodes = set()
            for u, v in sub_edges:
                sub_nodes.add(u)
                sub_nodes.add(v)
            G.add_nodes_from(sub_nodes)
            
        for u, v in sub_edges:
            G.add_edge(u, v)

        final_pos = _compute_galaxy_layout(G)
        if not final_pos:
            ax.axis('off')
            continue

        # Map edge thickness linearly with rank (earlier edges are thicker)
        edge_widths = []
        for rank_idx in range(len(sub_edges)):
            norm = rank_idx / max(1, k - 1)
            w = w_max - (w_max - w_min) * norm
            edge_widths.append(w)

        # Draw Network
        nx.draw_networkx_edges(G, final_pos, ax=ax, edge_color=edge_color, width=edge_widths, alpha=0.9)
        nx.draw_networkx_nodes(G, final_pos, ax=ax, node_color='black', node_size=node_size, linewidths=0)

        ax.set_aspect('equal')
        ax.axis('off')

        # Panel Title
        title_str = (
            rf"$\mathbf{{k = {k}}}$" if use_stix 
            else rf"{{\boldmath $k = {k}$}}"
        )
        ax.text(0.05, 0.95, title_str, transform=ax.transAxes, fontsize=title_fs, ha='left', va='top')

        # Auto-crop view bounds around Galaxy layout
        all_pts = np.array(list(final_pos.values()))
        margin = 0.8
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        span = max(x_max - x_min, y_max - y_min) / 2 + margin
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)

    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.02)

    # Add black vertical divider lines between panels
    for i in range(1, n_panels):
        x_sep = (axes[i-1].get_position().x1 + axes[i].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x_sep, x_sep], [0.05, 0.95], transform=fig.transFigure, color='black', linewidth=1.5))

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Successfully exported PDF to: {save_path}")

    plt.show()
    return fig