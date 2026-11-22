"""
Solvers and Utility Functions for graphical model recovery

This file implements solvers and evaluation metrics 
for sparse precision matrix estimation and graph structure recovery.

Key Components
--------------
1. Coordinate-Descent Solvers:
   - GSL_solver   : GSL solver with approximate full correction (via support-restricted GSL)
   - BBI_solver   : BBI solver with approximate full correction (via support-restricted GSL)
   - BFCI_solver  : BFCI solver based on approximate full correction (via support-restricted GSL)
   - GS_solver   : GS solver with approximate full correction (via support-restricted GSL)
   

2. Magnitude-Based Extraction:
   - magnitude_xtr: Magnitude-based graph recovery (thresholding).

3. Glasso
   - Glasso_ROC : Graphical Lasso benchmark evaluating precision-recall paths across 
                  discretized lambda values (via gglasso ADMM).

3. Utilities & Evaluation:
   - prec_to_pcorr       : Transforms a precision matrix into partial correlations.
   - compute_number_edges: Counts non-zero off-diagonal entries and extracts true edge indices.
   - compute_ROC         : Calculates precision, recall, and false positive rate trajectories 
                           from sequential edge activation outputs.
"""

import sys
import io
import numpy as np
import numba as nb
from gglasso.problem import glasso_problem


##############################
## Tool functions ############
##############################
## Compute partial correlation
def prec_to_pcorr(precision):
    diag = np.diag(precision)
    diag_sqrt_inv = 1.0 / np.sqrt(diag)
    pcorr = -precision * np.outer(diag_sqrt_inv, diag_sqrt_inv)
    np.fill_diagonal(pcorr, 1.0)
    
    return pcorr

#############################################
## Compute number of edges + identify indices
def compute_number_edges(Q):      # Q is the precision
    A_mat = np.abs(Q) > 0
    np.fill_diagonal(A_mat, False)
    nb_of_edges = int(np.sum(np.triu(A_mat, 1)))
    dim = Q.shape[0]
    true_edge = []

    kk = 0
    for i in range(dim):
        for j in range(i+1,dim):
            if A_mat[i,j] != 0:
                true_edge.append((i,j))
                kk +=1
    if (kk != nb_of_edges): print('Error')

    return nb_of_edges, true_edge

###################################
## Magnitude-based naive extraction 
def magnitude_xtr(P_mat,         # precision or partial correllation
                  k,             # number of edges to extract
                  true_P=None):  # true precision (for test)
    """
    Extracts the top-k largest off-diagonal entries (by absolute value) 
    from a symmetric matrix P_mat.
    """
    d = P_mat.shape[0]
    r, c = np.triu_indices(d, k=1)
    vals = np.abs(P_mat[r, c])
    k_eff = min(k, len(vals))
    top_k_idx = np.argpartition(vals, -k_eff)[-k_eff:]  # keep only top-k values
    top_k_idx = top_k_idx[np.argsort(-vals[top_k_idx])] # sort
    row_idx = r[top_k_idx]
    col_idx = c[top_k_idx]   
    evo_edge = list(zip(row_idx.tolist(), col_idx.tolist()))
    edge_is_true = None
    if true_P is not None:
        edge_is_true = (true_P[row_idx, col_idx] != 0).tolist()

    return { 'evo_edge': evo_edge,
             'edge_is_true': edge_is_true }

    
#######################################################
## Compute information to plot ROC and precision-recall  
def compute_ROC(nb_true_edge,  # nb_true_edge
                M_edge,        # max number of edges
                Run):          # Run is the output of a sequential growth method (list)
    recall = np.cumsum(Run['edge_is_true'])/nb_true_edge * 100   
    precision = np.cumsum(Run['edge_is_true'])/np.arange(1, len(recall)+1) * 100  
    FP = np.arange(1, len(recall)+1) - np.cumsum(Run['edge_is_true']) # False positive
    TN = (M_edge - nb_true_edge) + np.cumsum(Run['edge_is_true']) - np.arange(1, len(recall)+1)
    FP_rate = FP / (FP + TN) * 100    
    return precision, recall, FP_rate



#######################################
## Coordinate descent - update ########
#######################################    
#######################################
## Fast 1-update (i.e. diagonal update)
@nb.njit(fastmath=True)
def _diag_update_jit(S, Q, R, grad, i):
    """Updates Q, R, grad in-place and returns float scalar 'improv'."""
    S_ii = S[i, i]
    R_ii = R[i, i]
    step_size = (1.0 / S_ii) - (1.0 / R_ii)
    beta = S_ii / R_ii
    improv = beta - 1.0 - np.log(beta)
    
    Q[i, i] += step_size
    
    # In-place rank-1 update on R and grad
    factor = step_size / (1.0 + step_size * R_ii)
    R_i = R[i, :].copy()
    for row in range(R.shape[0]):
        val = factor * R_i[row]
        for col in range(R.shape[1]):
            delta = val * R_i[col]
            R[row, col] -= delta
            grad[row, col] += delta
            
    return improv

########################################
## Fast 2-update (i.e. 2x2-block update)
@nb.njit(fastmath=True)
def _block2_update_jit(S, Q, R, grad, i, j):
    """Updates Q, R, grad in-place and returns float scalar 'improv'."""
    S_ii, S_jj, S_ij = S[i, i], S[j, j], S[i, j]
    R_ii, R_jj, R_ij = R[i, i], R[j, j], R[i, j]
    
    det_S_sub = S_ii * S_jj - S_ij * S_ij
    det_R_sub = R_ii * R_jj - R_ij * R_ij
    
    inv_det_R = 1.0 / det_R_sub
    r00, r01, r11 = R_jj * inv_det_R, -R_ij * inv_det_R, R_ii * inv_det_R
    inv_det_S = 1.0 / det_S_sub
    
    Q[i, i] += S_jj * inv_det_S - r00
    Q[j, j] += S_ii * inv_det_S - r11
    q_ij_diff = S_ij * inv_det_S + r01
    Q[i, j] -= q_ij_diff
    Q[j, i] = Q[i, j]
    
    s00 = S_ii * r00 + S_ij * r01
    s01 = S_ii * r01 + S_ij * r11
    s10 = S_ij * r00 + S_jj * r01
    s11 = S_ij * r01 + S_jj * r11
    
    e00, e01 = 1.0 - s00, -s01
    e10, e11 = -s10, 1.0 - s11
    
    m00 = r00 * e00 + r01 * e10
    m01 = r00 * e01 + r01 * e11
    m10 = r01 * e00 + r11 * e10
    m11 = r01 * e01 + r11 * e11
    
    d = R.shape[0]
    Ri = R[i, :].copy()
    Rj = R[j, :].copy()
    
    for k in range(d):
        v0 = m00 * Ri[k] + m01 * Rj[k]
        v1 = m10 * Ri[k] + m11 * Rj[k]
        for m in range(d):
            delta = Ri[m] * v0 + Rj[m] * v1
            R[m, k] -= delta
            grad[m, k] += delta

    for k in range(d):
        r0 = s00 * Ri[k] + s01 * Rj[k]
        r1 = s10 * Ri[k] + s11 * Rj[k]
        
        grad[i, k] += R[i, k] - r0
        grad[k, i] += R[k, i] - r0
        grad[j, k] += R[j, k] - r1
        grad[k, j] += R[k, j] - r1
        
        R[i, k] = r0
        R[k, i] = r0
        R[j, k] = r1
        R[k, j] = r1
        
    R[i, i], R[j, j] = S_ii, S_jj
    R[i, j], R[j, i] = S_ij, S_ij
    grad[i, i], grad[j, j] = 0.0, 0.0
    grad[i, j], grad[j, i] = 0.0, 0.0

    return (s00 + s11) - 2.0 - np.log(det_S_sub) + np.log(det_R_sub)

##################    
### GS ###########   
##################
## Fast GS solver
@nb.njit(fastmath=True)
def _gs_solver_jit(S, max_iter, activ_mul, activ_add, tau_improv):
    dim = S.shape[0]
    diag_S = np.diag(S)
    
    Q = np.diag(1.0 / diag_S)
    R = np.diag(diag_S)
    grad = S - R
    
    evo_edge_i = np.empty(max_iter, dtype=np.int64)
    evo_edge_j = np.empty(max_iter, dtype=np.int64)
    evo_inner_up = np.empty(max_iter, dtype=np.int64)
    
    active_mask = np.zeros((dim, dim), dtype=nb.boolean)
    
    for k in range(max_iter):
        best_score = -1.0
        best_i, best_j = 0, 0
        
        for i in range(dim):
            for j in range(i + 1, dim):
                if active_mask[i, j]:
                    continue
                g = grad[i, j]
                score = 2.0 * g * g 
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j
                    
        improv_out = _block2_update_jit(S, Q, R, grad, best_i, best_j)
        
        evo_edge_i[k] = best_i
        evo_edge_j[k] = best_j
        active_mask[best_i, best_j] = True
        active_mask[best_j, best_i] = True
        
        n_off_activ = k + 1
        l_max = int(activ_mul * n_off_activ + activ_add)
        
        nb_inner_up = 1
        for l in range(l_max):
            best_act_score = -1.0
            best_act_type = 0  # 0: diag, 1: off-diag
            best_act_i, best_act_j = 0, 0
            
            for i in range(dim):
                g = grad[i, i]
                score = (g * g) / (R[i, i]**2)
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 0
                    best_act_i = i
                    
            for idx in range(n_off_activ):
                ai, aj = evo_edge_i[idx], evo_edge_j[idx]
                g = grad[ai, aj]
                lip = R[ai, ai] * R[aj, aj] + R[ai, aj]**2
                score = 2.0 * g * g / lip
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 1
                    best_act_i, best_act_j = ai, aj
                    
            # --- Single variable assignment (In-place update) ---
            if best_act_type == 0:
                improv_in = _diag_update_jit(S, Q, R, grad, best_act_i)
            else:
                improv_in = _block2_update_jit(S, Q, R, grad, best_act_i, best_act_j)
                
            if improv_in > 0:
                nb_inner_up += 1
                
            if improv_in < tau_improv * improv_out:
                break
                
        evo_inner_up[k] = nb_inner_up

    # Return statement returning a 3-tuple
    return evo_edge_i, evo_edge_j, evo_inner_up

###################################################
## GS solver Python wrapper + edge-selection check 
def GS_solver(S,                    # sample covariance
              max_iter=10,          # max iter (outer loop)
              true_P=None,          # true precision
              activ_mul=0.3,        # multiplicative cst for inner loop (FC)
              activ_add=10,         # additive cst for inner loop (FC)
              tau_improv=1e-3):     # treshold for inner loop (FC)

    # Force C-contiguous float64 matrix for Numba
    S_arr = np.ascontiguousarray(S, dtype=np.float64)
    
    # Execute compiled Numba kernel
    evo_edge_i, evo_edge_j, evo_inner_up = _gs_solver_jit(
        S_arr, int(max_iter), float(activ_mul), float(activ_add), float(tau_improv)
    )
    
    # Format Python output dictionary
    evo_edge = list(zip(evo_edge_i.tolist(), evo_edge_j.tolist()))

    # Check edge selection
    edge_is_true = None
    if true_P is not None:
        edge_is_true = [bool(true_P[i, j] != 0) for i, j in evo_edge]

    return {
        'evo_edge': evo_edge,
        'edge_is_true': edge_is_true,
        'evo_inner_up': evo_inner_up.tolist()
    }


##################    
### GSL ##########   
##################
## Fast GSL solver
@nb.njit(fastmath=True)
def _gsl_solver_jit(S, max_iter, activ_mul, activ_add, tau_improv):
    dim = S.shape[0]
    diag_S = np.diag(S)
    
    Q = np.diag(1.0 / diag_S)
    R = np.diag(diag_S)
    grad = S - R
    
    evo_edge_i = np.empty(max_iter, dtype=np.int64)
    evo_edge_j = np.empty(max_iter, dtype=np.int64)
    evo_inner_up = np.empty(max_iter, dtype=np.int64)
    
    active_mask = np.zeros((dim, dim), dtype=nb.boolean)
    
    for k in range(max_iter):
        best_score = -1.0
        best_i, best_j = 0, 0
        
        for i in range(dim):
            R_ii = R[i, i]
            for j in range(i + 1, dim):
                if active_mask[i, j]:
                    continue
                g = grad[i, j]
                lip = R_ii * R[j, j] + R[i, j]**2
                score = 2.0 * g * g / lip
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j
                    
        improv_out = _block2_update_jit(S, Q, R, grad, best_i, best_j)
        
        evo_edge_i[k] = best_i
        evo_edge_j[k] = best_j
        active_mask[best_i, best_j] = True
        active_mask[best_j, best_i] = True
        
        n_off_activ = k + 1
        l_max = int(activ_mul * n_off_activ + activ_add)
        
        nb_inner_up = 1
        for l in range(l_max):
            best_act_score = -1.0
            best_act_type = 0  # 0: diag, 1: off-diag
            best_act_i, best_act_j = 0, 0
            
            for i in range(dim):
                g = grad[i, i]
                score = (g * g) / (R[i, i]**2)
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 0
                    best_act_i = i
                    
            for idx in range(n_off_activ):
                ai, aj = evo_edge_i[idx], evo_edge_j[idx]
                g = grad[ai, aj]
                lip = R[ai, ai] * R[aj, aj] + R[ai, aj]**2
                score = 2.0 * g * g / lip
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 1
                    best_act_i, best_act_j = ai, aj
                    
            # --- Single variable assignment (In-place update) ---
            if best_act_type == 0:
                improv_in = _diag_update_jit(S, Q, R, grad, best_act_i)
            else:
                improv_in = _block2_update_jit(S, Q, R, grad, best_act_i, best_act_j)
                
            if improv_in > 0:
                nb_inner_up += 1
                
            if improv_in < tau_improv * improv_out:
                break
                
        evo_inner_up[k] = nb_inner_up

    # Return statement returning a 3-tuple
    return evo_edge_i, evo_edge_j, evo_inner_up

###################################################
## GSL solver Python wrapper + edge-selection check 
def GSL_solver(S,                    # sample covariance
               max_iter=10,          # max iter (outer loop)
               true_P=None,          # true precision
               activ_mul=0.3,        # multiplicative cst for inner loop (FC)
               activ_add=10,         # additive cst for inner loop (FC)
               tau_improv=1e-3):     # treshold for inner loop (FC)

    # Force C-contiguous float64 matrix for Numba
    S_arr = np.ascontiguousarray(S, dtype=np.float64)
    
    # Execute compiled Numba kernel
    evo_edge_i, evo_edge_j, evo_inner_up = _gsl_solver_jit(
        S_arr, int(max_iter), float(activ_mul), float(activ_add), float(tau_improv)
    )
    
    # Format Python output dictionary
    evo_edge = list(zip(evo_edge_i.tolist(), evo_edge_j.tolist()))

    # Check edge selection
    edge_is_true = None
    if true_P is not None:
        edge_is_true = [bool(true_P[i, j] != 0) for i, j in evo_edge]

    return {
        'evo_edge': evo_edge,
        'edge_is_true': edge_is_true,
        'evo_inner_up': evo_inner_up.tolist()
    }


#############################    
### BBI #####################   
#############################
## Compute 2-block improvemnt
@nb.njit(fastmath=True)
def _calc_bbi_score(S_ii, S_jj, S_ij, R_ii, R_jj, R_ij):
    """
    Computes the exact 2-block improvement score I_S^B(Q, (i,j)) in O(1) time
    """
    det_S_sub = S_ii * S_jj - S_ij * S_ij
    det_R_sub = R_ii * R_jj - R_ij * R_ij
    
    inv_det_R = 1.0 / det_R_sub
    
    # trace(S_sub @ R_sub_inv)
    tr_SxRinv = (S_ii * R_jj - 2.0 * S_ij * R_ij + S_jj * R_ii) * inv_det_R
    
    # I_S^B(Q, (i,j)) = trace(S_sub @ R_sub_inv) - 2 - log(det(S_sub)) + log(det(R_sub))
    return tr_SxRinv - 2.0 - np.log(det_S_sub) + np.log(det_R_sub)

##################
## Fast BBI solver
@nb.njit(fastmath=True)
def _bbi_solver_jit(S, max_iter, activ_mul, activ_add, tau_improv):
    """
    Core compiled kernel for Algorithm 3 using the BBI selection rule.
    """
    dim = S.shape[0]
    diag_S = np.diag(S)
    
    # Initialization (Section 2.2, Eq. 2.3)
    Q = np.diag(1.0 / diag_S)
    R = np.diag(diag_S)
    grad = S - R
    
    evo_edge_i = np.empty(max_iter, dtype=np.int64)
    evo_edge_j = np.empty(max_iter, dtype=np.int64)
    evo_inner_up = np.empty(max_iter, dtype=np.int64)
    
    active_mask = np.zeros((dim, dim), dtype=nb.boolean)
    
    # -------------------------------------------------------------------------
    # OUTER LOOP (Edge Activation via BBI Score)
    # -------------------------------------------------------------------------
    for k in range(max_iter):
        best_score = -1e15
        best_i, best_j = 0, 0
        
        # BBI Selection Rule: argmax_{free(Q)} I_S^B(Q, (i,j))
        for i in range(dim):
            S_ii, R_ii = S[i, i], R[i, i]
            for j in range(i + 1, dim):
                if active_mask[i, j]:
                    continue
                
                # Compute exact 2-block improvement score
                score = _calc_bbi_score(S_ii, S[j, j], S[i, j], R_ii, R[j, j], R[i, j])
                
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j
                    
        # Perform Block-2 update for activated edge
        improv_out = _block2_update_jit(S, Q, R, grad, best_i, best_j)
        
        evo_edge_i[k] = best_i
        evo_edge_j[k] = best_j
        active_mask[best_i, best_j] = True
        active_mask[best_j, best_i] = True
        
        n_off_activ = k + 1
        l_max = int(activ_mul * n_off_activ + activ_add)
        
        # ---------------------------------------------------------------------
        # INNER LOOP (Algorithm 2 Refinement / Approximate Full Correction)
        # ---------------------------------------------------------------------
        nb_inner_up = 1
        for l in range(l_max):
            best_act_score = -1e15
            best_act_type = 0  # 0: diagonal candidate, 1: active off-diagonal
            best_act_i, best_act_j = 0, 0
            
            # 1. Diagonal Candidates
            for i in range(dim):
                g = grad[i, i]
                score = (g * g) / (R[i, i]**2)
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 0
                    best_act_i = i
                    
            # 2. Active Off-Diagonal Candidates (using BBI score)
            for idx in range(n_off_activ):
                ai, aj = evo_edge_i[idx], evo_edge_j[idx]
                score = _calc_bbi_score(S[ai, ai], S[aj, aj], S[ai, aj], R[ai, ai], R[aj, aj], R[ai, aj])
                if score > best_act_score:
                    best_act_score = score
                    best_act_type = 1
                    best_act_i, best_act_j = ai, aj
                    
            if best_act_type == 0:
                improv_in = _diag_update_jit(S, Q, R, grad, best_act_i)
            else:
                improv_in = _block2_update_jit(S, Q, R, grad, best_act_i, best_act_j)
                
            if improv_in > 0:
                nb_inner_up += 1
                
            if improv_in < tau_improv * improv_out:
                break
                
        evo_inner_up[k] = nb_inner_up

    return evo_edge_i, evo_edge_j, evo_inner_up

###################################################
## BBI solver Python wrapper + edge-selection check 
def BBI_solver(S,                 # sample covariance
               max_iter=10,       # max iter (outer loop)
               true_P=None,       # true precision
               activ_mul=0.3,     # multiplicative cst for inner loop (FC)
               activ_add=10,      # additive cst for inner loop (FC)
               tau_improv=1e-3):  # treshold for inner loop (FC)    

    S_arr = np.ascontiguousarray(S, dtype=np.float64)
    
    evo_edge_i, evo_edge_j, evo_inner_up = _bbi_solver_jit(
        S_arr, int(max_iter), float(activ_mul), float(activ_add), float(tau_improv)
    )
    
    evo_edge = list(zip(evo_edge_i.tolist(), evo_edge_j.tolist()))
    
    edge_is_true = None
    if true_P is not None:
        edge_is_true = [bool(true_P[i, j] != 0) for i, j in evo_edge]

    return {
        'evo_edge': evo_edge,
        'edge_is_true': edge_is_true,
        'evo_inner_up': evo_inner_up.tolist()
    }


###################    
### BFCI ##########   
###################   
## Fast BFCI solver
@nb.njit(fastmath=True)
def _bfci_solver_jit(S, max_iter, activ_mul, activ_add, tau_improv):
    """
    Core JIT kernel for BFCI (Algorithm 1 using approximate full correction).
    """
    dim = S.shape[0]
    diag_S = np.diag(S)
    
    # Initial diagonal matrix Q_0 and inverse R_0
    Q = np.diag(1.0 / diag_S)
    R = np.diag(diag_S)
    grad = S - R
    
    evo_edge_i = np.empty(max_iter, dtype=np.int64)
    evo_edge_j = np.empty(max_iter, dtype=np.int64)
    evo_inner_up = np.empty(max_iter, dtype=np.int64)
    
    active_mask = np.zeros((dim, dim), dtype=nb.boolean)
    
    # Buffers to preserve state of the winning candidate across free-set trial runs
    Q_best = np.empty_like(Q)
    R_best = np.empty_like(R)
    grad_best = np.empty_like(grad)

    # -------------------------------------------------------------------------
    # OUTER LOOP (BFCI Edge Activation)
    # -------------------------------------------------------------------------
    for k in range(max_iter):
        best_cand_imp = -1e15
        best_i, best_j = 0, 0
        best_nb_inner_up = 0
        
        # Test every candidate edge (i,j) in the free set
        for i in range(dim):
            for j in range(i + 1, dim):
                if active_mask[i, j]:
                    continue
                
                # Copy current state matrices for temporary trial refinement
                Q_tmp = Q.copy()
                R_tmp = R.copy()
                grad_tmp = grad.copy()
                
                # 1. Initial 2-block update on candidate edge (i,j)
                imp_out = _block2_update_jit(S, Q_tmp, R_tmp, grad_tmp, i, j)
                cand_total_imp = imp_out
                
                n_off_activ = k + 1
                l_max = int(activ_mul * n_off_activ + activ_add)
                
                nb_inner = 1
                
                # 2. Approximate full correction on ACTIV U {(i,j)}
                for l in range(l_max):
                    best_act_score = -1e15
                    best_act_type = 0  # 0: diag, 1: off-diag
                    best_act_i, best_act_j = 0, 0
                    
                    # Evaluate diagonal candidates
                    for d_idx in range(dim):
                        g = grad_tmp[d_idx, d_idx]
                        score = (g * g) / (R_tmp[d_idx, d_idx]**2)
                        if score > best_act_score:
                            best_act_score = score
                            best_act_type = 0
                            best_act_i = d_idx
                            
                    # Evaluate active off-diagonal candidates (0 .. k-1)
                    for idx in range(k):
                        ai, aj = evo_edge_i[idx], evo_edge_j[idx]
                        g = grad_tmp[ai, aj]
                        lip = R_tmp[ai, ai] * R_tmp[aj, aj] + R_tmp[ai, aj]**2
                        score = 2.0 * g * g / lip
                        if score > best_act_score:
                            best_act_score = score
                            best_act_type = 1
                            best_act_i, best_act_j = ai, aj
                            
                    # Evaluate candidate edge (i,j) itself
                    g_cand = grad_tmp[i, j]
                    lip_cand = R_tmp[i, i] * R_tmp[j, j] + R_tmp[i, j]**2
                    score_cand = 2.0 * g_cand * g_cand / lip_cand
                    if score_cand > best_act_score:
                        best_act_score = score_cand
                        best_act_type = 1
                        best_act_i, best_act_j = i, j
                        
                    # Execute best coordinate update
                    if best_act_type == 0:
                        imp_in = _diag_update_jit(S, Q_tmp, R_tmp, grad_tmp, best_act_i)
                    else:
                        imp_in = _block2_update_jit(S, Q_tmp, R_tmp, grad_tmp, best_act_i, best_act_j)
                        
                    if imp_in > 0:
                        cand_total_imp += imp_in
                        nb_inner += 1
                        
                    if imp_in < tau_improv * imp_out:
                        break
                        
                # Keep candidate achieving the highest total fully-corrective improvement
                if cand_total_imp > best_cand_imp:
                    best_cand_imp = cand_total_imp
                    best_i, best_j = i, j
                    best_nb_inner_up = nb_inner
                    Q_best[:] = Q_tmp[:]
                    R_best[:] = R_tmp[:]
                    grad_best[:] = grad_tmp[:]

        # Advance state to the winning candidate
        Q[:] = Q_best[:]
        R[:] = R_best[:]
        grad[:] = grad_best[:]
        
        evo_edge_i[k] = best_i
        evo_edge_j[k] = best_j
        active_mask[best_i, best_j] = True
        active_mask[best_j, best_i] = True
        evo_inner_up[k] = best_nb_inner_up

    return evo_edge_i, evo_edge_j, evo_inner_up

####################################################
## BFCI solver Python wrapper + edge-selection check 
def BFCI_solver(S, 
                max_iter=10, 
                true_P=None, 
                activ_mul=1, 
                activ_add=10, 
                tau_improv=1e-3):
    """
    Wrapper for the BFCI variant of Algorithm 3.
    """
    S_arr = np.ascontiguousarray(S, dtype=np.float64)
    
    evo_edge_i, evo_edge_j, evo_inner_up = _bfci_solver_jit(
        S_arr, int(max_iter), float(activ_mul), float(activ_add), float(tau_improv)
    )
    
    evo_edge = list(zip(evo_edge_i.tolist(), evo_edge_j.tolist()))
    
    edge_is_true = None
    if true_P is not None:
        edge_is_true = [bool(true_P[i, j] != 0) for i, j in evo_edge]

    return {
        'evo_edge': evo_edge,
        'edge_is_true': edge_is_true,
        'evo_inner_up': evo_inner_up.tolist()
    }


###################    
### Glasso ########   
###################   
def Glasso_ROC(S, true_P, n_glasso):
    """
    Computes the regularisation path for Graphical Lasso using GGlasso across 
    uniformly-spaced lambda values in [0, max_{i != j} |S_{i,j}|] (Remark 6.1).
    Silences ADMM termination messages cleanly.
    """
    d = S.shape[0]
    
    # 1. Compute total true off-diagonal edges and non-edges
    true_off_mask = np.triu(true_P != 0, k=1)
    nb_true_edges = np.sum(true_off_mask)
    total_off_diag = d * (d - 1) // 2
    nb_true_non_edges = total_off_diag - nb_true_edges

    # 2. Discretize interval [0, max_{i != j} |S_{i,j}|] (Remark 6.1)
    S_off = S - np.diag(np.diag(S))
    lambda_max = np.max(np.abs(S_off))
    lambdas = np.linspace(lambda_max, 0, n_glasso)[:-1] # remove 0  
    n_glasso = n_glasso - 1 
    
    precision = np.zeros(n_glasso)
    recall = np.zeros(n_glasso)
    fpr = np.zeros(n_glasso)

    # Divert standard output to buffer
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        for idx, lmb in enumerate(lambdas):
            p = glasso_problem(S=S, N=100, # sample size N not used in the experiment 
                               reg_params={'lambda1': float(lmb)}, latent=False) # sample size not used in the  
            p.solve(verbose=False, tol=1e-6)  # request relatively high precision
            
            Theta_est = p.solution.precision_

            # Extract estimated active edges (upper triangle) --- discard numerical errors.  
            est_off_mask = np.triu(np.abs(Theta_est) > 1e-5, k=1) 
            
            # Calculate True Positives (TP) and False Positives (FP)
            tp = np.sum(est_off_mask & true_off_mask)
            fp = np.sum(est_off_mask & (~true_off_mask))
            total_est = tp + fp

            # Metrics
            precision[idx] = tp / total_est if total_est > 0 else 1.0
            recall[idx] = tp / nb_true_edges if nb_true_edges > 0 else 0.0
            fpr[idx] = fp / nb_true_non_edges if nb_true_non_edges > 0 else 0.0
            
    finally:
        # Always restore original stdout
        sys.stdout = old_stdout

    return precision*100, recall*100, fpr*100



