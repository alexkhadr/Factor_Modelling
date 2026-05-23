import cvxpy as cp
import numpy as np


def LASSO(returns, factRet, lambda_, K, return_loadings=False):
    """
    LASSO factor model using penalized regression.
    
    Solves for each asset i:
        min  ||r_i - X @ B_i||^2  +  lambda * ||B_i||_1
    
    where X = [1 | F] is the design matrix with intercept.
    
    Parameters
    ----------
    returns  : (T, n) DataFrame — excess asset returns
    factRet  : (T, p) DataFrame — factor returns
    lambda_  : float            — L1 penalty (controls sparsity)
    K        : unused           — kept for consistent function signature
    
    Returns
    -------
    mu : (n,)   array — expected excess returns
    Q  : (n, n) array — asset covariance matrix
    """

    T, n = returns.shape
    p    = factRet.shape[1]

    # ── 1. Build design matrix X = [1 | F],  shape (T, p+1) ─────────────────
    F = factRet.values                              # (T, p)
    X = np.hstack([np.ones((T, 1)), F])             # (T, p+1)  intercept + factors
    R = returns.values                              # (T, n)

    # ── 2. Solve LASSO for each asset via cvxpy ───────────────────────────────
    # Store results: each B_i is (p+1,) = [alpha_i, beta_i1, ..., beta_ip]
    B = np.zeros((p + 1, n))

    for i in range(n):
        r_i = R[:, i]                               # (T,) excess returns for asset i

        B_i = cp.Variable(p + 1)                   # [alpha_i, beta_i1, ..., beta_ip]

        residuals  = r_i - X @ B_i                 # (T,)
        sse        = cp.sum_squares(residuals)      # ||r_i - X B_i||^2_2
        l1_penalty = lambda_ * cp.norm1(B_i)        # lambda * ||B_i||_1

        objective = cp.Minimize(sse + l1_penalty)
        problem   = cp.Problem(objective)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"LASSO failed for asset {i}: status = {problem.status}")

        B[:, i] = B_i.value                        # store (p+1,) solution

    # ── 3. Decompose B into intercept (alpha) and factor loadings (beta) ──────
    alpha = B[0, :]                                 # (n,)   intercepts
    Beta  = B[1:, :]                                # (p, n) factor loadings

    # ── 4. Compute residuals and residual variances ───────────────────────────
    # epsilon = R - X @ B,  shape (T, n)
    epsilon    = R - X @ B                          # (T, n)
    sigma2_eps = np.var(epsilon, axis=0, ddof=1)    # (n,)  residual variances

    # ── 5. Estimate expected returns  mu = alpha + Beta' * E[f] ──────────────
    f_mean = factRet.mean().values                  # (p,)  mean of each factor
    mu     = alpha + Beta.T @ f_mean                # (n,)

    # ── 6. Estimate covariance matrix  Q = Beta' * Sigma_f * Beta + D ────────
    # Sigma_f : factor covariance matrix (p, p)
    # D       : diagonal matrix of residual variances (n, n)
    Sigma_f = factRet.cov().values                  # (p, p)
    D       = np.diag(sigma2_eps)                   # (n, n)

    Q = Beta.T @ Sigma_f @ Beta + D                # (n, n)

    # Symmetrize to eliminate floating-point asymmetry
    Q = (Q + Q.T) / 2

    if return_loadings:
        return mu, Q, B
    return mu, Q
