import itertools
import numpy as np


RIDGE_EPSILON = 1.0e-8


def BSS(returns, factRet, lambda_, K, return_loadings=False):
    """
    Best Subset Selection model using exhaustive search.

    Uses all 8 factors plus the intercept, but allows at most K coefficients
    to be non-zero. K includes the intercept.

    Inputs:
        returns : T x n matrix/DataFrame of asset excess returns
        factRet : T x 8 DataFrame of factor returns, excluding RF
        lambda_ : not used for BSS
        K       : maximum number of non-zero coefficients

    Outputs:
        mu : n-vector of expected excess returns
        Q  : n x n covariance matrix
    """

    print("USING EXHAUSTIVE SEARCH BSS")

    returns = np.asarray(returns, dtype=float)
    F = factRet.to_numpy(dtype=float)

    T, n = returns.shape

    # Add intercept column
    X = np.column_stack((np.ones(T), F))

    # Number of coefficients: 1 intercept + 8 factors
    p = X.shape[1]

    # Store best coefficients for each asset
    B = np.zeros((p, n))

    # Fit BSS separately for each asset
    for j in range(n):

        y = returns[:, j]

        best_sse = np.inf
        best_beta = np.zeros(p)

        # Try every subset size from 1 to K
        for subset_size in range(1, K + 1):
            for subset in itertools.combinations(range(p), subset_size):

                subset = list(subset)

                # Use only selected columns
                X_subset = X[:, subset]

                # OLS on selected subset
                beta_subset = np.linalg.lstsq(X_subset, y, rcond=None)[0]

                # Fitted values and residuals
                fitted = X_subset @ beta_subset
                residuals = y - fitted

                # Sum of squared errors
                sse = float(residuals @ residuals)

                # Keep best subset
                if sse < best_sse:
                    best_sse = sse

                    full_beta = np.zeros(p)
                    full_beta[subset] = beta_subset
                    best_beta = full_beta

        B[:, j] = best_beta

    # Fitted returns and residuals
    fitted_returns = X @ B
    residuals = returns - fitted_returns

    # Factor matrix and loadings
    factor_matrix = X[:, 1:]
    factor_loadings = B[1:, :]

    # Expected excess returns
    alpha = B[0, :]          # (n,)
    Beta  = B[1:, :]         # (p, n)
    f_mean = F.mean(axis=0)  # (p,)  mean of actual factors only
    mu = alpha + Beta.T @ f_mean  # (n,)

    # Factor covariance matrix
    factor_cov = np.cov(factor_matrix, rowvar=False, ddof=1)
    factor_cov = np.atleast_2d(factor_cov)

    # Residual covariance matrix
    residual_variance = np.var(residuals, axis=0, ddof=1)
    residual_cov = np.diag(np.maximum(residual_variance, 0.0))

    # Asset covariance matrix
    Q = factor_loadings.T @ factor_cov @ factor_loadings
    Q = Q + residual_cov

    # Numerical symmetry and tiny ridge term
    Q = 0.5 * (Q + Q.T)
    Q = Q + RIDGE_EPSILON * np.eye(Q.shape[0])

    if return_loadings:
        return mu, Q, B
    return mu, Q
