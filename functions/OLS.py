import cvxpy as cp
import numpy as np


def OLS(returns, factRet, lambda_, K):
    """
    Use this function to calibrate the OLS 8-factor model.

    Inputs:
        returns : T x n matrix/DataFrame of asset excess returns
        factRet : T x 8 DataFrame of factor returns, excluding RF
        lambda_ : not used for OLS
        K       : not used for OLS

    Outputs:
        mu : n-vector of expected excess returns
        Q  : n x n covariance matrix
    """

    # Convert returns to NumPy array
    returns = np.asarray(returns, dtype=float)

    # Use all 8 factor columns
    F = factRet.to_numpy()

    T, n = returns.shape

    # Add intercept column
    X = np.column_stack((np.ones(T), F))

    # OLS coefficients: B = (X'X)^(-1)X'Y
    B = np.linalg.inv(X.T @ X) @ X.T @ returns

    # Separate alpha and betas
    alpha = B[0, :]
    beta = B[1:, :]

    # Expected excess returns including alpha
    mu = alpha + np.mean(F, axis=0) @ beta

    # Residuals
    residuals = returns - X @ B

    # Factor covariance matrix
    factor_cov = np.cov(F, rowvar=False)

    # Residual covariance matrix
    residual_cov = np.diag(np.var(residuals, axis=0, ddof=1))

    # Asset covariance matrix
    Q = beta.T @ factor_cov @ beta + residual_cov

    return mu, Q