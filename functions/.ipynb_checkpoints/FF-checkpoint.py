import numpy as np


def FF(returns, factRet, lambda_, K):
    """
    Calibrate the Fama-French 3-factor model.

    Inputs:
        returns : T x n matrix of asset excess returns
        factRet : T x 8 DataFrame of factor returns, excluding RF
        lambda_ : not used for FF
        K       : not used for FF

    Outputs:
        mu : n-vector of expected excess returns
        Q  : n x n covariance matrix
    """

    # Convert returns to NumPy array
    returns = np.asarray(returns, dtype=float)

    # Select Fama-French factors
    F = factRet[["Mkt_RF", "SMB", "HML"]].to_numpy()

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