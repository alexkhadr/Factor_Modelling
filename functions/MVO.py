import cvxpy as cp
import numpy as np


def MVO(mu, Q, targetRet):
    """
    Use this function to construct the MVO portfolio subject to a target
    return constraint, with short sales disallowed.

    The optimization problem is:

        minimize     x'Qx
        subject to   mu'x >= targetRet
                     sum(x) = 1
                     x >= 0

    where x is the vector of portfolio weights.
    """

    # Find the total number of assets
    n = len(mu)

    # Convert inputs to NumPy arrays
    mu = np.asarray(mu).reshape(-1)
    Q = np.asarray(Q)

    # Define portfolio weights
    x_var = cp.Variable(n)

    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(x_var, Q))

    # Constraints:
    # 1. Portfolio expected return must meet target return
    # 2. Fully invested portfolio
    # 3. No short selling
    constraints = [
        mu @ x_var >= targetRet,
        cp.sum(x_var) == 1,
        x_var >= 0
    ]

    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"  MVO warning: '{problem.status}' — falling back to min-variance")
        fallback = cp.Problem(cp.Minimize(cp.quad_form(x_var, Q)),
                            [cp.sum(x_var) == 1, x_var >= 0])
        fallback.solve(solver=cp.CLARABEL, verbose=False)
        if fallback.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MVO fallback also failed: {fallback.status}")

    # Extract optimal weights
    x = x_var.value

    return x
