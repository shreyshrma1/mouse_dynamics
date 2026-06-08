import numpy as np

def solve_dual(phi, nu):
    import cvxpy as cp

    n = phi.shape[0]
    upper = 1.0 / (nu * n)

    Q = phi @ phi.T
    Q = (Q + Q.T) / 2

    jitter = 0
    for j in [1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        try:
            np.linalg.cholesky(Q + j * np.eye(n))
            jitter = j
            break
        except np.linalg.LinAlgError:
            continue

    if jitter == 0 and not _is_pd(Q):
        raise RuntimeError("Q is not positive definite even after jittering.")

    Q_jit = Q + jitter * np.eye(n)
    L = np.linalg.cholesky(Q_jit)

    alpha = cp.Variable(n)
    objective = cp.Minimize(
        cp.sum_squares(L.T @ alpha) - alpha @ np.diag(Q_jit)
    )
    constraints = [
        alpha >= 0,
        alpha <= upper,
        cp.sum(alpha) == 1,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if alpha.value is None:
        raise RuntimeError("QP solver failed to find a solution.")

    alpha_val = np.array(alpha.value).flatten()
    alpha_val = np.clip(alpha_val, 0, upper)

    c = alpha_val @ phi

    tol = 1e-4 * upper
    sv_mask = (alpha_val > tol) & (alpha_val < upper - tol)
    if sv_mask.sum() == 0:
        sv_mask = alpha_val > tol

    dists_sq = np.sum((phi[sv_mask] - c) ** 2, axis=1)
    R = float(np.mean(dists_sq))

    return c, R, alpha_val

def _is_pd(Q):
    try:
        np.linalg.cholesky(Q)
        return True
    except np.linalg.LinAlgError:
        return False