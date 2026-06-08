"""
classic_svdd.py

Classic kernel SVDD (Tax & Duin, 2004) implemented via the dual QP.

Finds the minimum enclosing hypersphere in kernel space such that at
most a fraction C of training points fall outside.

The dual problem is:
    maximize  sum_i(alpha_i * K_ii) - sum_ij(alpha_i * alpha_j * K_ij)
    subject to  sum(alpha_i) = 1
                0 <= alpha_i <= C

Decision function: distance from test point to sphere center in kernel
space. Points inside the sphere score <= 0 (accepted), points outside
score > 0 (rejected).

Reference:
    Tax & Duin (2004). Support Vector Data Description.
    Machine Learning, 54(1), 45-66.
"""

import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel


class ClassicSVDD:
    """
    Classic kernel SVDD.

    Parameters
    ----------
    C : float
        Soft-margin parameter. Equivalent to 1/(n*nu) in OCSVM terms.
        Fraction of training points allowed outside the sphere is at
        most C. Default 0.1.
    gamma : float or 'scale'
        RBF kernel bandwidth. 'scale' sets gamma = 1 / (n_features * X.var()).
        Default 'scale'.
    """

    def __init__(self, C=0.1, gamma='scale'):
        self.C        = C
        self.gamma    = gamma
        self._alpha   = None   # dual variables (n_train,)
        self._sv_mask = None   # support vector mask
        self._X_train = None   # training data
        self._gamma   = None   # resolved gamma value
        self._R2      = None   # sphere radius squared
        self._offset  = None   # K(c,c) - used in decision function

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        n, d = X.shape

        # Resolve gamma
        if self.gamma == 'scale':
            self._gamma = 1.0 / (d * X.var()) if X.var() > 0 else 1.0
        else:
            self._gamma = float(self.gamma)

        K = rbf_kernel(X, gamma=self._gamma)   # (n, n)

        # Dual QP — psd_wrap tells CVXPY to trust K is PSD without numerical
        # verification, which can fail for large RBF matrices due to float precision
        alpha = cp.Variable(n)
        diag_K = np.diag(K)
        objective   = cp.Maximize(diag_K @ alpha - cp.quad_form(alpha, cp.psd_wrap(K)))
        constraints = [cp.sum(alpha) == 1, alpha >= 0, alpha <= self.C]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f"SVDD QP did not converge: {prob.status}")

        alpha_val = np.array(alpha.value).ravel()
        alpha_val = np.clip(alpha_val, 0, self.C)

        self._alpha   = alpha_val
        self._X_train = X
        # Support vectors: 0 < alpha < C
        self._sv_mask = (alpha_val > 1e-6) & (alpha_val < self.C - 1e-6)

        # Radius: R^2 = K(sv, sv) - 2*sum_j(alpha_j*K(sv_j, sv))
        #                          + sum_ij(alpha_i*alpha_j*K(xi,xj))
        # Compute as decision_function of any support vector (should be ~0)
        self._offset  = float(alpha_val @ K @ alpha_val)

        # R^2 = distance of a support vector to center
        if self._sv_mask.any():
            sv_idx = np.where(self._sv_mask)[0][0]
            self._R2 = self._dist2_to_center(X[sv_idx:sv_idx+1], K[sv_idx])[0]
        else:
            # Fallback: use max distance among all training points
            dists = self._dist2_to_center(X, K)
            self._R2 = float(np.max(dists))

        return self

    # ── Internal distance computation ─────────────────────────────────────

    def _dist2_to_center(self, X_test, K_train_test=None):
        """
        Compute squared distance from each test point to sphere center.
        dist^2(x) = K(x,x) - 2*sum_i(alpha_i*K(xi,x)) + sum_ij(alpha_i*alpha_j*K(xi,xj))
        The last term is self._offset (constant, precomputed).
        """
        if K_train_test is None:
            K_cross = rbf_kernel(self._X_train, X_test, gamma=self._gamma)  # (n_train, n_test)
        else:
            K_cross = K_train_test.reshape(-1, 1) if K_train_test.ndim == 1 else K_train_test

        K_xx   = np.ones(len(X_test))   # RBF kernel K(x,x) = 1 for any x
        term2  = 2.0 * (self._alpha @ K_cross)   # (n_test,)
        dist2  = K_xx - term2 + self._offset
        return dist2

    # ── Decision function ─────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Returns dist^2(x) - R^2 for each point.
        Negative = inside sphere (accepted / inlier).
        Positive = outside sphere (rejected / outlier).
        """
        X = np.array(X, dtype=float)
        K_cross = rbf_kernel(self._X_train, X, gamma=self._gamma)
        dist2   = self._dist2_to_center(X, K_cross)
        return dist2 - self._R2

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(self, X):
        """Returns 1 (inlier/accepted) or -1 (outlier/rejected)."""
        scores = self.decision_function(X)
        return np.where(scores <= 0, 1, -1)