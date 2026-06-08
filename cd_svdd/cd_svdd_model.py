import numpy as np
import torch

from backbone import CDSVDDNetwork
from dual_solver import solve_dual

class CDSVDD:
    def __init__(self, input_dim=48, hidden_dim=32, output_dim=8,
                 nu=0.1, lr=1e-3, weight_decay=1e-6, n_epochs=50,
                 qp_subsample=256):
        self.nu = nu
        self.n_epochs = n_epochs
        self.qp_subsample = qp_subsample
        self.net = CDSVDDNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.c = None
        self.R = None

    def fit(self, X):
        if len(X) < 20:
            print(f"  Warning: only {len(X)} training windows — results may be unreliable.", flush=True)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        n = len(X)

        for epoch in range(self.n_epochs):
            print(f"  Epoch {epoch+1}/{self.n_epochs} — solving QP...", flush=True)
            self.net.eval()
            with torch.no_grad():
                phi_all = self.net(X_tensor).numpy()

            if n > self.qp_subsample:
                idx = np.random.choice(n, self.qp_subsample, replace=False)
                phi_sub = phi_all[idx]
            else:
                idx = np.arange(n)
                phi_sub = phi_all

            c, R, alpha = solve_dual(phi_sub, self.nu)
            self.c = c
            self.R = R

            print(f"  Epoch {epoch+1}/{self.n_epochs} — updating network...", flush=True)
            self.net.train()
            self.optimizer.zero_grad()
            phi_t = self.net(X_tensor)
            c_t = torch.tensor(c, dtype=torch.float32)
            loss = self._loss(phi_t, c_t, R, alpha, idx, n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            print(f"  Epoch {epoch+1}/{self.n_epochs} — loss={loss.item():.6f}  R={R:.6f}", flush=True)

    def _loss(self, phi, c, R, alpha, idx, n):
        nu = self.nu
        n_sub = len(idx)
        upper = 1.0 / (nu * n_sub)
        tol = 1e-4 * upper

        phi_sub = phi[idx]
        dist_sq_sub = torch.sum((phi_sub - c) ** 2, dim=1)

        sv_mask = torch.tensor(
            (alpha > tol) & (alpha < upper - tol), dtype=torch.bool
        )
        if sv_mask.sum() == 0:
            sv_mask = torch.tensor(alpha > tol, dtype=torch.bool)

        radius_term = dist_sq_sub[sv_mask].mean()

        dist_sq_all = torch.sum((phi - c) ** 2, dim=1)
        penalty = torch.mean(torch.clamp(dist_sq_all - R, min=0.0)) / nu

        return radius_term + penalty

    def score(self, X):
        if self.c is None or self.R is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        self.net.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            phi = self.net(X_tensor).numpy()
        dist_sq = np.sum((phi - self.c) ** 2, axis=1)
        return dist_sq - self.R

    def predict(self, X):
        scores = self.score(X)
        return np.where(scores <= 0, 1, -1)