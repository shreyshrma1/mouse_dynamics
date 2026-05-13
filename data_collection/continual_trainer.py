"""
continual_trainer.py

Continuously trains the OCSVM fusion model as new mouse data is collected.
Called by MouseCollector after each session flush.

Architecture:
  - Each new session is feature-extracted (39-feat + Zheng)
  - Features are appended to a replay buffer (deque with max size)
  - Model is retrained on the full replay buffer after each update
  - Trained model + scalers are saved to disk for authentication

The replay buffer prevents catastrophic forgetting — training always
uses all historical data up to buffer_size sessions, not just the latest.

Usage:
    trainer = ContinualTrainer(user_id="shrey")
    collector = MouseCollector(user_id="shrey", flush_interval=300)
    collector.start(trainer=trainer)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from collections import deque
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# add measurements/ to path so we can import extractors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features
from measurements.extract_zheng_features import (load_session, segment_pc_actions,
                                                 compute_curvature_angles, compute_curvature_distances,
                                                 ANGLE_BINS, DIST_BINS, DIST_MAX)

FEATURE_COLS_39 = [
    "type_of_action","traveled_distance_pixel","elapsed_time",
    "direction_of_movement","straightness","num_points","sum_of_angles",
    "mean_curv","sd_curv","max_curv","min_curv",
    "mean_omega","sd_omega","max_omega","min_omega",
    "largest_deviation","dist_end_to_end_line","num_critical_points",
    "mean_vx","sd_vx","max_vx","min_vx",
    "mean_vy","sd_vy","max_vy","min_vy",
    "mean_v","sd_v","max_v","min_v",
    "mean_a","sd_a","max_a","min_a",
    "mean_jerk","sd_jerk","max_jerk","min_jerk","a_beg_time"
]
FEATURE_COLS_ZHENG = [f"ca_{i}" for i in range(180)] + [f"cd_{i}" for i in range(35)]

# minimum windows before training is attempted
MIN_SESSIONS_39 = 3
MIN_SESSIONS_ZHENG = 3


class ContinualTrainer:
    def __init__(
        self,
        user_id,
        save_dir="checkpoints_ocsvm",
        buffer_size=50,
        window_size=200,
        nu=0.05,
        gamma="scale",
        use_kpca=True,
        kpca_kernel="rbf",
        kpca_gamma=None,
        kpca_variance=0.95,
    ):
        """
        Args:
            user_id:      user that is being identified
            save_dir:     directory to save trained models and scalers
            buffer_size:  maximum number of windows to keep in the replay buffer
            window_size:  number of PC/DD actions per window for Zheng histograms
            nu:           OCSVM nu — upper bound on fraction of training data
                          allowed outside the boundary (try 0.05–0.15)
            gamma:        OCSVM RBF kernel width; "scale" lets sklearn compute it
            use_kpca:     whether to apply Kernel PCA before OCSVM
            kpca_kernel:  kernel for KPCA (default "rbf")
            kpca_gamma:   gamma for KPCA kernel (None = sklearn default)
            kpca_variance: fraction of eigenvalue mass to retain (default 0.95)
        """
        self.user_id = user_id
        self.save_dir = save_dir
        self.window_size = window_size
        self.nu = nu
        self.gamma = gamma

        self.use_kpca = use_kpca
        self.kpca_kernel = kpca_kernel
        self.kpca_gamma = kpca_gamma
        self.kpca_variance = kpca_variance

        self.kpca_39 = None
        self.kpca_n_components_39 = None
        self.kpca_zheng = None
        self.kpca_n_components_zheng = None

        os.makedirs(save_dir, exist_ok=True)

        # replay buffers — one entry per window, each a numpy array of feature rows
        self.buffer_39 = deque(maxlen=buffer_size)
        self.buffer_zheng = deque(maxlen=buffer_size)

        # trained models (None until enough data is available)
        self.models_39 = None
        self.scaler_39 = None
        self.models_zheng = None
        self.scaler_zheng = None

        # session counter for logging
        self.n_sessions = 0

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_39(self, session_path):
        """Extract 39 features from a session file, split into windows."""
        try:
            df = extract_session_features(session_path, self.user_id,
                                          window_size=self.window_size)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_39)
            if len(df) == 0:
                return []
            windows = []
            for _, grp in df.groupby("session"):
                x = grp[FEATURE_COLS_39].values
                if len(x) >= 5:
                    windows.append(x)
            return windows
        except Exception as e:
            print(f"[Trainer] 39-feat extraction failed: {e}")
            return []

    def _extract_zheng(self, session_path):
        """Extract Zheng histograms from a session file, one per window."""
        try:
            session_df = load_session(session_path)
            actions = segment_pc_actions(session_df)
            rows = []
            for start in range(0, len(actions), self.window_size):
                window = actions[start: start + self.window_size]
                if len(window) < 5:
                    continue
                all_angles, all_dists = [], []
                for events in window:
                    all_angles.extend(compute_curvature_angles(events).tolist())
                    all_dists.extend(compute_curvature_distances(events).tolist())
                if not all_angles:
                    continue
                angle_hist, _ = np.histogram(
                    np.array(all_angles), bins=ANGLE_BINS,
                    range=(0, 180), density=True)
                dist_hist, _ = np.histogram(
                    np.clip(np.array(all_dists), 0, DIST_MAX),
                    bins=DIST_BINS, range=(0, DIST_MAX), density=True)
                vec = np.concatenate([
                    np.nan_to_num(angle_hist),
                    np.nan_to_num(dist_hist)
                ])
                rows.append(vec)
            print(f"[Trainer] Zheng extracted {len(rows)} windows from {session_path}")
            return rows
        except Exception as e:
            print(f"[Trainer] Zheng extraction failed: {e}")
            return []

    # ── KPCA ─────────────────────────────────────────────────────────────

    def _fit_kpca(self, x_sc):
        """
        Fit KernelPCA retaining enough components to explain kpca_variance
        of the eigenvalue mass.

        Returns:
            kpca, x_kpca, n_components
        """
        if not self.use_kpca:
            return None, x_sc, None

        max_components = min(x_sc.shape[0] - 1, x_sc.shape[1])
        if max_components < 1:
            return None, x_sc, None

        kpca = KernelPCA(
            n_components=max_components,
            kernel=self.kpca_kernel,
            gamma=self.kpca_gamma,
            fit_inverse_transform=False,
        )
        x_full = kpca.fit_transform(x_sc)

        eigvals = getattr(kpca, "eigenvalues_", None)
        if eigvals is None:
            eigvals = getattr(kpca, "lambdas_", None)

        if eigvals is None or np.sum(eigvals) <= 1e-12:
            n_components = x_full.shape[1]
        else:
            explained = np.cumsum(eigvals) / np.sum(eigvals)
            n_components = int(np.searchsorted(explained, self.kpca_variance) + 1)

        n_components = max(1, min(n_components, x_full.shape[1]))
        x_kpca = x_full[:, :n_components]

        return kpca, x_kpca, n_components

    def _transform_kpca(self, kpca, x_sc, n_components):
        """Transform scaled features through fitted KPCA."""
        if kpca is None:
            return x_sc
        return kpca.transform(x_sc)[:, :n_components]

    # ── Training ──────────────────────────────────────────────────────────

    def _train_pipeline(self, x_train):
        """
        Fit StandardScaler -> KPCA -> OCSVM.

        The OCSVM decision function is used directly at inference time:
          positive  = inside boundary  -> ACCEPT
          negative  = outside boundary -> REJECT

        Returns:
            models, scaler, kpca, n_components
        """
        x_train = np.asarray(x_train, dtype=float)

        scaler = StandardScaler()
        x_sc = scaler.fit_transform(x_train)

        kpca, x_proj, n_components = self._fit_kpca(x_sc)

        model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        model.fit(x_proj)

        print(
            f"[Trainer] Model trained with nu={self.nu}, gamma={self.gamma}, "
            f"kpca_components={n_components}, n_samples={len(x_train)}"
        )

        return {"ocsvm": model}, scaler, kpca, n_components

    def _retrain_39(self):
        all_rows = np.vstack(list(self.buffer_39))
        if len(all_rows) < 10:
            return
        (
            self.models_39,
            self.scaler_39,
            self.kpca_39,
            self.kpca_n_components_39,
        ) = self._train_pipeline(all_rows)

        # log training scores so we know where the boundary sits
        x_sc = self.scaler_39.transform(all_rows)
        x_proj = self._transform_kpca(self.kpca_39, x_sc, self.kpca_n_components_39)
        scores = self.models_39["ocsvm"].decision_function(x_proj)
        print(
            f"[Trainer] 39-feat training scores — "
            f"min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}"
        )

    def _retrain_zheng(self):
        x_train = np.vstack(list(self.buffer_zheng))
        if len(x_train) < 2:
            return
        (
            self.models_zheng,
            self.scaler_zheng,
            self.kpca_zheng,
            self.kpca_n_components_zheng,
        ) = self._train_pipeline(x_train)

        x_sc = self.scaler_zheng.transform(x_train)
        x_proj = self._transform_kpca(self.kpca_zheng, x_sc, self.kpca_n_components_zheng)
        scores = self.models_zheng["ocsvm"].decision_function(x_proj)
        print(
            f"[Trainer] Zheng training scores — "
            f"min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}"
        )

    # ── Public interface ──────────────────────────────────────────────────

    def update(self, session_path):
        """Called by MouseCollector after each session flush."""
        self.n_sessions += 1
        print(f"[Trainer] Processing session {self.n_sessions}: {session_path}")

        windows_39 = self._extract_39(session_path)
        windows_zheng = self._extract_zheng(session_path)

        for w in windows_39:
            self.buffer_39.append(w)
        for w in windows_zheng:
            self.buffer_zheng.append(w)

        print(f"[Trainer] Buffer: {len(self.buffer_39)} 39-feat windows, "
              f"{len(self.buffer_zheng)} Zheng windows")

        if len(self.buffer_39) >= MIN_SESSIONS_39:
            self._retrain_39()
        else:
            print(f"[Trainer] Need {MIN_SESSIONS_39 - len(self.buffer_39)} more windows before 39-feat model trains")

        if len(self.buffer_zheng) >= MIN_SESSIONS_ZHENG:
            self._retrain_zheng()
        else:
            print(f"[Trainer] Need {MIN_SESSIONS_ZHENG - len(self.buffer_zheng)} more windows before Zheng model trains")

        self._save()

    def backfill(self, data_dir):
        """Reprocess all existing session files and refill the replay buffer."""
        user_dir = os.path.join(data_dir, self.user_id)
        if not os.path.isdir(user_dir):
            print(f"[Trainer] No data directory found at {user_dir}")
            return

        files = sorted([
            os.path.join(user_dir, f)
            for f in os.listdir(user_dir)
            if f.endswith('.csv')
        ])
        print(f"[Trainer] Backfilling from {len(files)} session files...")

        self.buffer_39.clear()
        self.buffer_zheng.clear()

        for path in files:
            for w in self._extract_39(path):
                self.buffer_39.append(w)
            for w in self._extract_zheng(path):
                self.buffer_zheng.append(w)

        print(f"[Trainer] Backfill complete: {len(self.buffer_39)} 39-feat windows, "
              f"{len(self.buffer_zheng)} Zheng windows")

        if len(self.buffer_39) >= MIN_SESSIONS_39:
            self._retrain_39()
        else:
            print(f"[Trainer] Need {MIN_SESSIONS_39 - len(self.buffer_39)} more 39-feat windows")

        if len(self.buffer_zheng) >= MIN_SESSIONS_ZHENG:
            self._retrain_zheng()
        else:
            print(f"[Trainer] Need {MIN_SESSIONS_ZHENG - len(self.buffer_zheng)} more Zheng windows")

        self._save()

    def score(self, session_path):
        """
        Score a session. Uses the raw OCSVM decision function:
          positive -> inside boundary -> ACCEPT
          negative -> outside boundary -> REJECT

        Returns:
            (fused_score, accepted) or (None, None) if no model is ready
        """
        if self.models_39 is None and self.models_zheng is None:
            print("[Trainer] Models not yet trained — collecting more data")
            return None, None

        scores = []

        if self.models_39 is not None:
            windows_39 = self._extract_39(session_path)
            if windows_39:
                x = np.vstack(windows_39)
                x_sc = self.scaler_39.transform(x)
                x_proj = self._transform_kpca(self.kpca_39, x_sc, self.kpca_n_components_39)
                s = float(self.models_39["ocsvm"].decision_function(x_proj).mean())
                accepted_39 = s >= 0
                print(
                    f"[Trainer] 39-feat score: {s:.4f} → "
                    f"{'ACCEPT' if accepted_39 else 'REJECT'}"
                )
                scores.append(s)

        if self.models_zheng is not None:
            windows_zheng = self._extract_zheng(session_path)
            if windows_zheng:
                x = np.vstack(windows_zheng)
                x_sc = self.scaler_zheng.transform(x)
                x_proj = self._transform_kpca(self.kpca_zheng, x_sc, self.kpca_n_components_zheng)
                s = float(self.models_zheng["ocsvm"].decision_function(x_proj).mean())
                accepted_zh = s >= 0
                print(
                    f"[Trainer] Zheng score:   {s:.4f} → "
                    f"{'ACCEPT' if accepted_zh else 'REJECT'}"
                )
                scores.append(s)

        if not scores:
            return None, None

        fused = float(np.mean(scores))
        accepted = fused >= 0
        print(f"[Trainer] Fused score: {fused:+.4f} → {'ACCEPTED' if accepted else 'REJECTED'}")
        return fused, accepted

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        """Save models, scalers, KPCA objects, and replay buffers to disk."""
        path = os.path.join(self.save_dir, self.user_id)
        os.makedirs(path, exist_ok=True)

        if self.models_39 is not None:
            joblib.dump(self.models_39, os.path.join(path, "models_39.pkl"))
            joblib.dump(self.scaler_39, os.path.join(path, "scaler_39.pkl"))
        if self.models_zheng is not None:
            joblib.dump(self.models_zheng, os.path.join(path, "models_zheng.pkl"))
            joblib.dump(self.scaler_zheng, os.path.join(path, "scaler_zheng.pkl"))
        if self.kpca_39 is not None:
            joblib.dump(self.kpca_39, os.path.join(path, "kpca_39.pkl"))
        if self.kpca_zheng is not None:
            joblib.dump(self.kpca_zheng, os.path.join(path, "kpca_zheng.pkl"))

        joblib.dump({
            "buffer_39":              list(self.buffer_39),
            "buffer_zheng":           list(self.buffer_zheng),
            "n_sessions":             self.n_sessions,
            "nu":                     self.nu,
            "gamma":                  self.gamma,
            "use_kpca":               self.use_kpca,
            "kpca_kernel":            self.kpca_kernel,
            "kpca_gamma":             self.kpca_gamma,
            "kpca_variance":          self.kpca_variance,
            "kpca_n_components_39":   self.kpca_n_components_39,
            "kpca_n_components_zheng":self.kpca_n_components_zheng,
        }, os.path.join(path, "state.pkl"))

    def load(self):
        """Load previously saved models and buffers from disk."""
        path = os.path.join(self.save_dir, self.user_id)

        try:
            state = joblib.load(os.path.join(path, "state.pkl"))
            self.buffer_39    = deque(state["buffer_39"],    maxlen=self.buffer_39.maxlen)
            self.buffer_zheng = deque(state["buffer_zheng"], maxlen=self.buffer_zheng.maxlen)
            self.n_sessions   = state["n_sessions"]
            self.nu           = state.get("nu",           self.nu)
            self.gamma        = state.get("gamma",        self.gamma)
            self.use_kpca     = state.get("use_kpca",     self.use_kpca)
            self.kpca_kernel  = state.get("kpca_kernel",  self.kpca_kernel)
            self.kpca_gamma   = state.get("kpca_gamma",   self.kpca_gamma)
            self.kpca_variance= state.get("kpca_variance",self.kpca_variance)
            self.kpca_n_components_39    = state.get("kpca_n_components_39",    None)
            self.kpca_n_components_zheng = state.get("kpca_n_components_zheng", None)
            print(f"[Trainer] Loaded state: {self.n_sessions} sessions, "
                  f"{len(self.buffer_39)} 39-feat windows, "
                  f"{len(self.buffer_zheng)} Zheng windows")
        except FileNotFoundError:
            print("[Trainer] No saved state found — starting fresh")
            return

        try:
            self.models_39 = joblib.load(os.path.join(path, "models_39.pkl"))
            self.scaler_39 = joblib.load(os.path.join(path, "scaler_39.pkl"))
            print("[Trainer] Loaded 39-feat model from disk")
        except FileNotFoundError:
            print("[Trainer] No 39-feat model found")

        try:
            self.models_zheng = joblib.load(os.path.join(path, "models_zheng.pkl"))
            self.scaler_zheng = joblib.load(os.path.join(path, "scaler_zheng.pkl"))
            print("[Trainer] Loaded Zheng model from disk")
        except FileNotFoundError:
            print("[Trainer] No Zheng model found")

        try:
            self.kpca_39 = joblib.load(os.path.join(path, "kpca_39.pkl"))
        except FileNotFoundError:
            self.kpca_39 = None

        try:
            self.kpca_zheng = joblib.load(os.path.join(path, "kpca_zheng.pkl"))
        except FileNotFoundError:
            self.kpca_zheng = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_ready(self):
        """True if at least one model is trained and can score sessions."""
        return self.models_39 is not None or self.models_zheng is not None

    @property
    def enrollment_status(self):
        lines = [f"Sessions collected: {self.n_sessions}"]
        lines.append(f"39-feat windows:   {len(self.buffer_39)} / {MIN_SESSIONS_39} needed")
        lines.append(f"Zheng windows:     {len(self.buffer_zheng)} / {MIN_SESSIONS_ZHENG} needed")
        lines.append(f"39-feat model:     {'ready' if self.models_39    else 'not yet trained'}")
        lines.append(f"Zheng model:       {'ready' if self.models_zheng else 'not yet trained'}")
        return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python continual_trainer.py <user_id> <session1.csv> [session2.csv ...]")
        sys.exit(1)
    user_id  = sys.argv[1]
    sessions = sys.argv[2:]
    trainer  = ContinualTrainer(user_id=user_id)
    trainer.load()
    for s in sessions:
        trainer.update(s)
    print("\n" + trainer.enrollment_status)