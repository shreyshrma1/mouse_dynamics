"""
continual_trainer_39.py

Continual training using the 39-feature kinematic pipeline only.
Trains a OneClassSVM on legitimate user data, updating automatically
after each session flush.

Architecture:
  - Each new session is feature-extracted into 39 kinematic features
  - Features are appended to a replay buffer (deque with max size)
  - Model is retrained on the full replay buffer after each update
  - Trained model + scalers are saved to disk for authentication

Usage:
    trainer = ContinualTrainer39(user_id="user1")
    collector = MouseCollector(user_id="user1", flush_interval=300)
    collector.start(trainer=trainer)
"""

import os
import sys
import joblib
import numpy as np
from collections import deque
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features

FEATURE_COLS = [
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

MIN_WINDOWS = 3  # minimum windows before training begins


class ContinualTrainer39:
    def __init__(
        self,
        user_id,
        save_dir="checkpoints_ocsvm",
        buffer_size=50,
        window_size=200,
        nu=0.05,
        gamma="scale",
        kpca_variance=0.95,
    ):
        """
        Args:
            user_id:       user that is being identified
            save_dir:      directory to save trained models and scalers
            buffer_size:   maximum number of windows to keep in the replay
                           buffer — older windows are dropped when full
            window_size:   number of PC/DD actions per window
            nu:            OCSVM nu / contamination parameter — fraction of
                           training data allowed outside the boundary
            gamma:         OCSVM RBF kernel width ("scale" lets sklearn compute it)
            kpca_variance: fraction of KPCA eigenvalue mass to retain (default 0.95)
        """
        self.user_id       = user_id
        self.save_dir      = save_dir
        self.window_size   = window_size
        self.nu            = nu
        self.gamma         = gamma
        self.kpca_variance = kpca_variance
        self.n_sessions    = 0
        os.makedirs(save_dir, exist_ok=True)

        self.buffer      = deque(maxlen=buffer_size)
        self.model       = None
        self.scaler      = None
        self.kpca        = None
        self.kpca_n_components = None


    # ── Feature extraction ────────────────────────────────────────────────

    def _extract(self, session_path):
        """Extract 39 features from a session file, split into windows."""
        try:
            df = extract_session_features(
                session_path, self.user_id, window_size=self.window_size)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
            if len(df) == 0:
                return []
            windows = []
            for _, grp in df.groupby("session"):
                x = grp[FEATURE_COLS].values
                if len(x) >= 5:
                    windows.append(x)
            return windows
        except Exception as e:
            print(f"[Trainer] Extraction failed: {e}")
            return []


    # ── KPCA ─────────────────────────────────────────────────────────────

    def _fit_kpca(self, x_sc):
        """
        Fit KernelPCA on scaled training data and retain enough components
        to explain kpca_variance of the eigenvalue mass.

        KPCA is fit once per training run. The feature independence structure
        doesn't change between sessions, so there's no benefit to refitting it
        incrementally.

        Returns x_proj, the projected data.
        """
        max_components = min(x_sc.shape[0] - 1, x_sc.shape[1])
        if max_components < 1:
            self.kpca = None
            self.kpca_n_components = None
            return x_sc

        kpca = KernelPCA(n_components=max_components, kernel="linear", fit_inverse_transform=False)
        x_full = kpca.fit_transform(x_sc)

        eigvals = getattr(kpca, "eigenvalues_", None)
        if eigvals is None or np.sum(eigvals) <= 1e-12:
            n_components = x_full.shape[1]
        else:
            explained = np.cumsum(eigvals) / np.sum(eigvals)
            n_components = int(np.searchsorted(explained, self.kpca_variance) + 1)

        n_components = max(1, min(n_components, x_full.shape[1]))

        self.kpca = kpca
        self.kpca_n_components = n_components

        print(f"[Trainer] KPCA: retaining {n_components}/{max_components} components "
              f"({self.kpca_variance * 100:.0f}% variance)")

        return x_full[:, :n_components]

    def _apply_kpca(self, x_sc):
        """Apply fitted KPCA to scaled data. Falls back to identity if not fitted."""
        if self.kpca is None:
            return x_sc
        return self.kpca.transform(x_sc)[:, :self.kpca_n_components]


    # ── Training ──────────────────────────────────────────────────────────

    def _retrain(self):
        """Retrain scaler -> KPCA -> OCSVM on all buffered windows."""
        all_rows = np.vstack(list(self.buffer))
        if len(all_rows) < 10:
            return

        self.scaler = StandardScaler()
        x_sc = self.scaler.fit_transform(all_rows)

        x_proj = self._fit_kpca(x_sc)

        self.model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.model.fit(x_proj)

        # log training score distribution so we know where the boundary sits
        scores = self.model.decision_function(x_proj)
        print(f"[Trainer] Retrained on {len(all_rows)} rows from {len(self.buffer)} windows — "
              f"scores: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")


    # ── Public interface ──────────────────────────────────────────────────

    def update(self, session_path):
        """
        Called by MouseCollector after each session flush.
        Extracts features, updates replay buffer, retrains model.
        """
        self.n_sessions += 1
        print(f"[Trainer] Processing session {self.n_sessions}: {session_path}")

        windows = self._extract(session_path)
        for w in windows:
            self.buffer.append(w)

        print(f"[Trainer] Buffer: {len(self.buffer)} windows")

        if len(self.buffer) >= MIN_WINDOWS:
            self._retrain()
        else:
            print(f"[Trainer] Need {MIN_WINDOWS - len(self.buffer)} more "
                  f"windows before model trains")

        self._save()

    def backfill(self, data_dir):
        """Reprocess all existing session files and fill the replay buffer."""
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

        self.buffer.clear()
        for path in files:
            for w in self._extract(path):
                self.buffer.append(w)

        print(f"[Trainer] Backfill complete: {len(self.buffer)} windows")

        if len(self.buffer) >= MIN_WINDOWS:
            self._retrain()
        else:
            print(f"[Trainer] Need {MIN_WINDOWS - len(self.buffer)} more windows")

        self._save()

    def score(self, session_path):
        """
        Score a session file using the raw OCSVM decision function.
          positive -> inside boundary -> ACCEPT
          negative -> outside boundary -> REJECT

        Returns (score, accepted) or (None, None) if model not ready.
        """
        if self.model is None:
            print("[Trainer] Model not yet trained — collecting more data")
            return None, None

        windows = self._extract(session_path)
        if not windows:
            print("[Trainer] No features extracted from session")
            return None, None

        x = np.vstack(windows)
        x_sc = self.scaler.transform(x)
        x_proj = self._apply_kpca(x_sc)

        score    = float(self.model.decision_function(x_proj).mean())
        accepted = score >= 0

        print(f"[Trainer] Score: {score:+.4f} → {'ACCEPTED' if accepted else 'REJECTED'}")
        return score, accepted


    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        """Save model, scaler, KPCA, and state to disk."""
        path = os.path.join(self.save_dir, self.user_id)
        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            joblib.dump(self.model,  os.path.join(path, "model_39.pkl"))
            joblib.dump(self.scaler, os.path.join(path, "scaler_39.pkl"))
        if self.kpca is not None:
            joblib.dump(self.kpca,   os.path.join(path, "kpca_39.pkl"))
        joblib.dump({
            "buffer":           list(self.buffer),
            "n_sessions":       self.n_sessions,
            "kpca_n_components":self.kpca_n_components,
            "nu":               self.nu,
            "gamma":            self.gamma,
            "kpca_variance":    self.kpca_variance,
        }, os.path.join(path, "state_39.pkl"))

    def load(self):
        """Load model, scaler, KPCA, and state from disk."""
        path = os.path.join(self.save_dir, self.user_id)
        try:
            state = joblib.load(os.path.join(path, "state_39.pkl"))
            self.buffer            = deque(state["buffer"], maxlen=self.buffer.maxlen)
            self.n_sessions        = state["n_sessions"]
            self.kpca_n_components = state.get("kpca_n_components", None)
            self.nu                = state.get("nu",            self.nu)
            self.gamma             = state.get("gamma",         self.gamma)
            self.kpca_variance     = state.get("kpca_variance", self.kpca_variance)
            print(f"[Trainer] Loaded state: {self.n_sessions} sessions, "
                  f"{len(self.buffer)} windows")
        except FileNotFoundError:
            print("[Trainer] No saved state found — starting fresh")
            return

        try:
            self.model  = joblib.load(os.path.join(path, "model_39.pkl"))
            self.scaler = joblib.load(os.path.join(path, "scaler_39.pkl"))
            print("[Trainer] Loaded trained model from disk")
        except FileNotFoundError:
            print("[Trainer] No trained model found — will train after enough data")

        try:
            self.kpca = joblib.load(os.path.join(path, "kpca_39.pkl"))
        except FileNotFoundError:
            self.kpca = None


    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_ready(self):
        return self.model is not None

    @property
    def enrollment_status(self):
        lines = [f"Sessions collected: {self.n_sessions}"]
        lines.append(f"Windows buffered:  {len(self.buffer)} / {MIN_WINDOWS} needed")
        lines.append(f"Model:             {'ready' if self.model else 'not yet trained'}")
        lines.append(f"KPCA components:   {self.kpca_n_components}")
        return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python continual_trainer_39.py <user_id> <session1.csv> ...")
        sys.exit(1)
    user_id  = sys.argv[1]
    sessions = sys.argv[2:]
    trainer  = ContinualTrainer39(user_id=user_id)
    trainer.load()
    for s in sessions:
        trainer.update(s)
    print("\n" + trainer.enrollment_status)