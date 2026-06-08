"""
continual_trainer_shen_scroll.py

Continuously trains the Shen OCSVM model as new mouse data is collected.
Called by MouseCollector after each session flush (every ~1 minute).

Architecture:
  - Each new session is feature-extracted (39 holistic + scroll features)
  - System operates in one of two modes:

    EVALUATING (first EVAL_SESSIONS sessions):
      Windows are collected into an eval buffer but NOT used for training.
      After EVAL_SESSIONS sessions, a consensus vote is run:
        - If acceptance rate >= ACCEPT_THRESHOLD → user is legitimate →
          transition to TRAINING mode, fold eval windows into replay buffer,
          retrain the pre-loaded model on the combined data.
        - If acceptance rate < ACCEPT_THRESHOLD → user is flagged as an
          impostor → alert is printed to terminal and the program terminates.

    TRAINING (all sessions after legitimacy is confirmed):
      Windows are appended to a replay buffer and the model is retrained
      after every session. The replay buffer prevents catastrophic forgetting.

  This separation ensures that impostor samples are never used to train
  the one-class model, which would corrupt the learned legitimate boundary.

Flush / session assumptions:
  - MouseCollector flushes every ~1 minute → 1 session ≈ 1 minute of data
  - EVAL_SESSIONS = 2  →  ~2 minutes of data used for initial evaluation
  - Pre-loaded model (trained offline) is used during evaluation phase

Matches the Shen et al. preprocessing pipeline:
  raw features → find_reference → distance vectors → normalize → OCSVM

Usage:
    trainer = ContinualTrainerShenScroll(user_id="shrey")
    trainer.load()                       # load pre-trained model from disk
    trainer.update(session_path)         # call after every flush
"""

import os
import sys
import joblib
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_COLS_HOLISTIC = [
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

SCROLL_COLS = [
    "scroll_count","scroll_rate","scroll_ratio","scroll_up_ratio",
    "scroll_dur_mean","scroll_dur_std",
    "scroll_burst_count","scroll_burst_dur_mean","scroll_burst_len_mean",
]

MIN_WINDOWS   = 8    # minimum windows in replay buffer before (re)training
EVAL_SESSIONS = 2    # number of sessions used for initial evaluation
ACCEPT_THRESHOLD = 0.50  # minimum acceptance rate to pass evaluation
                          # set low (0.50) to account for 28% FRR on legitimate users

# Mode constants
MODE_EVALUATING = "EVALUATING"
MODE_TRAINING   = "TRAINING"


class ContinualTrainerShenScroll:
    def __init__(
        self,
        user_id,
        save_dir="checkpoints_shen_scroll_continual",
        buffer_size=50,
        window_size=5,
        nu=0.06,
        gamma="scale",
        more_scroll=False,
        dir_scroll=False,
    ):
        self.user_id     = user_id
        self.save_dir    = save_dir
        self.window_size = window_size
        self.nu          = nu
        self.gamma       = gamma
        self.more_scroll = more_scroll
        self.dir_scroll  = dir_scroll

        os.makedirs(save_dir, exist_ok=True)

        self.buffer      = deque(maxlen=buffer_size)
        self.model       = None
        self.reference   = None
        self.dist_mean   = None
        self.dist_std    = None
        self.n_sessions  = 0

        # Evaluation phase state
        self.mode        = MODE_EVALUATING
        self.eval_buffer = []   # windows collected during evaluation (not yet in replay buffer)
        self.eval_scores = []   # per-window boolean: True = accepted, False = rejected

    # ── Feature extraction ────────────────────────────────────────────────

    def _feature_cols(self):
        from measurements.extract_features_scroll import MORE_SCROLL_COLS, DIR_SCROLL_COLS
        return (FEATURE_COLS_HOLISTIC + SCROLL_COLS
                + (MORE_SCROLL_COLS if self.more_scroll else [])
                + (DIR_SCROLL_COLS  if self.dir_scroll  else []))

    def _extract_windows(self, session_path):
        from measurements.extract_features_scroll import extract_session_features
        feature_cols = self._feature_cols()
        all_vecs = []
        try:
            df = extract_session_features(
                session_path, self.user_id,
                window_size=self.window_size,
                more_scroll=self.more_scroll,
                dir_scroll=self.dir_scroll,
            )
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
            if len(df) == 0:
                return []
            for _, grp in df.groupby("session"):
                rows = grp[feature_cols].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"[Trainer] Extraction failed for {session_path}: {e}")
        return all_vecs

    # ── Shen preprocessing ────────────────────────────────────────────────

    @staticmethod
    def _find_reference(train_samples):
        n = len(train_samples)
        mean_dists = np.zeros(n)
        for i in range(n):
            dists = np.sum(np.abs(train_samples - train_samples[i]), axis=1)
            mean_dists[i] = dists.sum() / max(n - 1, 1)
        return train_samples[np.argmin(mean_dists)]

    @staticmethod
    def _distance_vectors(samples, reference):
        return np.abs(np.atleast_2d(samples) - reference)

    @staticmethod
    def _normalize(dist_vecs, mean, std):
        std_safe = np.where(std < 1e-9, 1.0, std)
        return (dist_vecs - mean) / std_safe

    def _preprocess(self, vecs):
        """Apply reference → distance → normalize using stored stats."""
        d = self._distance_vectors(np.array(vecs), self.reference)
        return self._normalize(d, self.dist_mean, self.dist_std)

    # ── Training ──────────────────────────────────────────────────────────

    def _retrain(self):
        from sklearn.svm import OneClassSVM

        train_samples = np.array(list(self.buffer))

        self.reference = self._find_reference(train_samples)
        train_dists    = self._distance_vectors(train_samples, self.reference)
        self.dist_mean = train_dists.mean(axis=0)
        self.dist_std  = train_dists.std(axis=0)
        train_norm     = self._normalize(train_dists, self.dist_mean, self.dist_std)

        self.model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.model.fit(train_norm)

        scores = self.model.decision_function(train_norm)
        print(f"[Trainer] Model retrained on {len(train_samples)} windows — "
              f"scores: min={scores.min():.4f}, mean={scores.mean():.4f}, "
              f"max={scores.max():.4f}")

    # ── Evaluation phase logic ────────────────────────────────────────────

    def _score_windows(self, windows):
        """
        Score a list of windows against the current model.
        Returns (per_window_accepted: list[bool], mean_score: float).
        Requires model to be loaded (pre-trained model used during eval phase).
        """
        x_norm  = self._preprocess(windows)
        scores  = self.model.decision_function(x_norm)
        accepted = [float(s) >= 0 for s in scores]
        return accepted, float(scores.mean())

    def _evaluate_and_decide(self):
        """
        Called after EVAL_SESSIONS sessions have been collected.
        Computes consensus acceptance rate over all eval windows and
        either transitions to TRAINING mode or terminates the program.
        """
        total    = len(self.eval_scores)
        n_accept = sum(self.eval_scores)
        rate     = n_accept / total if total > 0 else 0.0

        print(f"\n[Trainer] ── Evaluation complete ──────────────────────────")
        print(f"[Trainer]   Windows evaluated : {total}")
        print(f"[Trainer]   Accepted          : {n_accept}  ({rate*100:.1f}%)")
        print(f"[Trainer]   Threshold         : {ACCEPT_THRESHOLD*100:.0f}%")

        if rate >= ACCEPT_THRESHOLD:
            print(f"[Trainer]   Verdict           : LEGITIMATE ✓")
            print(f"[Trainer] Transitioning to TRAINING mode.")
            print(f"[Trainer] ────────────────────────────────────────────────\n")

            # Fold eval windows into the replay buffer and retrain
            self.mode = MODE_TRAINING
            for w in self.eval_buffer:
                self.buffer.append(w)
            self.eval_buffer.clear()

            if len(self.buffer) >= MIN_WINDOWS:
                self._retrain()
            else:
                print(f"[Trainer] Buffer has {len(self.buffer)} windows; "
                      f"need {MIN_WINDOWS} before retraining.")
        else:
            print(f"[Trainer]   Verdict           : IMPOSTOR ✗")
            print(f"[Trainer] ────────────────────────────────────────────────")
            print(f"\n[SECURITY ALERT] Impostor detected for user '{self.user_id}'.")
            print(f"[SECURITY ALERT] Session terminated.")
            sys.exit(1)

    # ── Public interface ──────────────────────────────────────────────────

    def update(self, session_path):
        """
        Called by MouseCollector after each session flush (~1 minute).

        EVALUATING mode: score windows against the pre-loaded model and
            accumulate evidence. After EVAL_SESSIONS sessions, decide.
        TRAINING mode: append windows to replay buffer and retrain.
        """
        self.n_sessions += 1
        print(f"[Trainer] Session {self.n_sessions} | mode={self.mode} | {session_path}")

        windows = self._extract_windows(session_path)
        if not windows:
            print(f"[Trainer] No windows extracted — skipping session.")
            return

        if self.mode == MODE_EVALUATING:
            if self.model is None:
                # No pre-trained model available — cannot evaluate.
                # Accumulate silently and wait; this shouldn't happen in
                # normal operation where load() is called before update().
                print(f"[Trainer] WARNING: No model loaded for evaluation. "
                      f"Accumulating session {self.n_sessions} without scoring.")
                self.eval_buffer.extend(windows)
            else:
                accepted, mean_score = self._score_windows(windows)
                self.eval_buffer.extend(windows)
                self.eval_scores.extend(accepted)

                n_accept = sum(accepted)
                print(f"[Trainer] Eval session {self.n_sessions}/{EVAL_SESSIONS}: "
                      f"{n_accept}/{len(accepted)} windows accepted "
                      f"(mean score {mean_score:+.4f})")

            if self.n_sessions >= EVAL_SESSIONS:
                self._evaluate_and_decide()

        else:  # MODE_TRAINING
            for w in windows:
                self.buffer.append(w)

            print(f"[Trainer] Buffer: {len(self.buffer)} windows")

            if len(self.buffer) >= MIN_WINDOWS:
                self._retrain()
            else:
                print(f"[Trainer] Need {MIN_WINDOWS - len(self.buffer)} more windows before retraining.")

        self._save()

    def backfill(self, data_dir):
        """
        Reprocess all existing session files and refill the replay buffer.
        Only meaningful when already in TRAINING mode (e.g. after a restart).
        """
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
            for w in self._extract_windows(path):
                self.buffer.append(w)

        print(f"[Trainer] Backfill complete: {len(self.buffer)} windows")

        if len(self.buffer) >= MIN_WINDOWS:
            self._retrain()
        else:
            print(f"[Trainer] Need {MIN_WINDOWS - len(self.buffer)} windows")

        self._save()

    def score(self, session_path):
        """
        Score a session against the trained model.
        Intended for ad-hoc evaluation outside the update() loop.

        Returns:
            (mean_score, accepted) or (None, None) if model not ready.
            Positive score = inside boundary = ACCEPT.
            Negative score = outside boundary = REJECT.
        """
        if self.model is None:
            print("[Trainer] Model not yet trained — collecting more data")
            return None, None

        windows = self._extract_windows(session_path)
        if not windows:
            print("[Trainer] No windows extracted from session")
            return None, None

        accepted, mean_score = self._score_windows(windows)
        n_accept = sum(accepted)
        verdict  = "ACCEPTED" if mean_score >= 0 else "REJECTED"
        print(f"[Trainer] Score: {mean_score:+.4f} → {verdict} "
              f"({n_accept}/{len(accepted)} windows accepted)")

        return mean_score, mean_score >= 0

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        path = os.path.join(self.save_dir, self.user_id)
        os.makedirs(path, exist_ok=True)

        if self.model is not None:
            joblib.dump(self.model, os.path.join(path, "model.pkl"))

        joblib.dump({
            "buffer":      list(self.buffer),
            "reference":   self.reference,
            "dist_mean":   self.dist_mean,
            "dist_std":    self.dist_std,
            "n_sessions":  self.n_sessions,
            "nu":          self.nu,
            "gamma":       self.gamma,
            "window_size": self.window_size,
            "more_scroll": self.more_scroll,
            "dir_scroll":  self.dir_scroll,
            "mode":        self.mode,
            "eval_buffer": self.eval_buffer,
            "eval_scores": self.eval_scores,
        }, os.path.join(path, "state.pkl"))

    def load(self):
        path = os.path.join(self.save_dir, self.user_id)

        try:
            state = joblib.load(os.path.join(path, "state.pkl"))
            self.buffer      = deque(state["buffer"], maxlen=self.buffer.maxlen)
            self.reference   = state.get("reference")
            self.dist_mean   = state.get("dist_mean")
            self.dist_std    = state.get("dist_std")
            self.n_sessions  = state.get("n_sessions",  0)
            self.nu          = state.get("nu",           self.nu)
            self.gamma       = state.get("gamma",        self.gamma)
            self.window_size = state.get("window_size",  self.window_size)
            self.more_scroll = state.get("more_scroll",  self.more_scroll)
            self.dir_scroll  = state.get("dir_scroll",   self.dir_scroll)
            self.mode        = state.get("mode",         MODE_EVALUATING)
            self.eval_buffer = state.get("eval_buffer",  [])
            self.eval_scores = state.get("eval_scores",  [])
            print(f"[Trainer] Loaded state: {self.n_sessions} sessions, "
                  f"{len(self.buffer)} windows in buffer, mode={self.mode}")
        except FileNotFoundError:
            print("[Trainer] No saved state found — starting fresh")
            return

        try:
            self.model = joblib.load(os.path.join(path, "model.pkl"))
            print("[Trainer] Loaded model from disk")
        except FileNotFoundError:
            print("[Trainer] No model found — evaluation phase will be skipped "
                  "until a model is available")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_ready(self):
        return self.model is not None

    @property
    def enrollment_status(self):
        lines = [
            f"Mode:               {self.mode}",
            f"Sessions collected: {self.n_sessions}",
        ]
        if self.mode == MODE_EVALUATING:
            remaining = max(0, EVAL_SESSIONS - self.n_sessions)
            lines += [
                f"Eval windows so far:{len(self.eval_scores)}",
                f"Sessions until eval:{remaining}",
            ]
        else:
            lines += [
                f"Windows in buffer:  {len(self.buffer)} / {MIN_WINDOWS} needed",
            ]
        lines.append(f"Model:              {'ready' if self.model else 'not yet trained'}")
        return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python continual_trainer_shen_scroll.py <user_id> <session1.csv> ...")
        sys.exit(1)
    user_id  = sys.argv[1]
    sessions = sys.argv[2:]
    trainer  = ContinualTrainerShenScroll(user_id=user_id)
    trainer.load()
    for s in sessions:
        trainer.update(s)
    print("\n" + trainer.enrollment_status)