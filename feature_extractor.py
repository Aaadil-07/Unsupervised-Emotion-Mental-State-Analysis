"""
feature_extractor.py
====================
Extracts 64-dim geometric features from 48x48 face patches.
Designed to separate all 7 emotions: Happy, Calm, Neutral,
Sad, Stressed, Angry, Surprised — with no positive bias.

Feature groups:
  [0:4]   Zone intensity ratios
  [4:11]  Eye region (brightness, std, asymmetry, dark fraction)
  [11:19] Mouth region (opening, smile, upper/lower)
  [19:26] Brow region (mean, tension, asymmetry, dark)  <- key for Angry/Sad
  [26:30] Cascade eye geometry
  [30:36] Gradient AU proxies (brow AU4, mouth AU12/26, eye AU5)
  [36:44] Brightness histogram (8 bins)
  [44:54] DCT shape coefficients (10)
  [54:64] Action unit proxies (AU4, AU5, AU9, AU12, AU15, AU17, AU25, AU43)

Autoencoder: 64->128->32->128->64 (unsupervised reconstruction)
No PCA needed — 32-dim latent is compact enough for KMeans directly.
"""

import cv2
import numpy as np
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

IMG_SIZE   = 48
AE_LATENT  = 32
MODELS_DIR = "models"

_EYE   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_SMILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def extract_geometric_features(flat_face):
    """
    Extract 64-dim feature vector from flat 48x48 face array.
    Returns float32 array, no NaN/Inf.
    """
    face   = flat_face.reshape(48, 48)
    face_u = (face * 255).astype(np.uint8)
    h, w   = 48, 48
    feats  = []

    # ── [0:4] Zone intensity ratios ─────────────────────────
    top = face[:16, :]
    mid = face[16:32, :]
    bot = face[32:, :]
    feats += [
        top.mean() / (bot.mean() + 1e-8),
        mid.mean() / (top.mean() + 1e-8),
        bot.std()  / (top.std()  + 1e-8),
        face[:, :24].mean() / (face[:, 24:].mean() + 1e-8),
    ]

    # ── [4:11] Eye region ────────────────────────────────────
    ey  = face[12:26, :]
    eu  = face_u[12:26, :]
    eyes = _EYE.detectMultiScale(eu, 1.1, 3, minSize=(5, 5))
    feats += [
        float(ey.mean()),
        float(ey.std()),
        float(ey.min()),
        float(ey[:, :24].mean()),
        float(ey[:, 24:].mean()),
        float(ey[:, :24].mean() - ey[:, 24:].mean()),
        float((ey < 0.3).mean()),
    ]

    # ── [11:19] Mouth region ─────────────────────────────────
    mz  = face[29:43, :]
    mu  = face_u[29:43, :]
    sm  = _SMILE.detectMultiScale(mu, 1.1, 3, minSize=(5, 5))
    mh, mw = mz.shape
    feats += [
        float(mz.mean()),
        float(mz.std()),
        float(mz.max() - mz.min()),
        float(mz[:mh//2, :].mean()),
        float(mz[mh//2:, :].mean()),
        float(len(sm)),
        float(sum(a * b for _, _, a, b in sm) / (mw * mh + 1e-8) if len(sm) else 0.0),
        float(mz[:, mw//4:3*mw//4].mean()),
    ]

    # ── [19:26] Brow region (critical for Angry/Sad/Stressed) ─
    bz  = face[2:14, :]
    bl  = bz[:, :24]
    br  = bz[:, 24:]
    feats += [
        float(bz.mean()),
        float(bz.std()),
        float(bl.mean()),
        float(br.mean()),
        float(bl.mean() - br.mean()),
        float((bz < bz.mean()).mean()),
        float(np.abs(np.diff(bz.astype(np.float64), axis=0)).mean()),
    ]

    # ── [26:30] Cascade eye geometry ────────────────────────
    if len(eyes) >= 2:
        es = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = es[0]
        ex2, ey2, ew2, eh2 = es[1]
        feats += [
            float(abs(ex2 - ex1)) / w,
            float(ew1 + ew2) / (2 * w),
            float(eh1 + eh2) / (2 * h),
            float(abs(ey1 - ey2)) / h,
        ]
    elif len(eyes) == 1:
        _, _, ew, eh = eyes[0]
        feats += [0.3, float(ew) / w, float(eh) / h, 0.0]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]

    # ── [30:36] Gradient AU proxies ──────────────────────────
    gx = np.diff(face.astype(np.float64), axis=1)
    gy = np.diff(face.astype(np.float64), axis=0)
    feats += [
        float(np.abs(gx).mean()),
        float(np.abs(gy).mean()),
        float(np.abs(gy[:16, :]).mean()),    # AU4 brow lowerer (anger/stress)
        float(np.abs(gx[29:, :]).mean()),    # AU12 lip corner
        float(np.abs(gy[29:, :]).mean()),    # AU26 jaw drop (surprise/sad)
        float(np.abs(gy[16:29, :]).mean()),  # AU5 eye widener (surprise)
    ]

    # ── [36:44] Brightness histogram (8 bins) ───────────────
    hist, _ = np.histogram(face, bins=8, range=(0.0, 1.0), density=True)
    feats += hist.tolist()

    # ── [44:54] DCT shape coefficients ──────────────────────
    dct = cv2.dct(face.astype(np.float32))
    dv  = dct[:4, :4].flatten()[:10]
    feats += (dv / (float(np.abs(dv).max()) + 1e-8)).tolist()

    # ── [54:64] Action unit proxies ──────────────────────────
    nz   = face[22:30, :]
    chin = face[43:, :]
    feats += [
        float(mz[:, :mw//4].mean()),               # AU12 left corner
        float(mz[:, 3*mw//4:].mean()),              # AU12 right corner
        float(mz[mh//2:, :mw//4].mean()),           # AU15 left depress
        float(mz[mh//2:, 3*mw//4:].mean()),         # AU15 right depress
        float(nz.std()),                             # AU9 nose wrinkle (disgust/anger)
        float(chin.mean()) if chin.size > 0 else 0.5,  # AU17 chin
        float((ey < 0.25).mean()),                  # AU43 eye closure
        float(mz.max() - mz[:mh//3, :].mean()),    # AU25 lips part
        float(bz.min()),                             # AU4 brow lowest
        float(ey.max() - ey.mean()),                # AU5 eye peak (surprise)
    ]

    result = np.array(feats, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)


def extract_batch(flat_faces):
    """Extract features for N faces. Returns (N, 64) float32."""
    print(f"[FE] Extracting features from {len(flat_faces)} faces...")
    out = []
    for i, flat in enumerate(flat_faces):
        out.append(extract_geometric_features(flat))
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(flat_faces)}")
    matrix = np.array(out, dtype=np.float32)
    print(f"[FE] Done: {matrix.shape}  NaN={np.isnan(matrix).any()}")
    return matrix


def get_feature_names():
    """Returns list of 64 feature name strings."""
    names  = ["zone_top_bot", "zone_mid_top", "zone_bot_std", "zone_sym"]
    names += ["eye_mean", "eye_std", "eye_min", "eye_left", "eye_right", "eye_asym", "eye_dark"]
    names += ["mouth_mean", "mouth_std", "mouth_open", "mouth_upper", "mouth_lower",
              "smile_count", "smile_area", "mouth_center"]
    names += ["brow_mean", "brow_std", "brow_left", "brow_right", "brow_asym",
              "brow_dark", "brow_tension"]
    names += ["eye_dist", "eye_w", "eye_h", "eye_vert"]
    names += ["grad_x", "grad_y", "grad_brow", "grad_mouth_h", "grad_mouth_v", "grad_eye_v"]
    names += [f"hist_{i}" for i in range(8)]
    names += [f"dct_{i}" for i in range(10)]
    names += ["AU12_l", "AU12_r", "AU15_l", "AU15_r", "AU9_nose",
              "AU17_chin", "AU43_close", "AU25_open", "AU4_brow", "AU5_eye"]
    return names  # 4+7+8+7+4+6+8+10+10 = 64


# ══════════════════════════════════════════════════════════════
# SHALLOW AUTOENCODER
# ══════════════════════════════════════════════════════════════

class ShallowAutoencoder:
    """
    Unsupervised reconstruction AE: 64->128->32(bottleneck)->128->64.
    Bottleneck latent codes are fed directly to KMeans (no PCA needed).
    """

    def __init__(self, latent_dim=AE_LATENT, max_iter=500, batch_size=512):
        self.latent_dim = latent_dim
        self.max_iter   = max_iter
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self._mlp       = None

    def fit(self, X):
        print(f"[AE] Training on {X.shape}  latent={self.latent_dim}  iter={self.max_iter}")
        Xs = self.scaler.fit_transform(X)
        n  = len(Xs)
        self._mlp = MLPRegressor(
            hidden_layer_sizes  = (128, self.latent_dim, 128),
            activation          = "relu",
            solver              = "adam",
            batch_size          = min(self.batch_size, n),
            max_iter            = self.max_iter,
            learning_rate_init  = 5e-4,
            early_stopping      = (n >= 200),
            validation_fraction = max(0.05, min(0.1, 100.0 / max(n, 1))),
            n_iter_no_change    = 30,
            tol                 = 1e-7,
            random_state        = 42,
            verbose             = True,
        )
        self._mlp.fit(Xs, Xs)
        print(f"[AE] Done. iters={self._mlp.n_iter_}  loss={self._mlp.loss_:.6f}")
        return self

    def encode(self, X):
        """Run forward pass, stop at bottleneck (layer index 1)."""
        a = self.scaler.transform(X)
        for i, (W, b) in enumerate(zip(self._mlp.coefs_, self._mlp.intercepts_)):
            a = a @ W + b
            if i < len(self._mlp.coefs_) - 1:
                a = np.maximum(0, a)
            if i == 1:
                break
        return a.astype(np.float32)

    def reconstruction_loss(self, X, n_sample=2000):
        idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
        Xs  = self.scaler.transform(X[idx])
        return float(np.mean((Xs - self._mlp.predict(Xs)) ** 2))

    @property
    def loss_curve(self):
        return getattr(self._mlp, "loss_curve_", [])

    def save(self, path=None):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = path or os.path.join(MODELS_DIR, "autoencoder.pkl")
        joblib.dump(self, path)
        print(f"[AE] Saved -> {path}")

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(MODELS_DIR, "autoencoder.pkl")
        ae   = joblib.load(path)
        print(f"[AE] Loaded <- {path}")
        return ae


def encode_single(flat_face, ae):
    """Single face -> 32-dim latent (real-time use)."""
    geo = extract_geometric_features(flat_face).reshape(1, -1)
    return ae.encode(geo)[0]
