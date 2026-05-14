"""
realtime.py
===========
Real-time webcam emotion analysis.
Uses trained AE -> KMeans pipeline with 10-frame smoothing window.

Run:
    python realtime.py
    python realtime.py --camera 1 --skip 2
"""

import cv2
import numpy as np
import time
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from data_loader       import load_single_frame, get_face_bbox
from feature_extractor import encode_single, ShallowAutoencoder
from clustering        import (
    load_models, interpret_mental_state, EMOTION_COLORS,
)
from utils import EmotionTracker, draw_overlay

WINDOW = "Emotion Analyser  [Q / ESC = quit]"


def _load_all_models():
    """Load AE + clustering models. Returns (ae, km, db, density, cluster_map, outlier_thresh) or None."""
    ae_path = os.path.join("models", "autoencoder.pkl")
    if not os.path.exists(ae_path):
        print(f"ERROR: Autoencoder not found at {ae_path}")
        print("  Run  python train.py  first.")
        return None

    result = load_models()
    if result is None:
        print("ERROR: Clustering models not found in ./models/")
        print("  Run  python train.py  first.")
        return None

    km, db, density, metrics, cluster_map, outlier_thresh = result
    ae = ShallowAutoencoder.load(ae_path)
    return ae, km, density, cluster_map, outlier_thresh


def run_realtime(camera=0, skip=2, save_session=True):
    """
    Open webcam and run real-time emotion pipeline.

    Parameters
    ----------
    camera       : camera device index (default 0)
    skip         : run inference every N frames (default 2)
    save_session : export session CSV on exit
    """

    # ── Load models ─────────────────────────────────────────
    loaded = _load_all_models()
    if loaded is None:
        return None
    ae, km, density, cluster_map, outlier_thresh = loaded

    centres = km.cluster_centers_
    print(f"Models loaded. Cluster map: {cluster_map}")
    print(f"Silhouette: loaded from models/metrics.pkl")

    # ── Tracker ──────────────────────────────────────────────
    tracker = EmotionTracker(window_size=10)

    # ── Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {camera}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # ── State ────────────────────────────────────────────────
    frame_n      = 0
    last_emotion = "Detecting..."
    last_state   = "—"
    last_cluster = -1
    last_alert   = None
    fps_timer    = time.time()
    fps_val      = 0.0

    print("Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_n += 1

        # ── Inference every `skip` frames ───────────────────
        if frame_n % skip == 0:
            flat = load_single_frame(frame)
            if flat is not None:
                try:
                    # Encode face -> latent -> cluster
                    latent_vec = encode_single(flat, ae).reshape(1, -1)
                    cluster    = int(km.predict(latent_vec)[0])

                    # Outlier check using per-cluster threshold
                    dist       = float(np.linalg.norm(
                        latent_vec[0] - centres[cluster]))
                    thresh     = outlier_thresh.get(cluster, 5.0)
                    is_outlier = dist > thresh

                    # Interpret mental state
                    info = interpret_mental_state(
                        cluster_id      = cluster,
                        is_outlier      = is_outlier,
                        recent_history  = tracker.recent_history(),
                        cluster_density = density,
                        cluster_map     = cluster_map,
                    )

                    last_emotion = info["emotion"]
                    last_state   = info["mental_state"]
                    last_cluster = cluster
                    last_alert   = info["alert"]

                    tracker.update(cluster, last_emotion,
                                   last_state, last_alert)

                except Exception as exc:
                    print(f"Inference error: {exc}")

        # ── Smoothed emotion (10-frame window) ───────────────
        smoothed = tracker.smoothed_emotion()

        # ── Draw overlay ─────────────────────────────────────
        bbox = get_face_bbox(frame)
        draw_overlay(
            frame,
            emotion      = last_emotion,
            mental_state = last_state,
            cluster_id   = last_cluster,
            alert        = last_alert,
            smoothed     = smoothed,
            bbox         = bbox,
        )

        # ── FPS counter ──────────────────────────────────────
        now     = time.time()
        fps_val = 0.9 * fps_val + 0.1 / max(now - fps_timer, 1e-6)
        fps_timer = now
        cv2.putText(frame, f"FPS {fps_val:.0f}",
                    (frame.shape[1] - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ──────────────────────────────────────
    summary = tracker.session_summary()
    print("\n--- Session Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if save_session and tracker.emotion_log:
        csv_path = tracker.export_csv()
        print(f"Session log -> {csv_path}")

    return tracker


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-Time Emotion Analyser")
    ap.add_argument("--camera",   type=int, default=0,
                    help="Camera index (default 0)")
    ap.add_argument("--skip",     type=int, default=2,
                    help="Run inference every N frames (default 2)")
    ap.add_argument("--no_save",  action="store_true",
                    help="Skip session CSV export")
    args = ap.parse_args()

    run_realtime(
        camera       = args.camera,
        skip         = args.skip,
        save_session = not args.no_save,
    )
