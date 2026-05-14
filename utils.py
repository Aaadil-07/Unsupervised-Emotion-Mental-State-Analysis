"""
utils.py
========
EmotionTracker, overlay drawing, session plots.
"""

import os, csv, time
import numpy as np
from collections import deque
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clustering import EMOTION_COLORS

LOGS_DIR  = "logs"
PLOTS_DIR = "plots"


def _mkdirs():
    os.makedirs(LOGS_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# EMOTION TRACKER
# ══════════════════════════════════════════════════════════════

class EmotionTracker:
    def __init__(self, window_size=10):
        self.window_size  = window_size
        self.cluster_hist = deque(maxlen=window_size)
        self.emotion_log  = []
        self.alerts_log   = []
        self._t0          = time.time()

    def update(self, cluster_id, emotion, mental_state, alert):
        ts = round(time.time() - self._t0, 3)
        self.cluster_hist.append(int(cluster_id))
        self.emotion_log.append({
            "time": ts, "cluster_id": int(cluster_id),
            "emotion": emotion, "mental_state": mental_state,
            "alert": alert or "None",
        })
        if alert:
            self.alerts_log.append({"time": ts, "alert": alert})

    def smoothed_emotion(self):
        if not self.emotion_log:
            return "—"
        recent = [e["emotion"] for e in self.emotion_log[-self.window_size:]]
        vals, cnts = np.unique(recent, return_counts=True)
        return str(vals[np.argmax(cnts)])

    def smoothed_cluster(self):
        if not self.cluster_hist:
            return -1
        vals, cnts = np.unique(list(self.cluster_hist), return_counts=True)
        return int(vals[np.argmax(cnts)])

    def is_mood_swing(self, threshold=4):
        return len(self.cluster_hist) >= 5 and len(set(self.cluster_hist)) >= threshold

    def is_stable(self):
        return len(self.cluster_hist) >= self.window_size and len(set(self.cluster_hist)) == 1

    def recent_history(self):
        return list(self.cluster_hist)

    def emotion_trend(self):
        return ([e["time"] for e in self.emotion_log],
                [e["emotion"] for e in self.emotion_log])

    def session_summary(self):
        if not self.emotion_log:
            return {}
        emotions  = [e["emotion"] for e in self.emotion_log]
        unique, cnts = np.unique(emotions, return_counts=True)
        return {
            "dominant_emotion": str(unique[np.argmax(cnts)]),
            "session_duration": round(self.emotion_log[-1]["time"] if self.emotion_log else 0, 1),
            "total_frames":     len(self.emotion_log),
            "n_alerts":         len(self.alerts_log),
            "emotion_counts":   dict(zip(unique.tolist(), cnts.tolist())),
        }

    def export_csv(self):
        _mkdirs()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"session_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["time", "cluster_id", "emotion", "mental_state", "alert"])
            w.writeheader()
            w.writerows(self.emotion_log)
        print(f"[Tracker] Exported -> {path}")
        return path

    def reset(self):
        self.cluster_hist.clear()
        self.emotion_log.clear()
        self.alerts_log.clear()
        self._t0 = time.time()


# ══════════════════════════════════════════════════════════════
# OVERLAY
# ══════════════════════════════════════════════════════════════

_EMOTION_BGR = {
    "Happy":     (0,   215, 255),
    "Calm":      (80,  200, 80),
    "Neutral":   (180, 180, 180),
    "Sad":       (200, 80,  80),
    "Stressed":  (50,  150, 255),
    "Angry":     (50,  50,  244),
    "Surprised": (200, 64,  224),
    "Outlier":   (40,  80,  255),
}


def draw_overlay(frame, emotion, mental_state, cluster_id, alert,
                 smoothed=None, bbox=None):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    display_emo = smoothed or emotion
    emo_col     = _EMOTION_BGR.get(display_emo, (0, 230, 180))

    cv2.putText(frame, f"Emotion: {display_emo}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emo_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"State: {mental_state}  |  Cluster: {cluster_id}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
    if alert:
        a_col = (50, 50, 255) if "Stress" in alert else (50, 150, 255)
        cv2.putText(frame, alert,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a_col, 2, cv2.LINE_AA)
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, display_emo, (x, max(y - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, emo_col, 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════
# SESSION PLOTS
# ══════════════════════════════════════════════════════════════

def plot_emotion_trend(tracker, save_path=None):
    _mkdirs()
    ts, labels = tracker.emotion_trend()
    if not ts:
        return None
    unique = sorted(set(labels))
    eto    = {e: i for i, e in enumerate(unique)}
    yvals  = [eto[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12122a")

    # Smoothed line
    k = min(5, len(yvals))
    smooth = np.convolve(yvals, np.ones(k) / k, mode="same")
    ax.fill_between(ts, smooth, alpha=0.15, color="#00e5ff")
    ax.plot(ts, smooth, color="#00e5ff", linewidth=1.8, alpha=0.9)
    ax.scatter(ts, yvals,
               c=[EMOTION_COLORS.get(l, "#aaa") for l in labels],
               s=14, zorder=5, alpha=0.8)

    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique, color="white", fontsize=8)
    ax.set_xlabel("Session time (s)", color="#aaa", fontsize=9)
    ax.set_title("Emotion Trend", color="white", fontsize=11)
    ax.tick_params(axis="x", colors="#555")
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    plt.tight_layout()

    path = save_path or os.path.join(PLOTS_DIR, "emotion_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#12122a")
    return fig


def plot_session_pie(tracker, save_path=None):
    _mkdirs()
    summary = tracker.session_summary()
    if not summary or not summary.get("emotion_counts"):
        return None
    counts = summary["emotion_counts"]
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [EMOTION_COLORS.get(l, "#aaa") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#12122a")
    ax.pie(sizes, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90,
           textprops={"color": "white", "fontsize": 10},
           wedgeprops={"edgecolor": "#12122a", "linewidth": 1.5})
    ax.set_title("Session Emotion Distribution", color="white", fontsize=12)
    plt.tight_layout()
    path = save_path or os.path.join(PLOTS_DIR, "session_pie.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#12122a")
    return fig
