"""
train.py
========
Unsupervised Emotion Analysis — Full Training Pipeline
7 Emotions: Happy, Calm, Neutral, Sad, Stressed, Angry, Surprised

Steps:
  1. Load dataset (multi-threaded, CLAHE preprocessing)
  2. Extract 64-dim geometric features (no HOG, no bias)
  3. Train Shallow Autoencoder 64->128->32->128->64
  4. Encode all faces to 32-dim latent (no PCA needed)
  5. Silhouette search K=2..10 to confirm best K
  6. KMeans K=7 on latent codes directly
  7. auto_label_clusters() — data-driven, no hardcoding, fixes bias
  8. DBSCAN with auto-tuned eps (k-distance graph)
  9. Compute quality metrics + outlier thresholds
 10. Save all models to models/
 11. Generate all plots to plots/
 12. Write training_report.txt

Run:
    python train.py
    python train.py --dataset final_dataset --ae_iter 500
    python train.py --force_ae    # retrain AE from scratch
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_dataset
from feature_extractor import extract_batch, ShallowAutoencoder, get_feature_names
from clustering import (
    fit_kmeans, fit_dbscan, auto_eps, search_best_k,
    auto_label_clusters, compute_metrics, tag_cluster_density,
    compute_outlier_threshold, save_models, save_assignments,
    plot_clusters_2d, plot_dbscan_2d, plot_kdist,
    plot_silhouette_bar, plot_silhouette_samples,
    plot_emotion_distribution, plot_metrics_summary,
    plot_ae_loss, plot_feature_importance, plot_noise_analysis,
    MODELS_DIR, PLOTS_DIR,
)

SEP = "=" * 62


def banner(txt):
    print(f"\n{SEP}\n  {txt}\n{SEP}")


# ──────────────────────────────────────────────────────────────
# Extra training-only plots
# ──────────────────────────────────────────────────────────────

def _plot_elbow(k_scores):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ks  = sorted(k_scores.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#12122a")
    for ax, key, title, color in zip(
        axes,
        ["davies_bouldin", "calinski_harabasz"],
        ["Davies-Bouldin (lower=better)", "Calinski-Harabasz (higher=better)"],
        ["#ff9800", "#00e5ff"],
    ):
        ax.set_facecolor("#1a1a2e")
        ax.plot([str(k) for k in ks],
                [k_scores[k][key] for k in ks],
                "o-", color=color, linewidth=2, markersize=8)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("K", color="#aaa")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values():
            s.set_edgecolor("#333")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "elbow_db_ch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#12122a")
    plt.close(fig)
    print(f"[Plot] {path}")


def _plot_latent_scatter(latent, km_labels, cluster_map):
    """2D scatter using latent dims 0 and 1."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    from clustering import EMOTION_COLORS, _dark_fig, _style_ax
    fig, ax = _dark_fig()
    for cid, emo in cluster_map.items():
        mask = km_labels == cid
        ax.scatter(latent[mask, 0], latent[mask, 1],
                   s=6, alpha=0.4,
                   color=EMOTION_COLORS.get(emo, "#aaa"),
                   label=f"{emo} ({mask.sum()})")
    ax.set_title("AE Latent Space (dims 0 & 1)", color="white", fontsize=12)
    ax.set_xlabel("Latent dim 0", color="#aaa")
    ax.set_ylabel("Latent dim 1", color="#aaa")
    _style_ax(ax)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "latent_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#12122a")
    plt.close(fig)
    print(f"[Plot] {path}")


def _plot_feature_heatmap(feats, km_labels, cluster_map, feat_names):
    """Mean feature value per cluster as heatmap."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n_clusters = len(cluster_map)
    n_show     = min(32, feats.shape[1])
    matrix     = np.array([
        feats[km_labels == c, :n_show].mean(0)
        for c in range(n_clusters)
    ])
    rows = [cluster_map.get(i, f"C{i}") for i in range(n_clusters)]
    cols = feat_names[:n_show]

    fig, ax = plt.subplots(figsize=(18, 3))
    fig.patch.set_facecolor("#12122a")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(rows, color="white", fontsize=10)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(cols, color="#aaa", fontsize=6,
                       rotation=45, ha="right")
    ax.set_title("Mean Feature Activation per Emotion Cluster",
                 color="white", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_heatmap.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#12122a")
    plt.close(fig)
    print(f"[Plot] {path}")


def _save_report(metrics, density, km_labels, db_labels,
                 cluster_map, n_images, elapsed, ae_loss,
                 best_k, k_scores):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "training_report.txt")
    n_noise = int((db_labels == -1).sum())
    pct     = n_noise / len(db_labels) * 100

    with open(path, "w") as f:
        f.write("Unsupervised Emotion Analysis — Training Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Images processed    : {n_images}\n")
        f.write(f"Training time       : {elapsed:.1f}s\n")
        f.write(f"AE reconstruction   : {ae_loss:.6f} MSE\n")
        f.write(f"Best K (silhouette) : {best_k}\n")
        f.write(f"Final K used        : {len(cluster_map)}\n\n")

        f.write("── K Search ─────────────────────────────\n")
        for k, sc in sorted(k_scores.items()):
            m = " <- BEST" if k == best_k else ""
            f.write(f"  K={k}  sil={sc['silhouette']:.4f}"
                    f"  DB={sc['davies_bouldin']:.4f}"
                    f"  CH={sc['calinski_harabasz']:.1f}{m}\n")

        f.write("\n── Cluster Map ──────────────────────────\n")
        for cid, emo in sorted(cluster_map.items()):
            cnt = int((km_labels == cid).sum())
            den = density.get(cid, "?")
            f.write(f"  C{cid} -> {emo:12s} n={cnt:6d}  [{den}]\n")

        f.write(f"\n── DBSCAN ───────────────────────────────\n")
        f.write(f"  Noise/outliers : {n_noise}  ({pct:.1f}%)\n")

        f.write(f"\n── Final Metrics ────────────────────────\n")
        for k, v in metrics.items():
            f.write(f"  {k:22s} : {v}\n")

    print(f"[Train] Report -> {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def train(
    dataset_dir = "final_dataset",
    ae_iter     = 500,
    ae_latent   = 32,
    max_images  = 35000,
    dbscan_eps  = None,
    dbscan_min  = 10,
    force_ae    = False,
    n_workers   = 8,
):
    t0 = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    # ── STEP 1: Load dataset ────────────────────────────────
    banner("STEP 1 — Loading Dataset")
    raw_images, img_paths = load_dataset(
        folder=dataset_dir, max_images=max_images, n_workers=n_workers)
    print(f"  Loaded: {raw_images.shape}")

    # ── STEP 2: Feature extraction ──────────────────────────
    banner("STEP 2 — Geometric Feature Extraction (64 dims)")
    feats = extract_batch(raw_images)
    print(f"  Features: {feats.shape}")
    feat_names = get_feature_names()

    # ── STEP 3: Autoencoder ─────────────────────────────────
    banner("STEP 3 — Shallow Autoencoder (64->128->32->128->64)")
    ae_path = os.path.join(MODELS_DIR, "autoencoder.pkl")

    if os.path.exists(ae_path) and not force_ae:
        print(f"  Loading existing AE from {ae_path}")
        print("  Use --force_ae to retrain from scratch")
        ae = ShallowAutoencoder.load(ae_path)
    else:
        ae = ShallowAutoencoder(
            latent_dim  = ae_latent,
            max_iter    = ae_iter,
            batch_size  = min(512, len(feats)),
        )
        ae.fit(feats)
        ae.save(ae_path)

    ae_loss = ae.reconstruction_loss(feats)
    print(f"  AE reconstruction loss: {ae_loss:.6f}")
    if ae.loss_curve:
        plot_ae_loss(ae.loss_curve)

    # ── STEP 4: Encode to latent ────────────────────────────
    banner("STEP 4 — Encoding to 32-dim Latent (no PCA)")
    latent = ae.encode(feats)
    print(f"  Latent: {latent.shape}")

    # ── STEP 5: K search ────────────────────────────────────
    banner("STEP 5 — Silhouette K Search (K=2..10)")
    best_k, k_scores = search_best_k(latent, k_range=range(2, 11))
    plot_silhouette_bar(k_scores)
    _plot_elbow(k_scores)

    # Always use K=7 for 7-emotion dataset
    # but warn if a very different K scores much better
    gap = (k_scores.get(best_k, {}).get("silhouette", 0) -
           k_scores.get(7,      {}).get("silhouette", 0))
    if best_k != 7 and gap > 0.15:
        print(f"  WARNING: K={best_k} scores +{gap:.4f} better than K=7.")
        print(f"  Using K=7 to match 7 emotion classes in dataset.")
    n_clusters = 7
    print(f"  Using K={n_clusters} (Happy/Calm/Neutral/Sad/Stressed/Angry/Surprised)")

    # ── STEP 6: KMeans ──────────────────────────────────────
    banner(f"STEP 6 — MiniBatchKMeans K={n_clusters}")
    km_model, km_labels = fit_kmeans(latent, n_clusters=n_clusters)

    # ── STEP 7: Auto-label clusters (bias fix) ──────────────
    banner("STEP 7 — Auto-Label Clusters (data-driven, no bias)")
    cluster_map = auto_label_clusters(feats, km_labels, n_clusters)
    print(f"  Cluster map: {cluster_map}")

    # Print emotion distribution
    from collections import Counter
    pred       = [cluster_map[l] for l in km_labels]
    pred_dist  = Counter(pred)
    print("\n  Emotion distribution:")
    for emo in ["Happy", "Calm", "Neutral", "Sad", "Stressed", "Angry", "Surprised"]:
        cnt = pred_dist.get(emo, 0)
        pct = 100 * cnt / len(pred)
        bar = "█" * int(pct / 2)
        print(f"    {emo:10s}: {cnt:6d} ({pct:5.1f}%) {bar}")

    # ── STEP 8: DBSCAN ──────────────────────────────────────
    banner("STEP 8 — DBSCAN (auto eps)")
    eps_auto, kd = auto_eps(latent, k=5, percentile=95)
    plot_kdist(kd, eps_auto)
    db_model, db_labels = fit_dbscan(
        latent,
        eps         = dbscan_eps or eps_auto,
        min_samples = dbscan_min,
    )

    # ── STEP 9: Metrics & thresholds ────────────────────────
    banner("STEP 9 — Quality Metrics & Outlier Thresholds")
    metrics        = compute_metrics(latent, km_labels)
    density        = tag_cluster_density(km_labels)
    outlier_thresh = compute_outlier_threshold(
        latent, km_labels, km_model, percentile=99)

    print(f"  Silhouette        : {metrics['silhouette']}")
    print(f"  Davies-Bouldin    : {metrics['davies_bouldin']}")
    print(f"  Calinski-Harabasz : {metrics['calinski_harabasz']}")
    print(f"  Density           : {density}")

    # ── STEP 10: Save models ─────────────────────────────────
    banner("STEP 10 — Saving Models")
    save_models(km_model, db_model, density, metrics,
                cluster_map, outlier_thresh, k_scores)
    ae.save(ae_path)
    save_assignments(img_paths, km_labels, cluster_map)

    # ── STEP 11: All plots ───────────────────────────────────
    banner("STEP 11 — Generating Plots")
    latent_2d = latent[:, :2]       # dims 0 & 1 for 2D scatter
    plot_clusters_2d(latent_2d, km_labels, cluster_map)
    _plot_latent_scatter(latent, km_labels, cluster_map)
    plot_dbscan_2d(latent_2d, db_labels)
    plot_emotion_distribution(km_labels, cluster_map)
    plot_noise_analysis(db_labels, km_labels, cluster_map)
    plot_silhouette_samples(latent, km_labels)
    plot_metrics_summary(metrics)
    plot_feature_importance(feats, km_labels, cluster_map)
    _plot_feature_heatmap(feats, km_labels, cluster_map, feat_names)

    # ── STEP 12: Report ──────────────────────────────────────
    elapsed = time.time() - t0
    banner("STEP 12 — Training Report")
    _save_report(metrics, density, km_labels, db_labels,
                 cluster_map, len(raw_images), elapsed,
                 ae_loss, best_k, k_scores)

    banner(f"TRAINING COMPLETE — {elapsed:.1f}s")
    print(f"\n  Silhouette Score : {metrics['silhouette']}")
    print(f"  Cluster map      : {cluster_map}")
    print(f"  Models  -> ./{MODELS_DIR}/")
    print(f"  Plots   -> ./{PLOTS_DIR}/\n")

    return dict(
        ae=ae, km=km_model, db=db_model,
        density=density, metrics=metrics,
        cluster_map=cluster_map, outlier_thresh=outlier_thresh,
        km_labels=km_labels, db_labels=db_labels,
        feats=feats, latent=latent, k_scores=k_scores,
    )


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train Unsupervised Emotion Analysis (7 emotions, no bias)")
    ap.add_argument("--dataset",    default="final_dataset",
                    help="Image folder (default: final_dataset)")
    ap.add_argument("--ae_iter",    type=int, default=500,
                    help="AE training epochs (default 500, more = better)")
    ap.add_argument("--ae_latent",  type=int, default=32,
                    help="AE bottleneck size (default 32)")
    ap.add_argument("--max_images", type=int, default=35000,
                    help="Max images to load (default 35000)")
    ap.add_argument("--dbscan_eps", type=float, default=None,
                    help="DBSCAN eps (default: auto from k-distance graph)")
    ap.add_argument("--dbscan_min", type=int, default=10,
                    help="DBSCAN min_samples (default 10)")
    ap.add_argument("--force_ae",   action="store_true",
                    help="Force AE retraining even if saved model exists")
    ap.add_argument("--workers",    type=int, default=8,
                    help="Number of data loading threads (default 8)")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"\nERROR: Dataset folder '{args.dataset}' not found.")
        print("  Create it and add your face images:")
        print("  mkdir final_dataset && cp *.jpg final_dataset/")
        sys.exit(1)

    train(
        dataset_dir = args.dataset,
        ae_iter     = args.ae_iter,
        ae_latent   = args.ae_latent,
        max_images  = args.max_images,
        dbscan_eps  = args.dbscan_eps,
        dbscan_min  = args.dbscan_min,
        force_ae    = args.force_ae,
        n_workers   = args.workers,
    )
