# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:33:37 2021

@author: fa
"""

# 
# Load many DESOM .h5 models, predict BMUs on a set, compute internal indices
# per clinical class for subclusters (k=2..4). Saves CSV/NPY summaries.

import os
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

from DESOM_helper import DESOM   # your helper (provides .load_weights, .predict)
from metrics import dunn               # your dunn() function

def model_list(save_dir):
    files = [f for f in os.listdir(save_dir) if f.endswith(".h5")]
    # Notebook excluded AE weights and generic "DESOM_bestModel_.h5"
    return [f for f in files if "ae" not in f and f != "DESOM_bestModel_.h5"]

def bmu_coords(bmu_idx, map_i, map_j):
    # Convert flattened BMU index -> (row, col)
    return np.column_stack([bmu_idx // map_j, bmu_idx % map_j]).astype(float)

def jitter(x, scale=1.0):
    return x + scale * (np.random.rand(*x.shape) - 0.5)

def score_for_class(coords, k):
    # KMeans on BMU coordinates (with jitter) then compute indices
    if len(coords) < k:
        return np.nan, np.nan, np.nan, np.nan
    km = KMeans(n_clusters=k, random_state=0).fit(coords)
    labels = km.labels_
    D = euclidean_distances(coords)
    dunn_sc = dunn(labels, D, 'farthest', 'farthest')
    db_sc   = 1.0 / davies_bouldin_score(coords, labels)
    sil_sc  = silhouette_score(coords, labels)
    ch_sc   = calinski_harabasz_score(coords, labels)
    return dunn_sc, db_sc, sil_sc, ch_sc

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    packs = np.load(args.packs_npz, allow_pickle=True)
    X = packs[args.split_x]     # e.g., "test_x" or "val_x"
    y = packs.get(args.split_y, None)  # optional clinical labels 1..8; if missing, we score all data together

    files = model_list(args.save_dir)
    print(f"[scan] {len(files)} models in {args.save_dir}")

    # Prepare arrays: [n_models, 8 classes, 3 k-values, 4 metrics]
    n_models = len(files)
    n_classes = 8
    k_values = [2,3,4]
    scores = np.full((n_models, n_classes, len(k_values), 4), np.nan, dtype=float)

    for m_i, fname in enumerate(sorted(files)):
        tf.keras.backend.clear_session()
        fpath = os.path.join(args.save_dir, fname)

        # Create a fresh DESOM with the expected map size.
        # NOTE: we only need it to load and predict, so AE dims don't matter as long as they match weights.
        desom = DESOM(encoder_dims=[X.shape[-1]] + list(args.ae_arch),
                      map_size=[args.map_i, args.map_j])
        desom.initialize(ae_act='relu', ae_init='glorot_uniform')
        desom.compile(gamma=args.gamma, gamma1=1, optimizer=tf.keras.optimizers.Adam(1e-3))
        desom.load_weights(fpath)

        y_pred, dist_pred = desom.predict(X)  # y_pred are BMU indices (flattened)

        if y is None:
            # Score as a single pool (put in class slot 0)
            coords = jitter(bmu_coords(y_pred, args.map_i, args.map_j), scale=1.0)
            for k_i, k in enumerate(k_values):
                scores[m_i, 0, k_i, :] = score_for_class(coords, k)
        else:
            # Score per CMSA class 1..8
            for c in range(1, n_classes+1):
                coords = jitter(bmu_coords(y_pred[y == c], args.map_i, args.map_j), scale=1.0)
                for k_i, k in enumerate(k_values):
                    scores[m_i, c-1, k_i, :] = score_for_class(coords, k)

        print(f"[scored] {fname}")

    # Save arrays + a simple CSV summary (mean across classes and k)
    npy_path = os.path.join(args.out_dir, "dnm_scores.npy")
    np.save(npy_path, scores)
    print(f"[write] {npy_path}")

    # Simple ranking by the average Silhouette (or choose your metric)
    mean_sil = np.nanmean(scores[:,:,:,2], axis=(1,2))  # metric index 2 = silhouette
    order = np.argsort(-mean_sil)
    with open(os.path.join(args.out_dir, "dnm_scores_summary.csv"), "w") as f:
        f.write("rank,model,mean_silhouette\n")
        for r, idx in enumerate(order):
            f.write(f"{r+1},{sorted(files)[idx]},{mean_sil[idx]:.6f}\n")
    print(f"[write] {os.path.join(args.out_dir, 'dnm_scores_summary.csv')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_npz", default="data/packed_train_val_test.npz",
                    help="NPZ with features + (optional) clinical labels.")
    ap.add_argument("--split_x", default="test_x", help="Key for features, e.g., test_x or val_x.")
    ap.add_argument("--split_y", default="test_y", help="Key for clinical labels (1..8). If absent, scores pooled.")
    ap.add_argument("--save_dir", default="results/tmp/", help="Folder with .h5 models.")
    ap.add_argument("--out_dir", default="results/dnm_scores/", help="Where to save scores.")
    ap.add_argument("--map_i", type=int, default=10)
    ap.add_argument("--map_j", type=int, default=10)
    ap.add_argument("--ae_arch", nargs="+", type=int, default=[7,6,5])
    ap.add_argument("--gamma", type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
