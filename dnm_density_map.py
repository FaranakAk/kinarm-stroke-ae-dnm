# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:34:47 2021

@author: fa
"""

# 
# Produce the 8-panel (side-by-side) density map for BMUs, 1..8 clinical classes.

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from DESOM_helper import DESOM  # your helper

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    packs = np.load(args.packs_npz, allow_pickle=True)
    X = packs[args.split_x]
    y = packs[args.split_y]  # 1..8

    # Build + load model
    tf.keras.backend.clear_session()
    desom = DESOM(encoder_dims=[X.shape[-1]] + list(args.ae_arch),
                  map_size=[args.map_i, args.map_j])
    desom.initialize(ae_act='relu', ae_init='glorot_uniform')
    desom.compile(gamma=args.gamma, gamma1=1, optimizer=tf.keras.optimizers.Adam(1e-3))
    desom.load_weights(args.model_h5)
    print(f"[load] {args.model_h5}")

    y_pred, _ = desom.predict(X)

    # Create per-class frequency maps
    den_maps = []
    H, W = args.map_i, args.map_j
    for cls in range(1, 9):
        freq = np.zeros((H, W), dtype=int)
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > 0:
            bmus = y_pred[cls_idx]
            rows = bmus // W
            cols = bmus %  W
            for r, c in zip(rows, cols):
                freq[int(r), int(c)] += 1
        den_maps.append(freq.T)  # transpose so columns map visually left->right as in your notebook

    # Concatenate horizontally with a spacer
    spacer = np.full((W, 1), np.nan)
    blocks = []
    for m in den_maps:
        blocks.append(np.concatenate([m, spacer], axis=1))
    full = np.concatenate(blocks, axis=1)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.imshow(full, cmap='hot', interpolation='nearest')
    plt.axis('off')
    title = os.path.splitext(os.path.basename(args.model_h5))[0]
    plt.title(f"DNM density map (model: {title})", fontsize=10)
    out_path = os.path.join(args.out_dir, f"dnm_density_{title}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[plot] {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_npz", default="data/packed_train_val_test.npz")
    ap.add_argument("--split_x", default="test_x")
    ap.add_argument("--split_y", default="test_y")
    ap.add_argument("--model_h5", required=True, help="Path to a DESOM .h5")
    ap.add_argument("--out_dir", default="results/dnm_figs/")
    ap.add_argument("--map_i", type=int, default=10)
    ap.add_argument("--map_j", type=int, default=10)
    ap.add_argument("--ae_arch", nargs="+", type=int, default=[7,6,5])
    ap.add_argument("--gamma", type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
