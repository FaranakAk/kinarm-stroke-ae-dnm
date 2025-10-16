# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:30:47 2021

@author: fa
"""

# Minimal multi-run DESOM training loop that trains N models with different seeds and saves each .h5 into save_dir.

import os
import argparse
import numpy as np
import random
import tensorflow as tf
from time import time

from DESOM_helper import DESOM  

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Load feature arrays (use same prepared arrays you used in the notebook).
    # Expect NPZ with at least train_x, val_x (optionally test_x/test_y for quick checks)
    packs = np.load(args.packs_npz, allow_pickle=True)
    X_train = packs["train_x"]
    X_val   = packs.get("val_x", None)

    print(f"[data] X_train: {X_train.shape}", f"X_val: {None if X_val is None else X_val.shape}")

    for run in range(args.n_models):
        seed = np.random.randint(4_000_000) if args.seed < 0 else args.seed + run
        set_seed(seed)

        t0 = time()
        tf.keras.backend.clear_session()

        # Build DESOM (AE encoder dims = [input_dim] + ae_arch)
        input_dim = X_train.shape[-1]
        desom = DESOM(encoder_dims=[input_dim] + list(args.ae_arch),
                      map_size=[args.map_i, args.map_j])

        # Initialize
        desom.initialize(ae_act='relu', ae_init='glorot_uniform')

        # Optimizers
        pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=args.ae_lr)
        optimizer          = tf.keras.optimizers.Adam(learning_rate=args.desom_lr)

        # Pretrain AE (optional)
        if args.pretrain_epochs > 0:
            print(f"[pretrain] epochs={args.pretrain_epochs}, bs={args.batch_size}")
            desom.pretrain(X=X_train, V=X_val, optimizer=pretrain_optimizer,
                           epochs=args.pretrain_epochs, batch_size=args.batch_size,
                           save_dir=args.save_dir)

        # Compile and train DESOM
        desom.compile(gamma=args.gamma, gamma1=1, optimizer=optimizer)
        print(f"[train] DESOM iterations={args.iterations}, batch_size={args.batch_size}")
        desom.fit(X=X_train,
                  iterations=args.iterations,
                  batch_size=args.batch_size,
                  Tmax=args.Tmax, Tmin=args.Tmin,
                  som_iterations=args.som_iterations,
                  eval_interval=args.eval_interval,
                  save_epochs=args.save_epochs,
                  save_dir=args.save_dir)

        # Save weights with a clear name
        model_name = f"DESOM_bestModel_seed{seed}_{args.map_i}x{args.map_j}.h5"
        save_path  = os.path.join(args.save_dir, model_name)
        desom.save_weights(save_path)
        dt = time() - t0
        print(f"[saved] {save_path}  ({dt:.1f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_npz", default="data/packed_train_val_test.npz",
                    help="NPZ with train_x (and optionally val_x).")
    ap.add_argument("--save_dir", default="results/tmp/",
                    help="Directory to save DESOM .h5 models.")
    ap.add_argument("--ae_arch", nargs="+", type=int, default=[7,6,5],
                    help="AE hidden layers after input_dim (e.g. 7 6 5).")
    ap.add_argument("--map_i", type=int, default=10)
    ap.add_argument("--map_j", type=int, default=10)
    ap.add_argument("--pretrain_epochs", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--som_iterations", type=int, default=1000)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--save_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--Tmax", type=float, default=10.0)
    ap.add_argument("--Tmin", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--desom_lr", type=float, default=1e-3)
    ap.add_argument("--n_models", type=int, default=10, help="How many seeds to train.")
    ap.add_argument("--seed", type=int, default=-1, help="Base seed; <0 → random each run.")
    args = ap.parse_args()
    main(args)
