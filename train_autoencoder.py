# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:03:52 2021

@author: fakbarifar


Minimal trainer for the dense AE+classifier. Saves: models/autoencoder_joint_{seed_out}_{seed}.h5
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def cmsa_to_binary(y):
    """CMSA==8 → 0 (control); else 1 (stroke)."""
    y = np.array(y).astype(int)
    return (y != 8).astype(np.int32).reshape(-1, 1)

def build_dense_ae_classifier(input_dim=14, ae_hidden=7, latent_dim=5, l2_reg=1e-4, lr=1e-3):
    """Dense AE with classifier head on the latent."""
    inp = Input(shape=(input_dim,), name="ae_in")
    x   = Dense(ae_hidden, activation="relu", activity_regularizer=l2(l2_reg))(inp)
    z   = Dense(latent_dim, activation="relu", name="latent_out")(x)
    xh  = Dense(ae_hidden, activation="relu", activity_regularizer=l2(l2_reg))(z)
    ae_out = Dense(input_dim, activation="sigmoid", name="ae_out")(xh)
    cls_out = Dense(1, activation="sigmoid", name="class_out")(z)

    ae_alone = Model(inp, ae_out, name="ae_alone")
    ae_joint = Model(inp, [ae_out, cls_out], name="ae_joint")

    opt = Adam(learning_rate=lr)
    ae_alone.compile(optimizer=opt, loss="mse", metrics=["mse"])
    ae_joint.compile(
        optimizer=opt,
        loss={"ae_out": "mse", "class_out": "binary_crossentropy"},
        loss_weights={"ae_out": 1.0, "class_out": 10.0},
        metrics={"ae_out": "mse", "class_out": "accuracy"},
    )
    return ae_alone, ae_joint

def main(args):
    K.clear_session()
    packs = np.load(args.packs_npz)

    # Expect arrays named train_x, train_y, val_x, val_y
    Xtr, Xval = packs["train_x"], packs["val_x"]
    ytr = cmsa_to_binary(packs["train_y"])
    yval = cmsa_to_binary(packs["val_y"])

    ae_alone, ae_joint = build_dense_ae_classifier(
        input_dim=args.input_dim,
        ae_hidden=args.ae_hidden,
        latent_dim=args.latent_dim,
        l2_reg=args.l2_reg,
        lr=args.lr,
    )

    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1)

    # 1) Pretrain AE
    ae_alone.fit(
        Xtr, Xtr,
        validation_data=(Xval, Xval),
        epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, verbose=1, callbacks=[es]
    )

    # 2) Joint training (AE + classifier)
    hist = ae_joint.fit(
        Xtr, [Xtr, ytr],
        validation_data=(Xval, [Xval, yval]),
        epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, verbose=1, callbacks=[es]
    )

    # Save model
    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"autoencoder_joint_{args.seed_out}_{args.seed}.h5"
    ae_joint.save(models_dir / save_name)
    print(f"[saved] {models_dir/save_name}")

    #quick loss plot
    if args.plot:
        plt.figure()
        plt.plot(hist.history["loss"], label="train")
        plt.plot(hist.history["val_loss"], label="val")
        plt.title("Joint training loss"); plt.legend()
        plt.tight_layout()
        out = Path("results"); out.mkdir(parents=True, exist_ok=True)
        fig_path = out / f"train_loss_{args.seed_out}_{args.seed}.png"
        plt.savefig(fig_path, dpi=160)
        print(f"[plot] {fig_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_npz", default="data/packed_train_val_test.npz",
                    help="NPZ with train_x, train_y, val_x, val_y")
    ap.add_argument("--input_dim", type=int, default=14)
    ap.add_argument("--ae_hidden", type=int, default=7)
    ap.add_argument("--latent_dim", type=int, default=5)
    ap.add_argument("--l2_reg", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed_out", type=int, default=14077)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--plot", action="store_true")
    main(ap.parse_args())
