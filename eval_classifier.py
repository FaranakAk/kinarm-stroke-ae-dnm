# eval_classifier.py

# Minimal evaluator for a saved AE+classifier .h5


import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model

def cmsa_to_binary(y):
    """CMSA==8 → 0 (control); else 1 (stroke)."""
    y = np.array(y).astype(int)
    return (y != 8).astype(np.int32).reshape(-1,)

def cm_metric(cm):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    return acc, sen, spe

def main(args):
    packs = np.load(args.packs_npz)
    Xte = packs["test_x"]
    yte = cmsa_to_binary(packs["test_y"])

    model_path = Path("models") / f"autoencoder_joint_{args.seed_out}_{args.seed}.h5"
    model = load_model(model_path)
    print(f"[load] {model_path}")

    # Keras model has two outputs during training; at inference we call predict(X) and take the second (class) output
    preds = model.predict(Xte, verbose=0)
    # Handle either [ae_out, class_out] or just class_out depending on Keras version
    if isinstance(preds, (list, tuple)):
        y_score = preds[1].ravel()
    else:
        y_score = preds.ravel()

    y_hat = (y_score >= args.threshold).astype(int)

    cm = confusion_matrix(yte, y_hat)
    acc, sen, spe = cm_metric(cm)
    auc = roc_auc_score(yte, y_score)

    print(f"ACC={acc:.3f}  SEN={sen:.3f}  SPE={spe:.3f}  AUROC={auc:.3f}")

    # Save metrics & optional ROC
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("seed_out,seed,acc,sen,spe,auroc,threshold\n")
        f.write(f"{args.seed_out},{args.seed},{acc:.6f},{sen:.6f},{spe:.6f},{auc:.6f},{args.threshold:.2f}\n")
    print(f"[write] {out_dir/'metrics.csv'}")

    if args.plot:
        fpr, tpr, _ = roc_curve(yte, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1], ls="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.legend(); plt.tight_layout()
        roc_path = out_dir / f"roc_{args.seed_out}_{args.seed}.png"
        plt.savefig(roc_path, dpi=160)
        print(f"[plot] {roc_path}")

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_npz", default="data/packed_train_val_test.npz",
                    help="NPZ with test_x and test_y (and typically train/val too)")
    ap.add_argument("--seed_out", type=int, default=14077)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--plot", action="store_true")
    main(ap.parse_args())
