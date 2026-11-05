"""Training & 5-Fold Group Cross-Validation Pipeline for CSI Respiration Net (v2).

Evaluates generalization strictly across unseen recording sessions via
StratifiedGroupKFold (5 folds), saves the cross-validated confusion matrix
and learning curves, and exports a 5-fold model ensemble checkpoint to
``data/weights/best_csi_model.pt``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "04_ai_model"))

from config.settings import DATA_DIR
from models import CSI_RespirationNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = DATA_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(train_losses, val_losses, val_accs, out_dir: Path) -> None:
    """Save training loss and validation accuracy plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss (5-Fold Avg)", color="#1e90ff", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss (Unseen Sessions)", color="#ff4757", linewidth=2)
    ax1.set_title("Multi-Task Learning Loss Curves", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Combined Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val Accuracy (%)", color="#2ed573", linewidth=2)
    ax2.set_title("5-Fold Unseen-Session Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    """Save formatted 5-fold out-of-fold confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2.0 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.upper() for c in class_names], fontsize=10)
    ax.set_yticklabels([c.upper() for c in class_names], fontsize=10)
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("True Class", fontweight="bold")
    ax.set_title("5-Fold Out-of-Session Confusion Matrix", pad=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def train_model(
    dataset_path: Path | None = None,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 2e-3,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """Train 5-fold group-cross-validated Dual-Branch CSI_RespirationNet and save ensemble."""
    if dataset_path is None:
        dataset_path = DATA_DIR / "processed" / "csi_respiration_dataset_4s.npz"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run dataset_prep.py first.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    data = np.load(dataset_path)
    X_wave = data["X"]
    X_spec = data["X_spec"]
    X_aux = data["X_aux"]
    y_class = data["y_class"]
    y_bpm = data["y_bpm"]
    groups = data["groups"]
    class_names = [str(c) for c in data["class_names"]]
    window_sec = float(data.get("window_sec", 16.0))
    target_fs = float(data.get("target_fs", 90.0))

    print("=" * 72)
    print("  5-FOLD GROUP CROSS-VALIDATION: DUAL-BRANCH CSI RESPIRATION NET")
    print("=" * 72)
    print(f"Device: {DEVICE} | Windows: {X_wave.shape[0]} ({window_sec:.1f}s) | Wave: {X_wave.shape[1:]} | Spec: {X_spec.shape[1]}")

    counts = np.bincount(y_class, minlength=len(class_names))
    cls_weights = torch.tensor(len(y_class) / (len(class_names) * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    oof_preds = np.zeros_like(y_class)
    oof_bpm = np.zeros_like(y_bpm)
    fold_state_dicts = []
    fold_f1s = []

    all_tr_losses = np.zeros((n_splits, epochs))
    all_va_losses = np.zeros((n_splits, epochs))
    all_va_accs = np.zeros((n_splits, epochs))

    for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X_wave, y_class, groups)):
        model = CSI_RespirationNet(
            in_channels=X_wave.shape[1],
            spec_dim=X_spec.shape[1],
            aux_dim=X_aux.shape[1],
            num_classes=len(class_names),
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
        criterion_cls = nn.CrossEntropyLoss(weight=cls_weights)
        criterion_reg = nn.SmoothL1Loss()

        tr_ds = TensorDataset(
            torch.tensor(X_wave[tr_idx], dtype=torch.float32),
            torch.tensor(X_spec[tr_idx], dtype=torch.float32),
            torch.tensor(X_aux[tr_idx], dtype=torch.float32),
            torch.tensor(y_class[tr_idx], dtype=torch.long),
            torch.tensor(y_bpm[tr_idx], dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

        vw = torch.tensor(X_wave[va_idx], dtype=torch.float32, device=DEVICE)
        vs = torch.tensor(X_spec[va_idx], dtype=torch.float32, device=DEVICE)
        va = torch.tensor(X_aux[va_idx], dtype=torch.float32, device=DEVICE)
        vy_cls = torch.tensor(y_class[va_idx], dtype=torch.long, device=DEVICE)
        vy_bpm = torch.tensor(y_bpm[va_idx], dtype=torch.float32, device=DEVICE)

        best_fold_f1 = -1.0
        best_state = None
        best_preds = None
        best_pbpm = None

        for ep in range(epochs):
            model.train()
            tr_loss_sum = 0.0
            for bw, bs, ba, by, bb in tr_loader:
                bw, bs, ba = bw.to(DEVICE), bs.to(DEVICE), ba.to(DEVICE)
                by, bb = by.to(DEVICE), bb.to(DEVICE)

                # Subcarrier dropout & Gaussian jitter augmentation
                if torch.rand(1).item() < 0.5:
                    mask = (torch.rand(bw.size(0), bw.size(1), 1, device=DEVICE) > 0.15).float()
                    bw = bw * mask
                bw = bw + 0.02 * bw.std() * torch.randn_like(bw)

                optimizer.zero_grad()
                logits, pb = model(bw, bs, ba)
                loss = criterion_cls(logits, by) + 1.5 * criterion_reg(pb / 30.0, bb / 30.0)
                loss.backward()
                optimizer.step()
                tr_loss_sum += loss.item() * bw.size(0)

            scheduler.step()
            all_tr_losses[fold_idx, ep] = tr_loss_sum / len(tr_idx)

            model.eval()
            with torch.no_grad():
                v_logits, v_pb = model(vw, vs, va)
                v_loss = criterion_cls(v_logits, vy_cls) + 1.5 * criterion_reg(v_pb / 30.0, vy_bpm / 30.0)
                all_va_losses[fold_idx, ep] = v_loss.item()

                preds = v_logits.argmax(dim=1).cpu().numpy()
                acc = float(np.mean(preds == y_class[va_idx]))
                f1 = float(f1_score(y_class[va_idx], preds, average="macro"))
                all_va_accs[fold_idx, ep] = acc

                if f1 >= best_fold_f1:
                    best_fold_f1 = f1
                    best_state = copy.deepcopy(model.state_dict())
                    best_preds = preds
                    best_pbpm = v_pb.cpu().numpy()

        oof_preds[va_idx] = best_preds
        oof_bpm[va_idx] = best_pbpm
        fold_state_dicts.append(best_state)
        fold_f1s.append(best_fold_f1)
        fold_acc = float(np.mean(best_preds == y_class[va_idx])) * 100.0
        print(f"  [+] Fold {fold_idx + 1}/{n_splits} (Unseen Sessions) -> Val Acc: {fold_acc:5.1f}% | Macro F1: {best_fold_f1:5.3f}")

    overall_acc = float(np.mean(oof_preds == y_class))
    overall_f1 = float(f1_score(y_class, oof_preds, average="macro"))
    mask_breath = y_class > 0
    mae_bpm = float(np.mean(np.abs(oof_bpm[mask_breath] - y_bpm[mask_breath])))

    best_fold = int(np.argmax(fold_f1s))
    best_weights_path = WEIGHTS_DIR / "best_csi_model.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": fold_state_dicts[best_fold],
            "ensemble_state_dicts": fold_state_dicts,
            "val_f1": overall_f1,
            "val_acc": overall_acc,
            "val_bpm_mae": mae_bpm,
            "class_names": class_names,
            "window_sec": window_sec,
            "target_fs": target_fs,
            "spec_dim": int(X_spec.shape[1]),
            "aux_dim": int(X_aux.shape[1]),
        },
        best_weights_path,
    )

    print("\n" + "=" * 72)
    print(f"  [v] 5-FOLD UNSEEN-SESSION ACCURACY : {overall_acc * 100:5.1f}%")
    print(f"  [v] 5-FOLD UNSEEN-SESSION MACRO F1 : {overall_f1:5.3f}")
    print(f"  [v] BREATHING RATE MAE             : {mae_bpm:4.2f} BPM")
    print(f"  [v] Ensemble Checkpoint Saved      : {best_weights_path}")
    print("=" * 72)
    print("\nOut-of-Session Classification Report:")
    print(classification_report(y_class, oof_preds, target_names=[c.upper() for c in class_names], digits=3))

    out_dir = DATA_DIR / "processed"
    plot_training_curves(all_tr_losses.mean(0), all_va_losses.mean(0), all_va_accs.mean(0), out_dir)
    cm = confusion_matrix(y_class, oof_preds)
    plot_confusion_matrix(cm, class_names, out_dir)
    print(f"[*] Saved plots to: {out_dir / 'training_curves.png'} & {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    train_model()
