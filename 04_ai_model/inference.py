"""Real-time AI Inference Module for CSI Respiration Sensing (v2).

Loads the trained Dual-Branch CSI_RespirationNet (and 5-fold ensemble if present)
and performs online inference using the exact same ``extract_ai_features``
pipeline used during training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "04_ai_model"))
sys.path.insert(0, str(PROJECT_ROOT / "02_phasebeat_dsp"))

from config.settings import DATA_DIR
from dataset_prep import ACTIVE_SUBCARRIERS, CLASS_NAMES
from models import CSI_RespirationNet
from respiration_pipeline import extract_ai_features


class CSIRespirationAIInference:
    """Wrapper for real-time inference using the trained Dual-Branch model ensemble."""

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        device: str = "cpu",
        window_sec: float = 16.0,
        target_fs: float = 90.0,
    ) -> None:
        self.device = torch.device(device)
        self.window_sec = float(window_sec)
        self.target_fs = float(target_fs)
        self.class_names = CLASS_NAMES
        self.is_ready = False
        self.models: List[CSI_RespirationNet] = []
        self.checkpoint_info: Dict[str, Any] = {}

        if weights_path is None:
            weights_path = DATA_DIR / "weights" / "best_csi_model.pt"

        self.weights_path = Path(weights_path)
        self._load_model()

    def _load_model(self) -> None:
        """Load model architecture and pre-trained weights (5-fold ensemble if available)."""
        if not self.weights_path.is_file():
            print(f"[!] AI Warning: Weights file not found at {self.weights_path}")
            self.is_ready = False
            return

        try:
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            self.window_sec = float(ckpt.get("window_sec", self.window_sec))
            self.target_fs = float(ckpt.get("target_fs", self.target_fs))
            spec_dim = int(ckpt.get("spec_dim", 131))
            aux_dim = int(ckpt.get("aux_dim", 6))

            state_dicts = ckpt.get("ensemble_state_dicts") or [ckpt["model_state_dict"]]
            self.models = []
            for sd in state_dicts:
                m = CSI_RespirationNet(
                    in_channels=len(ACTIVE_SUBCARRIERS),
                    spec_dim=spec_dim,
                    aux_dim=aux_dim,
                    num_classes=len(self.class_names),
                ).to(self.device)
                m.load_state_dict(sd)
                m.eval()
                self.models.append(m)

            self.is_ready = True
            self.checkpoint_info = {
                "epoch": ckpt.get("epoch", 0),
                "val_f1": ckpt.get("val_f1", 0.0),
                "val_acc": ckpt.get("val_acc", 0.0),
                "n_models": len(self.models),
            }
            print(
                f"[v] AI Ensemble loaded: {self.weights_path.name} "
                f"({len(self.models)} models, 5-Fold Unseen Acc: {self.checkpoint_info['val_acc']*100:.1f}%, "
                f"Macro F1: {self.checkpoint_info['val_f1']:.3f})"
            )
        except Exception as exc:
            print(f"[!] AI Error: Failed to load checkpoint: {exc}")
            self.is_ready = False

    def predict_window(
        self,
        uniform_raw_amps: np.ndarray,
        current_fs: float = 90.0,
    ) -> Dict[str, Any]:
        """Perform inference on uniformly sampled raw CSI amplitudes (latest 16.0s window).

        Parameters
        ----------
        uniform_raw_amps : np.ndarray
            Uniformly resampled CSI amplitudes (samples, 64) or (samples, 46) at ``current_fs`` Hz.
        current_fs : float
            Uniform sampling frequency (default 90.0 Hz).
        """
        fallback_res: Dict[str, Any] = {
            "class_name": "unknown",
            "confidence": 0.0,
            "probabilities": {c: 0.25 for c in self.class_names},
            "pred_bpm": 0.0,
            "is_apnea": False,
        }

        if not self.is_ready or not self.models:
            return fallback_res

        try:
            if uniform_raw_amps.shape[0] < int(4.0 * current_fs):
                return fallback_res

            xw, xs, xa, dsp_res = extract_ai_features(
                uniform_raw_amps,
                fs=current_fs,
                window_sec=self.window_sec,
            )

            tw = torch.tensor(xw, dtype=torch.float32, device=self.device).unsqueeze(0)
            ts = torch.tensor(xs, dtype=torch.float32, device=self.device).unsqueeze(0)
            ta = torch.tensor(xa, dtype=torch.float32, device=self.device).unsqueeze(0)

            probs_accum = np.zeros(len(self.class_names), dtype=np.float64)
            bpm_accum = 0.0

            with torch.no_grad():
                for m in self.models:
                    logits, pb = m(tw, ts, ta)
                    probs_accum += F.softmax(logits, dim=1)[0].cpu().numpy()
                    bpm_accum += float(pb.item())

            probs = probs_accum / len(self.models)
            raw_bpm = bpm_accum / len(self.models)

            cls_idx = int(np.argmax(probs))
            confidence = float(probs[cls_idx])
            predicted_class = self.class_names[cls_idx]
            predicted_bpm = max(0.0, raw_bpm) if predicted_class != "apnea" else 0.0

            prob_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

            return {
                "class_name": predicted_class,
                "confidence": confidence,
                "probabilities": prob_dict,
                "pred_bpm": predicted_bpm,
                "is_apnea": (predicted_class == "apnea"),
            }
        except Exception:
            return fallback_res
