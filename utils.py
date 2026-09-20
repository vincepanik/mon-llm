"""
Plomberie partagée : détection du device, dtype, seed, throttle, checkpoints,
chargement d'une config par chemin de fichier.

Rien ici ne dépend de l'architecture du modèle.
"""

from __future__ import annotations

import importlib.util
import random
import time
from pathlib import Path

import numpy as np
import torch

from configs.base import Config


def load_config(path: str | Path) -> Config:
    """Charge un fichier de config Python et renvoie son objet `config`."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg: Config = module.config
    cfg.validate()
    return cfg


def get_device(requested: str = "auto") -> torch.device:
    """cuda si disponible, sinon mps (Apple Silicon), sinon cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device, use_bf16: bool) -> torch.dtype:
    """bf16 uniquement sur CUDA. Partout ailleurs, float32."""
    if use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Throttle:
    """
    Fait dormir la boucle entre les étapes pour limiter la chauffe.

    ratio=1.0 : aucune pause.
    ratio=0.7 : le GPU travaille 70 % du temps, se repose 30 %.

    Usage :
        throttle = Throttle(cfg.throttle)
        for step in ...:
            throttle.start()
            ... étape d'entraînement ...
            throttle.wait()
    """

    def __init__(self, ratio: float):
        self.ratio = ratio
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def wait(self) -> None:
        if self.ratio >= 1.0:
            return
        elapsed = time.perf_counter() - self._t0
        time.sleep(elapsed * (1.0 / self.ratio - 1.0))


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: Config, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
