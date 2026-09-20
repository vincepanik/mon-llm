"""
Configuration de base.

Tous les paramètres du projet vivent ici, dans une dataclass. Chaque fichier
d'expérience (debug_mac.py, run_150m.py, ...) importe cette base et ne change
que ce qui diffère. Le script d'entraînement, lui, ne change jamais.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Identité du run
    run_name: str = "base"
    seed: int = 1337

    # Données
    data_dir: Path = Path("data")
    train_file: str = "train.bin"
    val_file: str = "val.bin"
    tokenizer_path: Path = Path("tokenizer/vocab.json")
    vocab_size: int = 256          # 256 = niveau octet, sans BPE

    # Architecture
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128          # longueur de contexte
    dropout: float = 0.0
    bias: bool = False

    # Entraînement
    batch_size: int = 16
    grad_accum_steps: int = 1      # batch effectif = batch_size * grad_accum_steps
    max_steps: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Évaluation et sauvegarde
    eval_interval: int = 50
    eval_steps: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: Path = Path("checkpoints")

    # Matériel
    device: str = "auto"           # auto | cuda | mps | cpu
    use_bf16: bool = False         # CUDA uniquement
    use_compile: bool = False      # torch.compile, CUDA uniquement
    use_flash_attn: bool = False   # CUDA uniquement
    throttle: float = 1.0          # 1.0 = plein régime, 0.7 = 30 % de pause entre les étapes

    # Journalisation
    log_interval: int = 10
    log_dir: Path = Path("logs")

    def validate(self) -> None:
        assert 0.0 < self.throttle <= 1.0, "throttle doit être entre 0 et 1"
        assert self.n_embd % self.n_head == 0, "n_embd doit être divisible par n_head"
        assert self.warmup_steps < self.max_steps

    @property
    def run_dir(self) -> Path:
        return self.checkpoint_dir / self.run_name
