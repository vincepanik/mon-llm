"""
Brique 2 : le transformer.

Version 1 (étape 2) : la plus simple possible, celle du papier original adaptée
au style GPT.
  - embeddings de tokens + embeddings de positions
  - N blocs identiques : [LayerNorm -> attention causale -> résidu -> LayerNorm -> MLP -> résidu]
  - LayerNorm final, puis projection vers le vocabulaire

Version 2 (étape 4) : on remplace brique par brique, en mesurant à chaque fois,
  - positions apprises -> RoPE
  - LayerNorm -> RMSNorm
  - MLP GELU -> SwiGLU
  - attention multi-têtes -> GQA
  - attention naïve -> FlashAttention (CUDA uniquement, derrière cfg.use_flash_attn)

Ce fichier est un squelette. On le remplit ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError("brique 2, à écrire ensemble")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError("brique 2, à écrire ensemble")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError("brique 2, à écrire ensemble")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("brique 2, à écrire ensemble")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx : (batch, temps) identifiants de tokens
        targets : (batch, temps) tokens suivants, ou None en génération
        Renvoie (logits, loss). loss vaut None si targets est None.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizer(self, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
        """AdamW, avec weight decay sur les matrices seulement, pas sur les biais ni les normes."""
        raise NotImplementedError
