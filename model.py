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

Conventions de nommage des tenseurs, valables partout dans ce fichier :
  B = batch (nombre de séquences traitées en parallèle)
  T = temps (position dans la séquence, de 0 à block_size - 1)
  C = canaux (n_embd, la largeur du modèle)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import Config


class CausalSelfAttention(nn.Module):
    """
    Attention multi-têtes, causale.

    Chaque position fabrique une question (query), une étiquette (key) et un
    contenu (value). On compare la question de chaque position aux étiquettes de
    toutes les positions précédentes : plus ça correspond, plus le contenu de
    cette position-là pèse dans le résultat. « Causale » veut dire qu'on regarde
    uniquement vers l'arrière, sinon le modèle lirait la réponse qu'on lui
    demande de deviner.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # Les trois projections q, k, v en une seule matrice : un seul gros
        # produit matriciel est plus rapide que trois petits.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Projection de sortie, qui remélange ce que les têtes ont trouvé.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # FlashAttention : même calcul, mais fusionné en un seul noyau CUDA qui
        # n'écrit jamais la matrice T x T en mémoire. Réservé à l'étape 4.
        self.flash = cfg.use_flash_attn and hasattr(F, "scaled_dot_product_attention")

        if not self.flash:
            # Masque triangulaire : mask[i, j] vaut 1 si la position i a le droit
            # de regarder la position j, c'est-à-dire si j <= i.
            # register_buffer : ça suit le modèle sur le GPU mais ce n'est pas un
            # paramètre, ça ne s'apprend pas.
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        head_size = C // self.n_head

        # Une passe, trois tenseurs de taille (B, T, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # On découpe les canaux en n_head têtes, et on met la tête en dimension 1
        # pour que les produits matriciels suivants traitent les têtes en parallèle.
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # Chaque question contre chaque étiquette : (B, n_head, T, T).
            # La division par sqrt(head_size) empêche les scores de devenir énormes
            # quand les têtes sont larges, ce qui écraserait le softmax.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
            # Interdit de regarder vers l'avant : -inf devient 0 après le softmax.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Moyenne des contenus, pondérée par les scores.
            y = att @ v  # (B, n_head, T, head_size)

        # On recolle les têtes : (B, n_head, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """
    Le petit réseau qui « digère » ce que l'attention a rapporté.

    On élargit à 4x, on applique une non-linéarité, on rétrécit. C'est là que
    vivent les deux tiers des paramètres du modèle.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """
    Un étage du transformer : attention puis MLP, chacun en pre-norm et en résidu.

    Résidu (`x = x + f(x)`) : chaque étage propose une correction, il ne réécrit
    pas tout. C'est ce qui permet d'empiler des dizaines d'étages sans que le
    gradient se perde en route.

    Pre-norm (LayerNorm avant, pas après) : plus stable à l'entraînement que
    l'ordre du papier original, et c'est ce que fait GPT-2.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Le modèle complet.

    Entrée : des identifiants de tokens (B, T).
    Sortie : pour chaque position, un score par token du vocabulaire, c'est-à-dire
    la prédiction du token suivant.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # wte : chaque token devient un vecteur de n_embd nombres, appris.
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # wpe : chaque position (0, 1, 2, ...) aussi. Sans ça, l'attention verrait
        # la phrase comme un sac de mots, sans ordre. RoPE remplacera ça à l'étape 4.
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Poids partagés entre l'entrée et la sortie : la matrice qui transforme
        # un token en vecteur sert aussi, transposée, à transformer un vecteur en
        # scores. Moins de paramètres, et un peu mieux en pratique.
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Les projections de sortie (attention et MLP) écrivent dans le résidu.
        # Avec n_layer étages qui s'additionnent, on réduit leur amplitude
        # initiale pour que la variance ne grossisse pas avec la profondeur.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx : (batch, temps) identifiants de tokens
        targets : (batch, temps) tokens suivants, ou None en génération
        Renvoie (logits, loss). loss vaut None si targets est None.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"séquence de {T} tokens, mais le contexte du modèle est de {self.cfg.block_size}"
        )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # En génération, seule la dernière position sert. Calculer les logits
            # des autres serait du travail jeté.
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

        logits = self.lm_head(x)  # (B, T, vocab_size)
        # cross_entropy veut du (N, vocab) contre du (N,) : on aplatit batch et temps.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Complète `idx` token par token. Chaque token tiré est réinjecté en entrée.

        temperature : < 1 rend le modèle prudent et répétitif, > 1 le rend audacieux.
        top_k : ne tire que parmi les k tokens les plus probables, pour éviter
        qu'un token absurde sorte par malchance.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # Le modèle ne voit que block_size tokens : on garde la fin.
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                seuil = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < seuil, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx

    def configure_optimizer(self, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
        """AdamW, avec weight decay sur les matrices seulement, pas sur les biais ni les normes."""
        params = [p for p in self.parameters() if p.requires_grad]
        # Une matrice (dim >= 2) participe à un produit matriciel : on la régularise.
        # Un biais ou un gain de LayerNorm (dim < 2) ne doit pas être tiré vers zéro.
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        print(
            f"optimizer : {len(decay)} tenseurs avec decay ({sum(p.numel() for p in decay):,} params), "
            f"{len(no_decay)} sans ({sum(p.numel() for p in no_decay):,} params)"
        )
        # fused : les mises à jour de tous les tenseurs en un seul noyau CUDA.
        fused = device.type == "cuda"
        return torch.optim.AdamW(
            groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused
        )
