"""
Brique 2 : le transformer.

Version 1 (étape 2) : la plus simple possible, celle du papier original adaptée
au style GPT.
  - embeddings de tokens + embeddings de positions
  - N blocs identiques : [LayerNorm -> attention causale -> résidu -> LayerNorm -> MLP -> résidu]
  - LayerNorm final, puis projection vers le vocabulaire

Version 2 (étape 4) : on remplace brique par brique, en mesurant à chaque fois.
Chaque remplacement est un interrupteur dans la config, la valeur par défaut
donne la version 1 :
  - cfg.pos  = "rope"     positions apprises -> RoPE
  - cfg.norm = "rmsnorm"  LayerNorm -> RMSNorm
  - cfg.mlp  = "swiglu"   MLP GELU -> SwiGLU
  - cfg.n_kv_head < n_head   attention multi-têtes -> GQA
  - cfg.use_flash_attn    attention naïve -> FlashAttention (CUDA uniquement)

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


# ----------------------------------------------------------------- normalisation


class RMSNorm(nn.Module):
    """
    Comme LayerNorm, sans soustraire la moyenne et sans biais : on divise
    seulement par la norme quadratique moyenne, puis on multiplie par un gain
    appris. Moins de calcul, et en pratique aussi bon. C'est ce qu'utilisent
    Llama, Mistral, Qwen, DeepSeek.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcul en float32 même sous bf16 : la somme des carrés déborde vite.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def make_norm(cfg: Config) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


# ----------------------------------------------------------------------- RoPE


def rope_cache(block_size: int, head_size: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tables cos/sin de RoPE, forme (block_size, head_size).

    Chaque paire de dimensions (i, i + head_size/2) de la tête est vue comme un
    point du plan qu'on fait tourner d'un angle proportionnel à la position.
    Les paires ont des vitesses de rotation différentes : rapides pour les
    premières, très lentes pour les dernières (theta règle l'étalement).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_size / 2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_size)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Fait tourner q ou k, forme (B, n_head, T, head_size).

    Le produit scalaire de deux vecteurs tournés ne dépend que de l'écart entre
    leurs positions : c'est ce qui donne au modèle la notion de distance
    relative, sans rien apprendre, et sans limite de longueur en principe.
    """
    T = x.size(2)
    cos, sin = cos[:T], sin[:T]
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


# ------------------------------------------------------------------ attention


class CausalSelfAttention(nn.Module):
    """
    Attention multi-têtes, causale.

    Chaque position fabrique une question (query), une étiquette (key) et un
    contenu (value). On compare la question de chaque position aux étiquettes de
    toutes les positions précédentes : plus ça correspond, plus le contenu de
    cette position-là pèse dans le résultat. « Causale » veut dire qu'on regarde
    uniquement vers l'arrière, sinon le modèle lirait la réponse qu'on lui
    demande de deviner.

    GQA (grouped-query attention) : plusieurs têtes de questions partagent la
    même paire étiquette/contenu. Ça ne change presque rien à la qualité, mais à
    l'inférence le cache des k et v (ce qui remplit la mémoire du GPU quand on
    génère) est divisé par n_head / n_kv_head.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # q pour toutes les têtes, k et v pour les têtes kv seulement, en une
        # seule matrice : un seul gros produit matriciel plutôt que trois petits.
        self.c_attn = nn.Linear(
            cfg.n_embd, (cfg.n_head + 2 * self.n_kv_head) * self.head_size, bias=cfg.bias
        )
        # Projection de sortie, qui remélange ce que les têtes ont trouvé.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.rope = cfg.pos == "rope"
        if self.rope:
            cos, sin = rope_cache(cfg.block_size, self.head_size, cfg.rope_theta)
            # persistent=False : recalculé à la construction, pas stocké dans le checkpoint.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # FlashAttention : même calcul, mais fusionné en un seul noyau CUDA qui
        # n'écrit jamais la matrice T x T en mémoire.
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
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, nkv, hs = self.n_head, self.n_kv_head, self.head_size

        q, k, v = self.c_attn(x).split([nh * hs, nkv * hs, nkv * hs], dim=2)
        # On découpe les canaux en têtes, et on met la tête en dimension 1 pour
        # que les produits matriciels suivants traitent les têtes en parallèle.
        # (B, T, n_head * hs) -> (B, n_head, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nkv, hs).transpose(1, 2)
        v = v.view(B, T, nkv, hs).transpose(1, 2)

        if self.rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=nkv != nh,
            )
        else:
            if nkv != nh:
                # GQA sur le chemin naïf : on duplique k et v pour que chaque
                # tête de question ait sa tête kv en face.
                k = k.repeat_interleave(nh // nkv, dim=1)
                v = v.repeat_interleave(nh // nkv, dim=1)
            # Chaque question contre chaque étiquette : (B, n_head, T, T).
            # La division par sqrt(head_size) empêche les scores de devenir énormes
            # quand les têtes sont larges, ce qui écraserait le softmax.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            # Interdit de regarder vers l'avant : -inf devient 0 après le softmax.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Moyenne des contenus, pondérée par les scores.
            y = att @ v  # (B, n_head, T, hs)

        # On recolle les têtes : (B, n_head, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ------------------------------------------------------------------------ MLP


class MLP(nn.Module):
    """
    Le petit réseau qui « digère » ce que l'attention a rapporté.

    On élargit à 4x, on applique une non-linéarité, on rétrécit. C'est là que
    vivent les deux tiers des paramètres du modèle.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.mlp_hidden or 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLU(nn.Module):
    """
    Le MLP de Llama. Deux projections montantes au lieu d'une : l'une passe par
    une non-linéarité (SiLU) et sert de porte, elle module l'autre terme à
    terme. Pour garder le même nombre de paramètres qu'un MLP 4x avec trois
    matrices au lieu de deux, la largeur cachée est ~8/3 x n_embd.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # 8/3 x n_embd, arrondi au multiple de 64 supérieur (bon pour les GPU).
        hidden = cfg.mlp_hidden or 64 * math.ceil(8 * cfg.n_embd / 3 / 64)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------- Block


class Block(nn.Module):
    """
    Un étage du transformer : attention puis MLP, chacun en pre-norm et en résidu.

    Résidu (`x = x + f(x)`) : chaque étage propose une correction, il ne réécrit
    pas tout. C'est ce qui permet d'empiler des dizaines d'étages sans que le
    gradient se perde en route.

    Pre-norm (normalisation avant, pas après) : plus stable à l'entraînement que
    l'ordre du papier original, et c'est ce que fait GPT-2.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ------------------------------------------------------------------------ GPT


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
        # la phrase comme un sac de mots, sans ordre. Avec RoPE, l'ordre est
        # injecté directement dans l'attention et wpe disparaît.
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
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

        x = self.wte(idx)  # (B, T, C)
        if self.wpe is not None:
            pos = torch.arange(T, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
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
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Complète `idx` token par token. Chaque token tiré est réinjecté en entrée.

        temperature : < 1 rend le modèle prudent et répétitif, > 1 le rend audacieux.
        top_k : ne tire que parmi les k tokens les plus probables, pour éviter
        qu'un token absurde sorte par malchance.
        stop_token : on s'arrête dès que toutes les séquences l'ont produit (en
        pratique <|endoftext|> : le modèle signale lui-même que son texte est fini).
        Le token d'arrêt reste dans la sortie, à l'appelant de le retirer.
        repetition_penalty : > 1 rend moins probables les tokens déjà écrits
        pendant cette génération (pas ceux de la question : répondre « la
        capitale de l'Italie est Rome » doit rester possible). Contre les
        boucles « la compréhension et la compréhension ».
        """
        was_training = self.training
        self.eval()
        debut = idx.size(1)
        for _ in range(max_new_tokens):
            # Le modèle ne voit que block_size tokens : on garde la fin.
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0 and idx.size(1) > debut:
                deja = idx[:, debut:]
                vus = logits.gather(1, deja)
                # Divisé si positif, multiplié si négatif : dans les deux cas, moins probable.
                vus = torch.where(vus > 0, vus / repetition_penalty, vus * repetition_penalty)
                logits = logits.scatter(1, deja, vus)
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                seuil = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < seuil, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_token is not None and bool((idx_next == stop_token).all()):
                break
        if was_training:
            self.train()
        return idx

    def configure_optimizer(self, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
        """AdamW, avec weight decay sur les matrices seulement, pas sur les biais ni les normes."""
        params = [p for p in self.parameters() if p.requires_grad]
        # Une matrice (dim >= 2) participe à un produit matriciel : on la régularise.
        # Un biais ou un gain de normalisation (dim < 2) ne doit pas être tiré vers zéro.
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
