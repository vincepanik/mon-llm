"""
Étape 5, deuxième partie : DPO (Direct Preference Optimization).

    python data/download_comparia.py
    python dpo.py --checkpoint checkpoints/carl/best.pt

Pour une même question, une réponse préférée par un humain (« chosen ») et une
réponse rejetée (« rejected »). On pousse le modèle à rendre la préférée plus
probable que la rejetée, mesuré par rapport à une copie figée du modèle de
départ (la « référence ») :

    loss = -log sigmoid( beta * [ (log p(chosen) - log p_ref(chosen))
                                 - (log p(rejected) - log p_ref(rejected)) ] )

La référence est un garde-fou : le modèle peut changer ses préférences, mais
chaque écart par rapport à ce qu'il savait faire coûte. beta règle la force de
ce rappel (petit beta : il peut s'éloigner davantage).

Pas de juge, pas de note : seulement des choix humains, ici ceux de compar:IA.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from chat_format import debut_de_reponse
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint, set_seed


def encoder_paire(tok, ex, max_len: int):
    """(ids_chosen, debut_chosen, ids_rejected, debut_rejected) ; None si trop long."""
    invite = debut_de_reponse(tok, ex["prompt"])
    fin = tok.special_tokens["<|im_end|>"]
    c = invite + tok.encode(ex["chosen"]) + [fin]
    r = invite + tok.encode(ex["rejected"]) + [fin]
    if len(c) > max_len or len(r) > max_len:
        return None
    return c, len(invite), r, len(invite)


def logp_reponses(model, sequences: list[tuple[list[int], int]], pad: int, device) -> torch.Tensor:
    """Somme des log-probabilités des tokens de réponse, pour chaque séquence."""
    L = max(len(s) for s, _ in sequences)
    x = torch.full((len(sequences), L), pad, dtype=torch.long)
    masque = torch.zeros((len(sequences), L), dtype=torch.bool)
    for i, (s, debut) in enumerate(sequences):
        x[i, : len(s)] = torch.tensor(s)
        masque[i, debut - 1 : len(s) - 1] = True  # position t prédit le token t+1
    x, masque = x.to(device), masque.to(device)
    logits, _ = model(x, x)  # des cibles quelconques : on veut juste tous les logits
    logp = F.log_softmax(logits.float(), dim=-1)
    suivants = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    par_token = logp.gather(-1, suivants.unsqueeze(-1)).squeeze(-1)
    return (par_token * masque).sum(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="modèle après SFT")
    parser.add_argument("--data", default="data/sft/comparia_dpo.json")
    parser.add_argument("--out", default="checkpoints/carl_dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--paires-par-pas", type=int, default=16)
    parser.add_argument("--lot", type=int, default=2, help="paires traitées à la fois (mémoire du Mac)")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-paires", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    cfg = ck["config"]
    politique = GPT(cfg).to(device)
    politique.load_state_dict(ck["model"])
    reference = GPT(cfg).to(device)
    reference.load_state_dict(ck["model"])
    reference.eval().requires_grad_(False)
    tok = BPETokenizer.load(cfg.tokenizer_path)
    pad = tok.special_tokens["<|pad|>"]

    brut = json.loads(Path(args.data).read_text(encoding="utf-8"))[: args.max_paires]
    paires = [p for p in (encoder_paire(tok, ex, args.max_len) for ex in brut) if p]
    n_val = max(20, len(paires) // 20)
    val, train = paires[:n_val], paires[n_val:]
    print(f"{len(train):,} paires d'entraînement, {len(val)} de validation, beta={args.beta}", flush=True)

    def marges(model_pol, lot):
        seqs = [(c, dc) for c, dc, _, _ in lot] + [(r, dr) for _, _, r, dr in lot]
        lp = logp_reponses(model_pol, seqs, pad, device)
        with torch.no_grad():
            lr_ = logp_reponses(reference, seqs, pad, device)
        n = len(lot)
        # Gain de la préférée moins gain de la rejetée, par rapport à la référence.
        return (lp[:n] - lr_[:n]) - (lp[n:] - lr_[n:])

    @torch.no_grad()
    def evaluer():
        politique.eval()
        m = torch.cat([marges(politique, val[i : i + args.lot]) for i in range(0, len(val), args.lot)])
        politique.train()
        # Précision : part des paires où le modèle préfère désormais la bonne réponse.
        return float((-F.logsigmoid(args.beta * m)).mean()), float((m > 0).float().mean())

    opt = torch.optim.AdamW([p for p in politique.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    accum = max(1, args.paires_par_pas // args.lot)
    total = math.ceil(len(train) / args.lot / accum)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loss_v, acc_v = evaluer()
    print(f"avant DPO : loss {loss_v:.4f}, préfère la bonne réponse {acc_v:.0%} du temps ({total} pas)", flush=True)
    random.Random(args.seed).shuffle(train)
    t0 = time.time()
    for pas in range(total):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (pas + 1) / max(1, total // 10))  # warmup puis constant
        for k in range(accum):
            i = (pas * accum + k) * args.lot
            lot = train[i : i + args.lot]
            if not lot:
                break
            loss = (-F.logsigmoid(args.beta * marges(politique, lot))).mean()
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "mps" and (pas + 1) % 5 == 0:
            torch.mps.empty_cache()  # même raison que dans sft.py
        if (pas + 1) % 20 == 0 or pas + 1 == total:
            loss_v, acc_v = evaluer()
            print(f"pas {pas + 1:4d}/{total} | val loss {loss_v:.4f} | préfère la bonne {acc_v:.0%} | "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)

    torch.save({"model": politique.state_dict(), "config": cfg, "step": total, "best_val": loss_v}, out / "best.pt")
    print(f"terminé -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
