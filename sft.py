"""
Étape 5 : SFT (supervised fine-tuning). On part du modèle pré-entraîné et on
lui apprend le format d'une conversation : répondre, puis s'arrêter.

    python data/download_sft.py
    python sft.py --checkpoint checkpoints/run_150m/best.pt

Différences avec train.py :
  - on part d'un checkpoint au lieu de poids aléatoires ;
  - les données sont des conversations (chat_format.py), pas un flux de texte :
    chaque exemple a sa longueur, on complète les lots avec <|pad|> ;
  - la loss ne porte que sur les réponses de l'assistant ;
  - taux d'apprentissage ~10 fois plus bas : on ajuste, on ne réapprend pas
    le français (sinon le modèle oublie ce qu'il sait, « l'oubli catastrophique »).

Le corpus est petit (~300k tokens) : ça tourne sur le Mac en moins d'une heure.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from chat_format import IGNORE, encoder_conversation
from model import GPT
from tokenizer import BPETokenizer
from utils import count_params, get_device, load_checkpoint, set_seed


def lots(exemples: list[tuple[list[int], list[int]]], taille: int, rng: random.Random) -> list[list[int]]:
    """
    Indices d'exemples regroupés en lots de longueurs voisines : on mélange,
    on trie par paquets de 20 lots, on découpe. Moins de remplissage par
    <|pad|>, donc moins de calcul perdu, tout en gardant du hasard.
    """
    idx = list(range(len(exemples)))
    rng.shuffle(idx)
    paquet = taille * 20
    resultat = []
    for i in range(0, len(idx), paquet):
        tranche = sorted(idx[i : i + paquet], key=lambda j: len(exemples[j][0]))
        resultat += [tranche[k : k + taille] for k in range(0, len(tranche), taille)]
    rng.shuffle(resultat)
    return resultat


def assembler(exemples, indices, pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    longueur = max(len(exemples[j][0]) for j in indices)
    x = torch.full((len(indices), longueur), pad, dtype=torch.long)
    y = torch.full((len(indices), longueur), IGNORE, dtype=torch.long)
    for ligne, j in enumerate(indices):
        ids, cibles = exemples[j]
        x[ligne, : len(ids)] = torch.tensor(ids)
        y[ligne, : len(cibles)] = torch.tensor(cibles)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluer(model, exemples, taille, pad, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(exemples), taille):
        indices = list(range(i, min(i + taille, len(exemples))))
        x, y = assembler(exemples, indices, pad, device)
        _, loss = model(x, y)
        k = int((y != IGNORE).sum())
        total, n = total + loss.item() * k, n + k
    model.train()
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="modèle pré-entraîné (étape 3)")
    parser.add_argument("--data", nargs="+", default=["data/sft/conversations.json"],
                        help="un ou plusieurs fichiers de conversations, mélangés")
    parser.add_argument("--out", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    # Réglages par défaut pour un Mac de 16 Go : au-delà, il déborde en swap et
    # devient des dizaines de fois plus lent. Sur un GPU de 24 Go et plus :
    # --batch-size 16 --grad-accum 1 --max-len 1024.
    parser.add_argument("--batch-size", type=int, default=2, help="conversations par lot")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512,
                        help="tokens par conversation au plus (les plus longues sont coupées)")
    parser.add_argument("--extra", default=None,
                        help="conversations en plus (ex. data/identite_carl.json), entraînement seulement")
    parser.add_argument("--extra-repeat", type=int, default=3,
                        help="nombre de copies de --extra : peu nombreuses, elles doivent peser")
    parser.add_argument("--enchainer", type=int, default=0,
                        help="conversations fabriquées : un salut de --extra, puis une vraie "
                             "conversation d'entraînement. Apprend à répondre à la question "
                             "suivante au lieu de recopier le salut.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    cfg = ck["config"]
    cfg.use_compile = False  # quelques centaines d'étapes : la compilation ne rapporterait rien
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"{args.checkpoint} (étape {ck['step']}), {count_params(model) / 1e6:.1f}M paramètres, device={device}")

    tok = BPETokenizer.load(cfg.tokenizer_path)
    pad = tok.special_tokens["<|pad|>"]
    convs = [c for f in args.data for c in json.loads(Path(f).read_text(encoding="utf-8"))]
    if len(args.data) > 1:
        # Mélangés pour que la validation contienne un peu de chaque source.
        random.Random(args.seed).shuffle(convs)
    extra = json.loads(Path(args.extra).read_text(encoding="utf-8")) if args.extra else []

    def encoder(c):
        ids, cibles = encoder_conversation(tok, c)
        # Trop long pour le contexte : on garde le début (souvent la question
        # et le début de la réponse), du moment qu'il reste une réponse à apprendre.
        n = min(args.max_len, cfg.block_size)
        ids, cibles = ids[:n], cibles[:n]
        return (ids, cibles) if any(c != IGNORE for c in cibles) else None

    paires = [(c, e) for c in convs if (e := encoder(c)) is not None]
    n_val = max(1, int(len(paires) * args.val_ratio))
    # La validation ne contient que des conversations générales : elle mesure
    # si le modèle répond mieux, pas s'il a retenu son propre nom.
    val = [e for _, e in paires[:n_val]]
    train = [e for _, e in paires[n_val:]]
    train += [e for e in map(encoder, extra) if e is not None] * args.extra_repeat
    if extra:
        print(f"+ {len(extra)} conversations de {args.extra}, x{args.extra_repeat}")
    if args.enchainer and extra:
        saluts = [c for c in extra if len(c) == 2 and len(c[0]["content"]) <= 25]
        generales = [c for c, _ in paires[n_val:]]  # jamais de validation ici
        r = random.Random(args.seed + 1)
        fabriquees = [r.choice(saluts) + r.choice(generales) for _ in range(args.enchainer)]
        train += [e for e in map(encoder, fabriquees) if e is not None]
        print(f"+ {args.enchainer} conversations enchaînées (salut, puis vraie question)")
    n_rep = sum(sum(1 for c in cib if c != IGNORE) for _, cib in train)
    print(f"{len(train):,} conversations d'entraînement ({n_rep:,} tokens de réponse), {len(val)} de validation")

    opt = model.configure_optimizer(cfg, device)
    rng = random.Random(args.seed)
    pas_par_epoch = math.ceil(math.ceil(len(train) / args.batch_size) / args.grad_accum)
    total_pas = pas_par_epoch * args.epochs
    warmup = max(1, total_pas // 20)

    def lr_a(pas: int) -> float:
        if pas < warmup:
            return args.lr * (pas + 1) / warmup
        progres = (pas - warmup) / max(1, total_pas - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progres)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # On garde la meilleure epoch, même si elle fait moins bien que le point de
    # départ sur la validation : une séance qui renforce l'identité (répétée 10
    # fois) coûte un peu en loss générale, c'est le compromis voulu. Comparé au
    # départ, ce cas ne sauvait rien, et le DPO derrière ne trouvait pas son modèle.
    depart = evaluer(model, val, args.batch_size, pad, device)
    meilleure = float("inf")
    print(f"avant SFT : val {depart:.4f}  ({total_pas} pas sur {args.epochs} epochs)", flush=True)

    pas, t0 = 0, time.time()
    # Mesurée sur le dernier intervalle : sur toute la durée, une pause du
    # processus (kill -STOP, capot fermé) fausserait durablement le chiffre.
    tokens, depuis = 0, time.time()
    for epoch in range(args.epochs):
        tous = lots(train, args.batch_size, rng)
        for i in range(0, len(tous), args.grad_accum):
            for g in opt.param_groups:
                g["lr"] = lr_a(pas)
            groupe = tous[i : i + args.grad_accum]
            for indices in groupe:
                x, y = assembler(train, indices, pad, device)
                _, loss = model(x, y)
                (loss / len(groupe)).backward()
                tokens += x.numel()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pas += 1
            if device.type == "mps" and pas % 10 == 0:
                # Les lots ont tous des longueurs différentes : l'allocateur du GPU
                # garde un bloc de chaque taille « au cas où » et gonfle jusqu'à
                # 12 Go sur un Mac de 16 Go (tout part en swap). On lui fait rendre
                # sa réserve régulièrement.
                torch.mps.empty_cache()
            if pas % 10 == 0:
                dt = time.time() - depuis
                print(f"epoch {epoch + 1} | pas {pas:4d}/{total_pas} | loss {loss.item():.4f} | "
                      f"lr {lr_a(pas):.2e} | {tokens / dt:.0f} tokens/s"
                      + (f" | GPU {torch.mps.current_allocated_memory() / 2**30:.1f} Go" if device.type == "mps" else ""),
                      flush=True)
                tokens, depuis = 0, time.time()

        v = evaluer(model, val, args.batch_size, pad, device)
        print(f"fin de l'epoch {epoch + 1} : val {v:.4f}", flush=True)
        if v < meilleure:
            meilleure = v
            # Sans l'optimiseur : 500 Mo au lieu de 1,5 Go, et c'est tout ce
            # qu'il faut pour discuter avec le modèle.
            torch.save({"model": model.state_dict(), "config": cfg, "step": pas, "best_val": v}, out / "best.pt")
            print(f"  -> {out / 'best.pt'}", flush=True)

    print(f"terminé en {(time.time() - t0) / 60:.0f} min, meilleure val {meilleure:.4f} (départ {depart:.4f})")


if __name__ == "__main__":
    main()
