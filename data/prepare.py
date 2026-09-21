"""
Prépare les données : texte brut -> fichiers binaires de tokens.

Entrée  : data/raw/*.txt  (ignoré par git), un document par bloc, séparés par
          une ligne <|endoftext|> (c'est ce qu'écrit data/download_fineweb.py)
Sortie  : data/train.bin et data/val.bin, tableaux numpy uint16 (ou uint8 si vocab <= 256)

Tout est fait en flux : on lit les documents un par un, on les encode par
paquets, on écrit au fur et à mesure. La mémoire ne dépend pas de la taille du
corpus : 12 Go passent sur le Mac. L'encodage passe par tiktoken (Rust), avec
exactement les fusions apprises par tokenizer/bpe.py.

Un document sur 1/val_ratio part dans val.bin, les autres dans train.bin. On
sépare par document, jamais au milieu d'un texte, pour que la validation ne
voie rien de ce que l'entraînement a vu.

Ce script tourne sur CPU, donc sur le Mac ou sur le VPS. Jamais sur un GPU loué.
On ne l'exécute qu'une fois par corpus ; les .bin sont ensuite envoyés une seule
fois sur le volume persistant de la machine louée.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer import BPETokenizer  # noqa: E402
from utils import load_config  # noqa: E402

SEPARATEUR = "<|endoftext|>"
DOCS_PAR_PAQUET = 1000


def documents(files: list[Path]) -> Iterator[str]:
    """Lit les fichiers ligne à ligne et rend les documents un par un."""
    lignes: list[str] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for ligne in fh:
                if ligne.strip() == SEPARATEUR:
                    doc = "".join(lignes).strip()
                    if doc:
                        yield doc
                    lignes = []
                else:
                    lignes.append(ligne)
        # Fin de fichier = fin de document, même sans séparateur.
        doc = "".join(lignes).strip()
        if doc:
            yield doc
        lignes = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = cfg.data_dir / "raw"
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        sys.exit(f"Aucun fichier .txt dans {raw_dir}. Dépose ton corpus là.")

    if cfg.vocab_size <= 256:
        # Niveau octet, sans BPE. Une ligne vide marque la fin d'un document.
        dtype = np.uint8

        def encoder(docs: list[str]) -> list[list[int]]:
            return [list((d + "\n\n").encode("utf-8")) for d in docs]

    else:
        assert cfg.vocab_size <= 65536, "uint16 ne va pas au-delà de 65536 tokens"
        dtype = np.uint16
        tok = BPETokenizer.load(cfg.tokenizer_path)
        assert tok.vocab_size == cfg.vocab_size, (tok.vocab_size, cfg.vocab_size)
        enc = tok.as_tiktoken()
        eot = tok.eot

        def encoder(docs: list[str]) -> list[list[int]]:
            return [ids + [eot] for ids in enc.encode_ordinary_batch(docs)]

    # Un document sur `un_sur` va dans val : le 0, le 100, le 200, ...
    un_sur = max(2, round(1 / args.val_ratio))
    n_tokens = {"train": 0, "val": 0}
    n_docs = 0
    t0 = time.time()

    with (cfg.data_dir / cfg.train_file).open("wb") as f_train, \
         (cfg.data_dir / cfg.val_file).open("wb") as f_val:

        def ecrire(paquet: list[str], premier: int) -> None:
            encodes = encoder(paquet)
            train, val = [], []
            for j, ids in enumerate(encodes):
                (val if (premier + j) % un_sur == 0 else train).extend(ids)
            np.array(train, dtype=dtype).tofile(f_train)
            np.array(val, dtype=dtype).tofile(f_val)
            n_tokens["train"] += len(train)
            n_tokens["val"] += len(val)

        paquet: list[str] = []
        for doc in documents(files):
            paquet.append(doc)
            if len(paquet) == DOCS_PAR_PAQUET:
                ecrire(paquet, n_docs)
                n_docs += len(paquet)
                paquet = []
                if n_docs % (100 * DOCS_PAR_PAQUET) == 0:
                    total = n_tokens["train"] + n_tokens["val"]
                    print(f"  {n_docs:,} documents, {total / 1e6:.0f}M tokens, {time.time() - t0:.0f} s")
        if paquet:
            ecrire(paquet, n_docs)
            n_docs += len(paquet)

    print(f"{len(files)} fichier(s), {n_docs:,} documents, {time.time() - t0:.0f} s")
    print(f"train : {n_tokens['train']:,} tokens, val : {n_tokens['val']:,} tokens")


if __name__ == "__main__":
    main()
