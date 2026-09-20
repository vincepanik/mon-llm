"""
Prépare les données : texte brut -> fichiers binaires de tokens.

Entrée  : data/raw/*.txt  (ignoré par git)
Sortie  : data/train.bin et data/val.bin, tableaux numpy uint16 (ou uint8 si vocab <= 256)

Ce script tourne sur CPU, donc sur le Mac ou sur le VPS. Jamais sur un GPU loué.
On ne l'exécute qu'une fois par corpus ; les .bin sont ensuite envoyés une seule
fois sur le volume persistant de la machine louée.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer import BPETokenizer  # noqa: E402
from utils import load_config  # noqa: E402


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

    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    print(f"{len(files)} fichier(s), {len(text):,} caractères")

    if cfg.vocab_size <= 256:
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    else:
        tok = BPETokenizer.load(cfg.tokenizer_path)
        assert tok.vocab_size == cfg.vocab_size, (tok.vocab_size, cfg.vocab_size)
        ids = np.array(tok.encode(text), dtype=np.uint16)

    n_val = int(len(ids) * args.val_ratio)
    train, val = ids[:-n_val], ids[-n_val:]
    train.tofile(cfg.data_dir / cfg.train_file)
    val.tofile(cfg.data_dir / cfg.val_file)
    print(f"train : {len(train):,} tokens, val : {len(val):,} tokens")


if __name__ == "__main__":
    main()
