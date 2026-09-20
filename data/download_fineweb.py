"""
Récupère un échantillon de FineWeb-2 français (web moderne, nettoyé, dédupliqué).

    python data/download_fineweb.py --mb 20                    # pour déboguer sur le Mac
    python data/download_fineweb.py --mb 12000 --out data/raw/fineweb_fr.txt   # étape 3

On lit le dataset en streaming : rien n'est stocké en entier sur le disque, on
s'arrête dès qu'on a la quantité de texte demandée. La première requête met
quelques minutes (HuggingFace liste les milliers de fichiers du dataset), la
suite défile vite. La liste est ensuite en cache dans ~/.cache/huggingface.

Ce script tourne sur CPU, donc sur le Mac ou sur le VPS. Jamais sur un GPU loué :
sur une machine louée, on envoie le .txt (ou mieux, le .bin déjà tokenisé) sur le
volume persistant, on ne le retélécharge pas.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "fra_Latn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=float, default=20.0, help="taille visée, en mégaoctets")
    parser.add_argument("--out", default="data/raw/fineweb2_fr.txt")
    parser.add_argument("--min-chars", type=int, default=500,
                        help="on jette les documents plus courts (menus, pages vides)")
    args = parser.parse_args()

    cible = int(args.mb * 1024 * 1024)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{DATASET} ({CONFIG}) -> {out}, cible {args.mb:.0f} Mo")
    print("résolution des fichiers du dataset, quelques minutes la première fois...")

    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)

    ecrits = gardes = jetes = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            texte = ex["text"].strip()
            if len(texte) < args.min_chars:
                jetes += 1
                continue
            # Un document par bloc, séparés par une ligne vide : le modèle apprend
            # ainsi où un texte commence et se termine.
            f.write(texte + "\n\n")
            ecrits += len(texte.encode("utf-8")) + 2
            gardes += 1
            if gardes % 2000 == 0:
                print(f"  {ecrits / 1024 / 1024:6.1f} Mo | {gardes:,} documents | "
                      f"{time.time() - t0:.0f} s")
            if ecrits >= cible:
                break

    print(f"terminé : {out.stat().st_size / 1024 / 1024:.1f} Mo, {gardes:,} documents gardés, "
          f"{jetes:,} jetés (trop courts), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
