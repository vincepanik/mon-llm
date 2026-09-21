"""
Récupère un échantillon de FineWeb-2 français (web moderne, nettoyé, dédupliqué).

    python data/download_fineweb.py --mb 20                    # pour déboguer sur le Mac (web >= 2022)
    python data/download_fineweb.py --mb 12000 --out data/raw/fineweb_fr.txt   # étape 3

On lit le dataset en streaming : rien n'est stocké en entier sur le disque, on
s'arrête dès qu'on a la quantité de texte demandée.

FineWeb-2 est construit à partir de collectes successives du web (Common Crawl),
de 2013 à 2024, rangées dans l'ordre : lu tel quel, le début du dataset est
100 % du web de 2013. On mélange donc l'ordre des fichiers et on ne garde que
les collectes à partir de --depuis, pour avoir du français d'aujourd'hui. La première requête met
quelques minutes (HuggingFace liste les milliers de fichiers du dataset), la
suite défile vite. La liste est ensuite en cache dans ~/.cache/huggingface.

Ce script tourne sur CPU, donc sur le Mac ou sur le VPS. Jamais sur un GPU loué :
sur une machine louée, on envoie le .txt (ou mieux, le .bin déjà tokenisé) sur le
volume persistant, on ne le retélécharge pas.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    parser.add_argument("--depuis", type=int, default=2022,
                        help="année de collecte minimale : on jette le web plus ancien")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cible = int(args.mb * 1024 * 1024)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{DATASET} ({CONFIG}) -> {out}, cible {args.mb:.0f} Mo")
    print("résolution des fichiers du dataset, quelques minutes la première fois...")

    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    # Mélange l'ordre des fichiers (et un peu les documents entre eux), sinon on
    # lirait d'abord tous les vieux fichiers pour les jeter un par un.
    ds = ds.shuffle(seed=args.seed, buffer_size=1000)

    ecrits = gardes = jetes = trop_vieux = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            # "CC-MAIN-2024-10" -> 2024
            if int(ex["dump"][8:12]) < args.depuis:
                trop_vieux += 1
                continue
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
                      f"{time.time() - t0:.0f} s", flush=True)
            if ecrits >= cible:
                break

    print(f"terminé : {out.stat().st_size / 1024 / 1024:.1f} Mo, {gardes:,} documents gardés, "
          f"{jetes:,} jetés (trop courts), {trop_vieux:,} jetés (avant {args.depuis}), "
          f"{time.time() - t0:.0f} s", flush=True)

    # Sortie brutale, volontaire. En streaming, `datasets` laisse tourner des
    # threads de lecture qu'on a interrompus avec `break`, et Python attend
    # indéfiniment qu'ils se terminent : le programme ne rend jamais la main
    # (vu en pratique : bloqué 12 h après avoir fini). Le fichier est déjà
    # fermé à ce stade, on ne perd rien.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
