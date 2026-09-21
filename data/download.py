"""
Récupère un échantillon de corpus français, une source à la fois.

    python data/download.py --source wikipedia --mb 20
    python data/download.py --source science    --mb 20
    python data/download.py --source fineweb_hq --mb 20

Sources :
  wikipedia  Wikipédia FR (wikimedia/wikipedia, dump 2023-11). Licence CC-BY-SA.
             C'est la part « OpenWeb » française de Common Corpus (PleIAs).
  science    Articles scientifiques français en accès ouvert
             (PleIAs/French-Science-Commons, 37 Go, surtout 2010-2024). Licence
             CC-BY. C'est la part « OpenScience » française de Common Corpus.
  fineweb_hq Les 10 % meilleurs documents de FineWeb-2 français, choisis par un
             classifieur de qualité (epfml/FineWeb2-HQ, licence ODC-By). C'est
             la base du mélange de l'étape 3. Collectes >= --depuis.
  fineweb    Web français récent (HuggingFaceFW/fineweb-2), collectes Common
             Crawl >= --depuis. Pas de licence : le web tel quel.

Tout est lu en streaming : rien n'est stocké en entier sur le disque, on
s'arrête dès qu'on a la quantité de texte demandée. La première requête sur une
source peut mettre quelques minutes (HuggingFace liste ses fichiers), la suite
défile vite. On mélange l'ordre des fichiers pour ne pas lire que le début.

Le fichier de sortie contient un document par bloc, séparés par une ligne
<|endoftext|> : data/prepare.py la remplace par le vrai token de fin, pour que
le modèle apprenne où un texte commence et se termine.

Ce script tourne sur CPU, donc sur le Mac ou sur le VPS. Jamais sur un GPU loué :
sur une machine louée, on envoie le .txt (ou mieux, le .bin déjà tokenisé) sur le
volume persistant, on ne le retélécharge pas.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset

SEPARATEUR = "<|endoftext|>"


def wikipedia(seed: int, **_) -> Iterator[str]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    for ex in ds:
        yield f"{ex['title']}\n\n{ex['text']}"


def science(seed: int, **_) -> Iterator[str]:
    ds = load_dataset("PleIAs/French-Science-Commons", split="train", streaming=True)
    # Une ligne = une page d'article. Selon le fichier, les pages d'un même
    # article se suivent, ou bien sont entrelacées avec celles d'une dizaine
    # d'autres (page 1 de chacun, puis page 2 de chacun, ...). On garde donc
    # les articles en cours dans une fenêtre, et on recolle un article quand il
    # est sorti de la fenêtre : ses pages sont toutes passées.
    ds = ds.shuffle(seed=seed, buffer_size=1)  # mélange les fichiers, pas les lignes
    fenetre = 1000
    en_cours: dict[str, list[tuple[int, str]]] = {}
    for ex in ds:
        if ex["language"] != "fr":  # quelques articles en anglais malgré la source
            continue
        en_cours.setdefault(ex["id"], []).append((int(ex["page"]), ex["text"].strip()))
        if len(en_cours) > fenetre:
            plus_ancien = next(iter(en_cours))
            pages = en_cours.pop(plus_ancien)
            yield "\n\n".join(t for _, t in sorted(pages))
    for pages in en_cours.values():
        yield "\n\n".join(t for _, t in sorted(pages))


def fineweb(seed: int, depuis: int, **_) -> Iterator[str]:
    # FineWeb-2 range ses collectes Common Crawl dans l'ordre chronologique
    # (2013 -> 2024) : sans mélange ni filtre, on ne lirait que du web de 2013.
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    for ex in ds:
        if int(ex["dump"][8:12]) >= depuis:  # "CC-MAIN-2024-10" -> 2024
            yield ex["text"]


def fineweb_hq(seed: int, depuis: int, **_) -> Iterator[str]:
    # Même organisation chronologique que FineWeb-2. Chaque ligne embarque un
    # vecteur d'embedding de plusieurs Ko : on ne lit que les colonnes utiles,
    # sinon on téléchargerait dix fois plus que le texte.
    ds = load_dataset(
        "epfml/FineWeb2-HQ", data_files="fra_Latn/*.parquet", split="train",
        streaming=True, columns=["text", "dump"],
    )
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    for ex in ds:
        if int(ex["dump"][8:12]) >= depuis:
            yield ex["text"]


SOURCES = {"wikipedia": wikipedia, "science": science, "fineweb_hq": fineweb_hq, "fineweb": fineweb}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--mb", type=float, default=20.0, help="taille visée, en mégaoctets")
    parser.add_argument("--out", default=None, help="défaut : data/raw/<source>_fr.txt")
    parser.add_argument("--min-chars", type=int, default=500,
                        help="on jette les documents plus courts (ébauches, pages vides)")
    parser.add_argument("--depuis", type=int, default=2022,
                        help="fineweb / fineweb_hq : année de collecte minimale")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cible = int(args.mb * 1024 * 1024)
    out = Path(args.out or f"data/raw/{args.source}_fr.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.source} -> {out}, cible {args.mb:.0f} Mo")
    print("résolution des fichiers du dataset, quelques minutes la première fois...", flush=True)

    ecrits = gardes = jetes = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for texte in SOURCES[args.source](seed=args.seed, depuis=args.depuis):
            # Si un document parle de notre séparateur, on ne veut pas qu'il le contienne.
            texte = texte.replace(SEPARATEUR, "").strip()
            if len(texte) < args.min_chars:
                jetes += 1
                continue
            f.write(texte + "\n" + SEPARATEUR + "\n")
            ecrits += len(texte.encode("utf-8")) + len(SEPARATEUR) + 2
            gardes += 1
            if gardes % 2000 == 0:
                print(f"  {ecrits / 1024 / 1024:6.1f} Mo | {gardes:,} documents | "
                      f"{time.time() - t0:.0f} s", flush=True)
            if ecrits >= cible:
                break

    print(f"terminé : {out.stat().st_size / 1024 / 1024:.1f} Mo, {gardes:,} documents gardés, "
          f"{jetes:,} jetés (trop courts), {time.time() - t0:.0f} s", flush=True)

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
