"""
Recherche dans Wikipédia pour Carl (RAG : retrieval-augmented generation).

À 125M paramètres, Carl ne retient qu'environ la moitié des faits simples
(le modèle de base en complète 20 sur 40). Au lieu de lui demander de tout
savoir, on cherche le passage de Wikipédia qui parle de la question et on le
lui donne à lire : il n'a plus qu'à y trouver la réponse, ce qu'un petit
modèle sait bien faire. C'est ce que font les assistants modernes pour éviter
d'inventer.

    python rag.py --construire        # une fois : indexe data/big/raw/wikipedia_fr.txt
    python rag.py "Quelle est la capitale du Japon ?"

L'index porte sur le début de chaque article (titre + ~900 caractères) : c'est
là que Wikipédia met l'essentiel, et 1,36 million de passages tiennent en
quelques Go. Recherche par BM25 (mots-clés), avec les mots vides et la
racinisation du français.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import bm25s
import Stemmer

SOURCE = Path("data/big/raw/wikipedia_fr.txt")
INDEX = Path("data/big/rag/index")
SEPARATEUR = "<|endoftext|>"
LONGUEUR = 900  # caractères de texte par passage, en plus du titre


def passage(document: str) -> str | None:
    """Titre + début du texte, coupé à la fin d'une phrase."""
    titre, _, texte = document.strip().partition("\n")
    texte = re.sub(r"\s+", " ", texte).strip()
    if len(texte) < 200:
        return None
    if len(texte) > LONGUEUR:
        coupe = texte.rfind(". ", 0, LONGUEUR)
        texte = texte[: coupe + 1] if coupe > LONGUEUR // 2 else texte[:LONGUEUR]
    return f"{titre.strip()}\n{texte}"


def documents(source: Path = SOURCE):
    """Les articles un par un, en flux (le fichier fait 5 Go)."""
    lignes: list[str] = []
    with source.open(encoding="utf-8") as f:
        for ligne in f:
            if ligne.strip() == SEPARATEUR:
                if lignes:
                    yield "".join(lignes)
                lignes = []
            else:
                lignes.append(ligne)
    if lignes:
        yield "".join(lignes)


def construire() -> None:
    t0 = time.time()
    corpus = [p for d in documents() if (p := passage(d))]
    print(f"{len(corpus):,} passages en {time.time() - t0:.0f} s", flush=True)
    stemmer = Stemmer.Stemmer("french")
    jetons = bm25s.tokenize(corpus, stopwords="fr", stemmer=stemmer, show_progress=True)
    moteur = bm25s.BM25()
    moteur.index(jetons, show_progress=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    moteur.save(str(INDEX), corpus=corpus)
    print(f"index écrit dans {INDEX} en {(time.time() - t0) / 60:.0f} min")


@lru_cache(maxsize=1)
def _moteur():
    return bm25s.BM25.load(str(INDEX), load_corpus=True, mmap=True), Stemmer.Stemmer("french")


def disponible() -> bool:
    return INDEX.exists()


def chercher(question: str, k: int = 1) -> list[tuple[str, float]]:
    """Les k passages les plus proches de la question, avec leur score BM25."""
    moteur, stemmer = _moteur()
    requete = bm25s.tokenize([question], stopwords="fr", stemmer=stemmer, show_progress=False)
    if not requete.vocab:  # que des mots vides (« bonjour », « merci »...)
        return []
    resultats, scores = moteur.retrieve(requete, k=k, show_progress=False)
    sortie = []
    for doc, score in zip(resultats[0], scores[0]):
        texte = doc["text"] if isinstance(doc, dict) else str(doc)
        sortie.append((texte, float(score)))
    return sortie


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--construire", action="store_true")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    if args.construire:
        construire()
    if args.question:
        for texte, score in chercher(args.question, args.k):
            print(f"[{score:.1f}] {texte[:200]}\n")


if __name__ == "__main__":
    sys.exit(main())
