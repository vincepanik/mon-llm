"""
Banc d'essai de la recherche dans Wikipédia (rag.py), sans Carl : le passage
trouvé contient-il la réponse ? Sur les 40 questions de savoirs de l'examen
et les 15 faits de examen_faits.py.

    python examen_rag.py            # mots-clés (BM25), sens (embeddings), hybride

« en tête » : le premier passage contient la réponse ; « top 3 » : l'un des
trois premiers. Carl ne lit qu'un passage : c'est le premier chiffre qui compte.
"""

from __future__ import annotations

import re
import sys

import faits
import rag
from examen import FAITS
from examen_faits import QUESTIONS as JAMAIS_VUES


def questions() -> list[tuple[str, list[str]]]:
    qs = [(q, mots) for q, _, mots in FAITS]
    for q, e, r in JAMAIS_VUES:
        verite = faits.chercher(e, r)
        qs.append((q, [verite.split(" et ")[0].split(",")[0].split()[-1].lower()]))
    return qs


def contient(passage: str, mots: list[str]) -> bool:
    texte = passage.lower()
    return any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", texte) for m in mots)


def main() -> None:
    methodes = [a for a in sys.argv[1:] if not a.startswith("-")] or [m for m in ("mots", "sens", "hybride") if m == "mots" or rag.VECTEURS.exists()]
    qs = questions()
    for methode in methodes:
        tete = top3 = 0
        rates = []
        for q, mots in qs:
            passages = [t for t, _ in rag.chercher(q, k=3, methode=methode)]
            tete += bool(passages) and contient(passages[0], mots)
            top3 += any(contient(p, mots) for p in passages)
            if not (passages and contient(passages[0], mots)):
                rates.append(f"{q[:45]:45} -> {passages[0].split(chr(10))[0][:40] if passages else '(rien)'}")
        print(f"{methode:8} en tête {tete}/{len(qs)}   top 3 {top3}/{len(qs)}")
        if "-v" in sys.argv or len(methodes) == 1:
            print("\n".join("   ✗ " + r for r in rates))


if __name__ == "__main__":
    main()
