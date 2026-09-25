"""
Compare deux passages de l'examen question par question, sans relancer Carl.

    python comparer.py resultats/carl_v10.jsonl resultats/carl_v11.jsonl

Les deux fichiers sont renotés avec l'examen actuel (examen.py), puis on liste
les questions gagnées et perdues. Un total ne dit pas si « 24 contre 27 » vient
de trois vraies pertes, ou de deux réponses fausses que la notation comptait
justes : la liste, si.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from examen import TESTS, VOLETS
from notation import texte_note, wilson


def lire(chemin: Path) -> dict[str, tuple[str, str, bool]]:
    """question -> (volet, réponse, juste ?), noté avec l'examen actuel."""
    reponses = {}
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        r = json.loads(ligne)
        if r["question"] in TESTS:
            volet, test = TESTS[r["question"]]
            reponses[r["question"]] = (volet, r["reponse"], bool(test(r["reponse"])))
    return reponses


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ra, rb = lire(a), lire(b)
    print(f"{'':12}{a.stem:>22}{b.stem:>22}")
    for volet in VOLETS:
        na = [ok for v, _, ok in ra.values() if v == volet]
        nb = [ok for v, _, ok in rb.values() if v == volet]
        ia, ib = wilson(sum(na), len(na)), wilson(sum(nb), len(nb))
        print(f"{volet:12}{sum(na):>6}/{len(na):<3} [{ia[0]:.0%}-{ia[1]:.0%}]"
              f"{sum(nb):>6}/{len(nb):<3} [{ib[0]:.0%}-{ib[1]:.0%}]")
    communes = [q for q in ra if q in rb]
    for titre, garde in [("GAGNÉES", lambda x, y: not x and y), ("PERDUES", lambda x, y: x and not y)]:
        liste = [q for q in communes if garde(ra[q][2], rb[q][2])]
        print(f"\n{titre} par {b.stem} ({len(liste)}) :")
        for q in liste:
            print(f"  [{ra[q][0]}] {q}\n      {a.stem}: {texte_note(ra[q][1])[:90]!r}\n      {b.stem}: {texte_note(rb[q][1])[:90]!r}")
    manquantes = set(ra) ^ set(rb)
    if manquantes:
        print(f"\n({len(manquantes)} questions présentes dans un seul des deux fichiers, ignorées)")


if __name__ == "__main__":
    main()
