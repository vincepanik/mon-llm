"""
Garde, dans des conversations venues d'ailleurs (compar:IA, oasst), celles
dont le style convient à Carl.

    python data/style.py data/sft/comparia_2000_outils.json data/sft/conversations_outils.json
    # écrit data/sft/comparia_2000_outils_style.json, etc.

compar:IA, ce sont des réponses de gros modèles : 1 084 caractères en moyenne,
60 % en listes et en gras, 45 % au vouvoiement. Carl imitait la forme sans
avoir le fond (« Soupe de légumes aux légumes », « 8. » coupé en fin de
liste), et passait du « tu » au « vous » d'une réponse à l'autre. On garde les
réponses courtes (450 caractères au plus), sans mise en forme, au tutoiement.

On écarte aussi une question que la base de faits sait traiter (« Quelle est
la capitale de la Mongolie ? ») quand la réponse ne l'appelle pas : elle
apprendrait à Carl à répondre de mémoire, ce qui lui faisait perdre le
réflexe d'outil (voir l'expérience des faits dans le README).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outiller import MOTIFS  # noqa: E402

MISE_EN_FORME = re.compile(r"\*\*|^\s*(\d+[.)]|[-*•])\s|^#", re.M)
VOUVOIEMENT = re.compile(r"\b(vous|votre|vos)\b", re.I)


def bon_style(reponse: str) -> bool:
    return len(reponse) <= 450 and not MISE_EN_FORME.search(reponse) and not VOUVOIEMENT.search(reponse)


def sans_outil(conv: list[dict]) -> bool:
    """Une question pour la base de faits, à laquelle la réponse répond de mémoire ?"""
    for i, m in enumerate(conv[:-1]):
        if m["role"] == "user" and any(motif.search(m["content"].strip()) for motif, _ in MOTIFS):
            if "[fait:" not in conv[i + 1]["content"]:
                return True
    return False


def filtrer(convs: list[list[dict]]) -> list[list[dict]]:
    return [c for c in convs
            if all(bon_style(m["content"]) for m in c if m["role"] == "assistant") and not sans_outil(c)]


if __name__ == "__main__":
    for chemin in map(Path, sys.argv[1:]):
        convs = json.loads(chemin.read_text(encoding="utf-8"))
        gardees = filtrer(convs)
        sortie = chemin.with_name(chemin.stem + "_style.json")
        sortie.write_text(json.dumps(gardees, ensure_ascii=False), encoding="utf-8")
        print(f"{chemin} : {len(gardees)}/{len(convs)} gardées -> {sortie}")
