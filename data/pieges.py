"""
Exemples de lecture avec les passages que notre recherche ramène vraiment
(rag.py), pour apprendre à Carl à se méfier d'un passage qui ressemble à la
question sans contenir la réponse.

    python data/pieges.py   # écrit data/sft/pieges.json

Avec des passages hors sujet tirés au hasard (data/lecture.py), Carl avait
appris à ignorer ce qui ne ressemble pas à la question, mais pas un article
voisin : l'article « Royaume-Uni » sans le mot Londres lui faisait répondre
« Guernesey ». Pour chaque question de PIAF et SQuAD, on lance la recherche :
  - le passage ramené contient la réponse : exemple de lecture ordinaire ;
  - il ne la contient pas : la bonne réponse est « le document ne le dit
    pas », et chat.py repose alors la question sans document.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag  # noqa: E402

R = random.Random(7)
PAS_DANS_LE_TEXTE = ["Le document ne le dit pas.", "Je ne trouve pas cette information dans le document.",
                     "Le passage que j'ai trouvé ne répond pas à cette question."]


def main() -> None:
    lecture = json.load(open("data/sft/lecture.json", encoding="utf-8"))
    # Les questions à réponse de PIAF et SQuAD (pas les impossibles ni les hors sujet).
    questions = [(c[1]["content"], c[2]["content"].rstrip(".")) for c in lecture
                 if c[0]["role"] == "document" and len(c) == 3 and c[2]["content"] not in PAS_DANS_LE_TEXTE
                 and not rag.A_CARL.search(c[1]["content"])]
    R.shuffle(questions)
    positifs, pieges, t0 = [], [], time.time()
    for i, (q, rep) in enumerate(questions[:9000]):
        trouves = rag.chercher(q, k=1)
        if not trouves:
            continue
        doc = trouves[0][0]
        conv = [{"role": "document", "content": doc}, {"role": "user", "content": q}]
        if rep.lower() in doc.lower():
            positifs.append(conv + [{"role": "assistant", "content": rep + "."}])
        else:
            pieges.append(conv + [{"role": "assistant", "content": R.choice(PAS_DANS_LE_TEXTE)}])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} questions, {len(positifs)} passages utiles, {len(pieges)} pièges ({time.time() - t0:.0f} s)", flush=True)
    tout = positifs + pieges
    R.shuffle(tout)
    Path("data/sft/pieges.json").write_text(json.dumps(tout, ensure_ascii=False), encoding="utf-8")
    print(f"{len(positifs)} passages utiles + {len(pieges)} pièges -> data/sft/pieges.json")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
