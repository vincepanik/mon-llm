"""
Exemples de lecture pour apprendre à Carl à répondre à partir d'un passage de
Wikipédia (voir rag.py), au lieu de sa mémoire.

    python data/lecture.py   # écrit data/sft/lecture.json

Chaque conversation commence par un message « document » (le passage trouvé),
puis la question, puis la réponse. Trois sortes :

  1. le passage contient la réponse : PIAF (Etalab, MIT, questions humaines
     sur Wikipédia FR) et SQuAD v2 traduit (Apache 2.0). Carl doit la trouver ;
  2. il ne la contient pas (questions « impossibles » de SQuAD v2) : Carl doit
     le dire au lieu d'inventer ;
  3. le passage est hors sujet (article tiré au hasard) devant un salut, une
     question d'identité, un calcul ou une conversation ordinaire : Carl doit
     répondre normalement. La recherche ramène toujours quelque chose, même
     quand la question n'a rien de factuel.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import SOURCE, passage  # noqa: E402

R = random.Random(42)
MAX_DOC = 1400  # caractères : le document, la question et la réponse doivent tenir en 512 tokens
PAS_DANS_LE_TEXTE = [
    "Le document ne le dit pas.",
    "Je ne trouve pas cette information dans le document.",
    "Le passage que j'ai trouvé ne répond pas à cette question.",
]


def fenetre(contexte: str, debut: int, longueur: int) -> str:
    """Un extrait d'au plus MAX_DOC caractères qui contient la réponse."""
    if len(contexte) <= MAX_DOC:
        return contexte
    gauche = max(0, min(debut - MAX_DOC // 3, len(contexte) - MAX_DOC))
    return contexte[gauche : gauche + MAX_DOC]


def phrase(reponse: str) -> str:
    reponse = reponse.strip()
    reponse = reponse[0].upper() + reponse[1:]
    return reponse if reponse[-1] in ".!?" else reponse + "."


def conv(document: str, question: str, reponse: str) -> list[dict]:
    return [{"role": "document", "content": document.strip()},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": reponse}]


def piaf() -> list[list[dict]]:
    sortie = []
    for ex in load_dataset("AgentPublic/piaf", split="train"):
        rep = ex["answers"]["text"][0]
        doc = fenetre(ex["context"], ex["answers"]["answer_start"][0], len(rep))
        if rep in doc:
            sortie.append(conv(f"{ex['title']}\n{doc}", ex["question"], phrase(rep)))
    return sortie


def squad(n_possibles: int, n_impossibles: int) -> list[list[dict]]:
    chemin = hf_hub_download("pragnakalp/squad_v2_french_translated", "squad_v2_french_translated.json",
                             repo_type="dataset")
    donnees = json.load(open(chemin, encoding="utf-8"))["data"]
    possibles, impossibles = [], []
    for article in donnees:
        titre = article["title"].replace("_", " ")
        for par in article["paragraphs"]:
            contexte = par["context"]
            for qa in par["qas"]:
                if qa.get("is_impossible"):
                    if len(contexte) <= MAX_DOC:
                        impossibles.append(conv(f"{titre}\n{contexte}", qa["question"], R.choice(PAS_DANS_LE_TEXTE)))
                elif qa["answers"]:
                    rep = qa["answers"][0]["text"]
                    debut = contexte.find(rep)
                    # La traduction décale parfois la réponse : on ne garde que
                    # les cas où elle figure vraiment dans le passage traduit.
                    if rep and debut >= 0:
                        possibles.append(conv(f"{titre}\n{fenetre(contexte, debut, len(rep))}", qa["question"], phrase(rep)))
    R.shuffle(possibles)
    R.shuffle(impossibles)
    return possibles[:n_possibles] + impossibles[:n_impossibles]


def passages_au_hasard(n: int) -> list[str]:
    """n articles tirés au hasard dans le fichier de Wikipédia, sans tout lire."""
    taille = SOURCE.stat().st_size
    sortie = []
    with SOURCE.open("rb") as f:
        while len(sortie) < n:
            f.seek(R.randrange(taille))
            bloc = f.read(12000).decode("utf-8", errors="ignore")
            morceaux = bloc.split("\n<|endoftext|>\n")
            if len(morceaux) >= 3 and (p := passage(morceaux[1])):
                sortie.append(p)
    return sortie


def hors_sujet() -> list[list[dict]]:
    """Un passage sans rapport devant des conversations qui n'ont rien de factuel."""
    ident = json.load(open("data/identite_carl.json", encoding="utf-8"))
    calculs = R.sample(json.load(open("data/sft/calculs.json", encoding="utf-8")), 600)
    generales = R.sample(json.load(open("data/sft/comparia_reparation.json", encoding="utf-8")), 1500)
    bases = [c for c in ident + calculs + generales if len(c) == 2]
    docs = passages_au_hasard(len(bases))
    return [[{"role": "document", "content": d}] + c for d, c in zip(docs, bases)]


def main() -> None:
    a = piaf()
    print(f"PIAF : {len(a):,}", flush=True)
    b = squad(n_possibles=9000, n_impossibles=3000)
    print(f"SQuAD v2 : {len(b):,} (dont 3 000 sans réponse dans le passage)", flush=True)
    c = hors_sujet()
    print(f"passages hors sujet : {len(c):,}", flush=True)
    tout = a + b + c
    R.shuffle(tout)
    Path("data/sft/lecture.json").write_text(json.dumps(tout, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tout):,} conversations -> data/sft/lecture.json")
    for ex in R.sample(tout, 3):
        print(f"  [doc] {ex[0]['content'][:70]!r}\n  [q]   {ex[1]['content']!r}\n  [r]   {ex[2]['content'][:90]!r}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
