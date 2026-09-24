"""
Ajoute l'appel à la base de faits dans les anciennes conversations.

    python data/outiller.py data/sft/claude.json data/sft/comparia_2000.json data/sft/conversations.json
    # écrit data/sft/claude_outils.json, etc.

Carl v9 n'appelait jamais l'outil pour les capitales : les conversations de
l'étape D en contenaient des dizaines, avec la réponse donnée de mémoire
(« La capitale de la Suède est Stockholm. »), et cette leçon l'emportait sur
celle de data/faits_sft.py. Ici, la même réponse commence par l'appel :

    [fait: Suède | capitale = Stockholm] La capitale de la Suède est Stockholm.

Seulement quand la base donne la réponse que contenait déjà le texte : on
ne change pas un mot de ce qui était écrit. Jamais pour une entité de
l'examen (examen.py) : certaines figuraient déjà dans ces données.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faits  # noqa: E402
from faits_sft import cite_dans_examen, interdits_examen  # noqa: E402

EXAMEN = interdits_examen()

DE = r"(?:de la |de l'|de l’|du |des |de |d'|d’)"
MOTIFS = [
    (re.compile(rf"capitale {DE}(.+?)\s*\?", re.I), "capitale"),
    (re.compile(rf"monnaie {DE}(.+?)\s*\?", re.I), "monnaie"),
    (re.compile(r"^qui a (?:écrit|peint) (?:le livre |le roman |le tableau )?(.+?)\s*\?", re.I), "auteur"),
    (re.compile(r"^qui a composé (.+?)\s*\?", re.I), "compositeur"),
    (re.compile(r"^qui a réalisé (?:le film )?(.+?)\s*\?", re.I), "réalisateur"),
    (re.compile(r"^(?:quand|en quelle année) est née? (.+?)\s*\?", re.I), "naissance"),
    (re.compile(r"^(?:quand|en quelle année) est morte? (.+?)\s*\?", re.I), "décès"),
    (re.compile(r"^qui a fondé (.+?)\s*\?", re.I), "fondateur"),
]


def appel(question: str, reponse: str) -> str | None:
    for motif, relation in MOTIFS:
        m = motif.search(question.strip())
        if not m:
            continue
        entite = m.group(1)
        if cite_dans_examen(entite, EXAMEN):
            return None
        valeur = faits.chercher(entite, relation)
        if not valeur:
            return None
        # La base doit confirmer la réponse d'origine (un mot distinctif au moins).
        mots = [w for w in re.findall(r"\w+", valeur.split(" et ")[0]) if len(w) >= 3 or w.isdigit()]
        if mots and all(w.lower() in reponse.lower() for w in mots):
            return f"[fait: {entite} | {relation} = {valeur}]"
        return None
    return None


def outiller(convs: list[list[dict]]) -> tuple[list[list[dict]], int]:
    n = 0
    sortie = []
    for conv in convs:
        nouvelle = []
        for i, m in enumerate(conv):
            if m["role"] == "assistant" and i > 0 and conv[i - 1]["role"] == "user" and "[fait:" not in m["content"]:
                a = appel(conv[i - 1]["content"], m["content"])
                if a:
                    m = {**m, "content": f"{a} {m['content']}"}
                    n += 1
            nouvelle.append(m)
        sortie.append(nouvelle)
    return sortie, n


if __name__ == "__main__":
    for chemin in map(Path, sys.argv[1:]):
        convs, n = outiller(json.loads(chemin.read_text(encoding="utf-8")))
        sortie = chemin.with_name(chemin.stem + "_outils.json")
        sortie.write_text(json.dumps(convs, ensure_ascii=False), encoding="utf-8")
        print(f"{chemin} : {n} réponses avec appel -> {sortie}")
        exemples = [m["content"] for c in convs for m in c if m["role"] == "assistant" and m["content"].startswith("[fait:")]
        for e in exemples[:4]:
            print(f"    {e[:110]}")
