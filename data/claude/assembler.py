"""
Conversations écrites par Claude (Anthropic) pour le SFT de Carl (étape D).

Les lots sont des fichiers texte, lot_*.txt, dans un format compact :

    Q: la question
    R: la réponse (plusieurs lignes possibles)
    Q: une relance
    R: sa réponse
    ===

    python data/claude/assembler.py   # écrit data/sft/claude.json

Réserve : les conditions d'utilisation d'Anthropic interdisent d'utiliser les
réponses de Claude pour entraîner un modèle qui ferait concurrence à ses
services. Carl est un projet personnel d'apprentissage ; pour tout usage
commercial, ne pas garder ces données (préférer des réponses de Gemma, Apache 2.0).

Vérifications : alternance question/réponse, et aucune question de examen.py
ni des relances de mesure (sinon l'examen mesurerait de la récitation).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DOSSIER = Path(__file__).resolve().parent


def normaliser(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower().replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a"))


def lire(chemin: Path) -> list[list[dict]]:
    convs, courante, role, lignes = [], [], None, []

    def fermer_message():
        nonlocal role, lignes
        if role:
            courante.append({"role": role, "content": "\n".join(lignes).strip()})
        role, lignes = None, []

    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if ligne.strip() == "===":
            fermer_message()
            if courante:
                convs.append(courante)
            courante = []
        elif ligne.startswith("Q: ") or ligne.startswith("R: "):
            fermer_message()
            role = "user" if ligne.startswith("Q: ") else "assistant"
            lignes = [ligne[3:]]
        elif role:
            lignes.append(ligne)
    fermer_message()
    if courante:
        convs.append(courante)
    return convs


def main() -> None:
    import examen

    interdites = {normaliser(q) for q in examen.QUI + examen.CREA + [q for q, _, _ in examen.FAITS]
                  + [q for q, _ in examen.CONDUITE] + [q for q, _ in examen.CALCULS]}
    tout, problemes = [], 0
    for lot in sorted(DOSSIER.glob("lot_*.txt")):
        convs = lire(lot)
        for c in convs:
            roles = [m["role"] for m in c]
            if roles != ["user", "assistant"] * (len(c) // 2) or len(c) % 2 or any(not m["content"] for m in c):
                print(f"  {lot.name} : conversation mal formée : {c[0]['content'][:60]!r}")
                problemes += 1
                continue
            if any(normaliser(m["content"]) in interdites for m in c if m["role"] == "user"):
                print(f"  {lot.name} : question d'examen écartée : {c[0]['content'][:60]!r}")
                problemes += 1
                continue
            tout.append(c)
        print(f"{lot.name} : {len(convs)} conversations")
    tours = sum(len(c) // 2 for c in tout)
    Path("data/sft/claude.json").write_text(json.dumps(tout, ensure_ascii=False), encoding="utf-8")
    print(f"total : {len(tout)} conversations, {tours} échanges, {problemes} écartées -> data/sft/claude.json")


if __name__ == "__main__":
    main()
