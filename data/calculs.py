"""
Conversations de calcul pour apprendre à Carl à se servir de la calculatrice
(outils.py), fabriquées par programme : le résultat est toujours juste.

    python data/calculs.py   # écrit data/sft/calculs.json

Carl écrit l'opération, puis « [calc: expression = résultat] » ; le résultat
est calculé par le programme au moment de la conversation, jamais par Carl.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outils import calculer  # noqa: E402

R = random.Random(2026)


def nombre() -> int | float:
    tirage = R.random()
    if tirage < 0.35:
        return R.randint(1, 20)
    if tirage < 0.7:
        return R.randint(10, 999)
    if tirage < 0.9:
        return R.randint(100, 99999)
    return round(R.uniform(0.5, 500), R.choice([1, 2]))


def fr(x) -> str:
    return str(x).replace(".", ",")


def outil(expr: str) -> str:
    return f"[calc: {expr} = {calculer(expr)}]"


OPS = [  # symbole de l'expression, façons de le dire, signe affiché
    ("+", ["plus", "+", "et"], "+"),
    ("-", ["moins", "-"], "-"),
    ("*", ["fois", "x", "×", "*", "multiplié par"], "×"),
    ("/", ["divisé par", "/", "sur"], "÷"),
]
QUESTIONS = [
    "Combien font {a} {mot} {b} ?", "combien fait {a} {mot} {b} ?", "{a} {mot} {b} ?", "{a}{mot}{b}",
    "Calcule {a} {mot} {b}.", "Quel est le résultat de {a} {mot} {b} ?", "Peux-tu calculer {a} {mot} {b} ?",
    "Ça fait combien {a} {mot} {b} ?", "{a} {mot} {b} = ?", "Dis-moi combien font {a} {mot} {b}.",
]
PROBLEMES = [
    ("J'ai {a} euros et je dépense {b} euros. Combien me reste-t-il ?", "{a}-{b}",
     "Il te reste {a} - {b} = {r} euros."),
    ("Un paquet contient {a} biscuits. Combien y a-t-il de biscuits dans {b} paquets ?", "{a}*{b}",
     "Il y a {a} × {b} = {r} biscuits."),
    ("{b} personnes se partagent {a} euros à parts égales. Combien chacune reçoit-elle ?", "{a}/{b}",
     "Chacune reçoit {a} ÷ {b} = {r} euros."),
    ("Un livre coûte {a} euros. Combien coûtent {b} livres ?", "{a}*{b}",
     "{b} livres coûtent {a} × {b} = {r} euros."),
    ("J'avais {a} billes et j'en gagne {b}. Combien en ai-je maintenant ?", "{a}+{b}",
     "Tu as maintenant {a} + {b} = {r} billes."),
]


def conversations(n: int = 1500) -> list[list[dict]]:
    convs = []
    for _ in range(n):
        tirage = R.random()
        if tirage < 0.6:
            symbole, mots, signe = R.choice(OPS)
            a, b = nombre(), nombre()
            if symbole == "-" and R.random() < 0.7:
                a, b = max(a, b), min(a, b)
            if symbole == "/" and R.random() < 0.6 and isinstance(a, int) and isinstance(b, int) and b <= 100:
                a = a * b  # divisions qui tombent juste, souvent
            q = R.choice(QUESTIONS).format(a=fr(a), mot=R.choice(mots), b=fr(b))
            rep = f"{fr(a)} {signe} {fr(b)} = {outil(f'{a}{symbole}{b}')}."
        elif tirage < 0.72:
            p, n_ = R.choice([5, 10, 15, 20, 25, 30, 50, 75, R.randint(1, 99)]), R.randint(10, 5000)
            q = R.choice(["Combien font {p} % de {n} ?", "Calcule {p} % de {n}.", "C'est combien {p}% de {n} ?"]).format(p=p, n=n_)
            rep = f"{p} % de {n_} = {outil(f'{n_}*{p}/100')}."
        elif tirage < 0.8:
            a = R.randint(2, 99)
            q = R.choice(["Combien fait {a} au carré ?", "{a} au carré ?", "Calcule le carré de {a}."]).format(a=a)
            rep = f"{a} au carré, c'est {a} × {a} = {outil(f'{a}*{a}')}."
        else:
            enonce, expr, modele = R.choice(PROBLEMES)
            a, b = R.randint(2, 500), R.randint(2, 30)
            if "-" in expr:
                a, b = max(a, b), min(a, b)
            q = enonce.format(a=a, b=b)
            rep = modele.format(a=a, b=b, r=outil(expr.format(a=a, b=b)))
        convs.append([{"role": "user", "content": q}, {"role": "assistant", "content": rep}])
    # Quelques questions sur la calculatrice elle-même.
    for q in ["Sais-tu calculer ?", "Tu sais faire des maths ?", "Tu peux faire des calculs ?", "Tu es bon en calcul ?"]:
        convs.append([{"role": "user", "content": q}, {"role": "assistant", "content":
            "Oui : je me sers d'une calculatrice pour les opérations, pour ne pas me tromper. "
            "Pose-moi un calcul, par exemple « combien font 17 × 23 ? »."}])
    return convs


if __name__ == "__main__":
    c = conversations()
    sortie = Path("data/sft/calculs.json")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    print(f"{len(c)} conversations -> {sortie}")
    for conv in R.sample(c, 5):
        print(f"  {conv[0]['content']!r} -> {conv[1]['content']!r}")
