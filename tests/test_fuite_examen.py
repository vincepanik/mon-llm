"""
Aucune question d'examen ne doit figurer mot pour mot dans les données
d'entraînement, et aucun fait des tests de la base de faits ne doit y avoir
été appris avec l'outil. Sans ce test, « Combien fait 37 au carré ? » était
dans data/sft/calculs.json, et « Qui a peint Guernica ? » dans les
conversations de l'étape D, alors que les commentaires affirmaient le
contraire.

Les données (data/sft/, gitignoré) ne sont que sur la machine qui entraîne :
ailleurs, le test est sauté.
"""

import glob
import json
import re
import unicodedata
from pathlib import Path

import pytest

import faits
from examen import TESTS
from examen_faits import jeux

SFT = Path("data/sft")
# Fichiers qui ne servent jamais à entraîner Carl (le test de l'expérience).
PAS_ENTRAINEMENT = {"exp_test.json"}
# Tolérées : un salut ne se reformule pas.
AUTORISEES = {"bonjour"}

pytestmark = pytest.mark.skipif(not SFT.exists(), reason="données d'entraînement absentes")


def norm(texte: str) -> str:
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", texte).strip()


def conversations():
    fichiers = [f for f in glob.glob(str(SFT / "*.json")) if Path(f).name not in PAS_ENTRAINEMENT]
    for f in fichiers + ["data/identite_carl.json"]:
        for conv in json.loads(Path(f).read_text(encoding="utf-8")):
            if isinstance(conv, list):
                yield f, conv


def test_aucune_question_d_examen_dans_l_entrainement():
    from examen_style import CLOTURES, OUVERTES, OUVERTURES

    questions = ({norm(q): q for q in TESTS} | {norm(q): q for _, jeu in jeux() for q, _, _ in jeu}
                 | {norm(q): q for q in OUVERTURES + CLOTURES + OUVERTES})
    fuites = set()
    for f, conv in conversations():
        for m in conv:
            if m.get("role") == "user" and (n := norm(m["content"])) in questions and n not in AUTORISEES:
                fuites.add(f"{questions[n]!r} dans {Path(f).name}")
    assert not fuites, "\n".join(sorted(fuites))


def test_faits_des_tests_jamais_appris_avec_l_outil():
    appris = set()
    for _, conv in conversations():
        for m in conv:
            if m.get("role") == "assistant":
                for entite, relation in re.findall(r"\[fait:\s*([^|\]]+?)\s*\|\s*([^=\]]+?)\s*=", m["content"]):
                    appris.add((faits.normaliser(entite), relation.strip()))
    fuites = [(q, e, r) for _, jeu in jeux(tous=False) for q, e, r in jeu if (faits.normaliser(e), r) in appris]
    assert not fuites, fuites


def test_calculs_d_examen_jamais_faits_a_l_entrainement():
    # « Combien fait 43 au carré ? » n'était pas mot pour mot dans calculs.json,
    # mais « Calcule le carré de 43. » -> [calc: 43*43 = 1849] si : même calcul.
    from examen import CALCULS
    from examen_conversation import CONVERSATIONS

    faits_ = set()
    for _, conv in conversations():
        for m in conv:
            for expr, res in re.findall(r"\[calc:\s*([^=\]]+?)\s*=\s*([^\]]*)\]", m["content"]):
                faits_.add((frozenset(re.findall(r"\d+", expr)), res.strip()))
    examen = list(CALCULS) + [(q, a[0]) for conv in CONVERSATIONS for q, a in conv
                              if a and re.search(r"\d+ (fois|plus|moins)", q)]
    fuites = [(q, r) for q, r in examen if (frozenset(re.findall(r"\d+", q)), r) in faits_]
    assert not fuites, fuites
