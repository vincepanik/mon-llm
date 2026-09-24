"""
Base de faits de Carl (Wikidata, téléchargée par data/wikidata.py).

Comme pour la calculatrice (outils.py), Carl n'a pas à savoir : il apprend
(data/faits_sft.py) à écrire « [fait: Espagne | capitale = », le programme
cherche ici, insère « Madrid] », et Carl recopie le résultat dans sa phrase.
À l'affichage, l'appel disparaît : il ne reste que « La capitale de
l'Espagne est Madrid. ».

La recherche tolère ce qu'un petit modèle écrit à peu près : majuscules,
accents, article en trop (« l'Espagne »), nom incomplet (« Hugo » pour
« Victor Hugo ») ou légère faute de frappe.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

CHEMIN = Path("data/big/wikidata/faits.json")

# Ce que Carl peut écrire après « | », ramené aux relations de la base.
RELATIONS = {
    "capitale": ["capitale"], "monnaie": ["monnaie"], "devise": ["monnaie"],
    "langue": ["langue"], "langues": ["langue"], "continent": ["continent"],
    "population": ["population"], "habitants": ["population"], "pays": ["pays"],
    "naissance": ["naissance"], "ne": ["naissance"], "date de naissance": ["naissance"],
    "deces": ["décès"], "mort": ["décès"], "date de deces": ["décès"],
    "lieu de naissance": ["lieu de naissance"],
    "auteur": ["auteur", "compositeur", "réalisateur"], "ecrivain": ["auteur"], "peintre": ["auteur"],
    "compositeur": ["compositeur"], "realisateur": ["réalisateur"],
    "date": ["date", "début", "création"], "annee": ["date", "début", "création"],
    "debut": ["début", "date"], "fin": ["fin", "date"],
    "symbole": ["symbole"], "numero atomique": ["numéro atomique"],
    "altitude": ["altitude"], "hauteur": ["altitude"],
    "fondateur": ["fondateur"], "creation": ["création"], "siege": ["siège"],
}
LIEUX = {"lieu de naissance", "siège", "pays"}
DATES = {"naissance", "décès", "date", "début", "fin", "création"}
ARTICLES = re.compile(r"^(?:de la |de l |du |des |de |d |la |le |les |l )+")


def normaliser(texte: str) -> str:
    """« de l'Espagne » -> « espagne » ; « Saint-Exupéry » -> « saint exupery »."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^a-z0-9]+", " ", texte).strip()
    return ARTICLES.sub("", texte).strip()


@lru_cache(maxsize=1)
def _base() -> tuple[dict, dict, dict[str, list[str]]]:
    donnees = json.loads(CHEMIN.read_text(encoding="utf-8"))
    entites, faits = donnees["entites"], donnees["faits"]
    index: dict[str, list[str]] = {}
    for qid, e in entites.items():
        index.setdefault(normaliser(e["nom"]), []).append(qid)
    for qids in index.values():  # le plus connu d'abord : « Paris » la ville, pas un homonyme
        qids.sort(key=lambda q: -entites[q]["liens"])
    return entites, faits, index


def _annee(date: str) -> int:
    """« 14 juin 1991 » -> 1991 ; « 1389 av. J.-C. » -> -1389."""
    nombres = re.findall(r"\d+", date)
    annee = int(nombres[-1]) if nombres else 0
    return -annee if "av. J.-C." in date else annee


def disponible() -> bool:
    return CHEMIN.exists()


def _candidats(nom: str, relations: list[str]) -> list[str]:
    entites, faits, index = _base()
    a_la_relation = lambda q: any(r in faits.get(q, {}) for r in relations)  # noqa: E731
    exacts = [q for q in index.get(nom, []) if a_la_relation(q)]
    if exacts:
        return exacts
    # Nom incomplet : « hugo » dans « victor hugo », « napoleon » dans « napoleon ier ».
    motif = re.compile(rf"(?:^| ){re.escape(nom)}(?: |$)")
    partiels = [q for n, qs in index.items() if motif.search(n) for q in qs if a_la_relation(q)]
    if partiels:
        return sorted(partiels, key=lambda q: -entites[q]["liens"])
    # Mot en trop : « mont everest » pour « everest », « ville de lyon » pour « lyon ».
    mots = nom.split()
    for i in range(1, len(mots)):
        fin = [q for q in index.get(" ".join(mots[i:]), []) if a_la_relation(q)]
        if fin:
            return fin
    # Faute de frappe : « espangne ».
    proches = difflib.get_close_matches(nom, list(index), n=3, cutoff=0.85)
    return [q for n in proches for q in index[n] if a_la_relation(q)]


def chercher(entite: str, relation: str) -> str | None:
    """« Espagne », « capitale » -> « Madrid » ; None si la base ne sait pas."""
    if not disponible():
        return None
    nom, cle = normaliser(entite), normaliser(relation)
    relations = RELATIONS.get(cle, [relation.strip().lower()])
    if not nom:
        return None
    entites, faits, index = _base()
    # Dans l'ordre : « auteur » cherche d'abord un écrivain, et seulement
    # ensuite un réalisateur (sinon « Hamlet » donnait le film de 1948).
    for r in relations:
        for qid in _candidats(nom, [r])[:1]:
            valeurs = faits[qid][r]
            if valeurs:
                if r in LIEUX and len(valeurs) > 1:  # « Varsovie », pas la rue « Ulica Freta »
                    connus = [v for v in valeurs if normaliser(v) in index]
                    valeurs = sorted(connus, key=lambda v: -entites[index[normaliser(v)][0]]["liens"])[:1] or valeurs[:1]
                if r in DATES:  # plusieurs dates de sortie : la première
                    valeurs = [min(valeurs, key=_annee)]
                valeurs = valeurs[:3]
                return valeurs[0] if len(valeurs) == 1 else ", ".join(valeurs[:-1]) + " et " + valeurs[-1]
    return None
