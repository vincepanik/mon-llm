"""
Télécharge une base de faits depuis Wikidata, une fois pour toutes.

    python data/wikidata.py   # écrit data/big/wikidata/faits.json (quelques minutes)

Wikidata range les connaissances de Wikipédia en tableau : Espagne -> capitale
-> Madrid. On en garde une petite partie (pays, villes, personnes célèbres,
œuvres, éléments chimiques...), en français, par requêtes SPARQL (données
Wikidata sous licence CC0). Ensuite, tout est local : Carl interroge ce
fichier avec l'outil « [fait: Espagne | capitale = » (faits.py).

Les requêtes passent par QLever (université de Fribourg), une copie de
Wikidata bien plus rapide que le service officiel, qui coupe au bout de 60 s
et refuse les grosses requêtes. Les noms communs à toutes les langues
(« Victor Hugo ») sont rangés sous la langue « mul », pas « fr » : on prend
le français s'il existe, sinon « mul ».
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

SORTIE = Path("data/big/wikidata/faits.json")
# Chaque requête réussie est gardée : si le service coupe, on reprend où on en était.
CACHE = SORTIE.parent / "requetes"
URL = "https://qlever.dev/api/wikidata"
AGENT = "mon-llm/0.1 (https://github.com/vincepanik/mon-llm; projet d'apprentissage)"

MOIS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()

# Catégorie : (motif SPARQL qui choisit les entités, combien on en garde, les
# plus connues d'abord (en nombre de Wikipédias), {relation : propriété}).
CATEGORIES = {
    "pays": ("?e wdt:P31 wd:Q3624078 . MINUS { ?e wdt:P576 ?fin }", 400,
             {"capitale": "P36", "monnaie": "P38", "langue": "P37", "continent": "P30", "population": "P1082"}),
    "ville": ("VALUES ?type { wd:Q515 wd:Q1549591 wd:Q1637706 wd:Q200250 wd:Q5119 wd:Q484170 } ?e wdt:P31 ?type .",
              8000, {"pays": "P17", "population": "P1082"}),
    "personne": ("?e wdt:P31 wd:Q5 .", 30000,
                 {"naissance": "P569", "décès": "P570", "lieu de naissance": "P19"}),
    "livre": ("VALUES ?type { wd:Q7725634 wd:Q8261 wd:Q47461344 wd:Q25379 } ?e wdt:P31 ?type .", 5000,
              {"auteur": "P50", "date": "P577"}),
    "tableau": ("?e wdt:P31 wd:Q3305213 .", 1500, {"auteur": "P170", "date": "P571"}),
    "musique": ("VALUES ?type { wd:Q105543609 wd:Q1344 wd:Q207628 } ?e wdt:P31 ?type .", 2000,
                {"compositeur": "P86", "date": "P577"}),
    "film": ("?e wdt:P31 wd:Q11424 .", 6000, {"réalisateur": "P57", "date": "P577"}),
    "élément": ("?e wdt:P31 wd:Q11344 .", 200, {"symbole": "P246", "numéro atomique": "P1086"}),
    "montagne": ("?e wdt:P31 wd:Q8502 .", 1000, {"altitude": "P2044", "pays": "P17"}),
    "événement": ("VALUES ?type { wd:Q198 wd:Q10931 wd:Q178561 wd:Q103495 wd:Q1261499 } ?e wdt:P31 ?type .", 4000,
                  {"début": "P580", "fin": "P582", "date": "P585"}),
    "entreprise": ("VALUES ?type { wd:Q4830453 wd:Q891723 wd:Q6881511 } ?e wdt:P31 ?type .", 2500,
                   {"fondateur": "P112", "création": "P571", "siège": "P159"}),
}
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
DATES = {"P569", "P570", "P577", "P571", "P580", "P582", "P585"}
NOMBRES = {"P1082", "P2044", "P1086"}
TEXTES = {"P246"}


def sparql(requete: str, essais: int = 8) -> list[dict]:
    data = urllib.parse.urlencode({"query": requete}).encode()
    for essai in range(essais):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "User-Agent": AGENT, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:  # délai dépassé, limite de débit : on attend et on recommence
            attente = 30 * (essai + 1)
            if getattr(e, "code", None) == 429:  # « trop de requêtes » : le service dit combien attendre
                attente = max(attente, int(e.headers.get("Retry-After", 60)))
            print(f"    (essai {essai + 1} raté : {e} ; nouvel essai dans {attente} s)", flush=True)
            time.sleep(attente)
    raise RuntimeError("Wikidata ne répond pas")


def date_fr(valeur: str, precision: int) -> str | None:
    """« +1802-02-26T00:00:00Z », précision 11 (jour) -> « 26 février 1802 ». None si plus vague que l'année."""
    if precision < 9:  # décennie, siècle : « 1560 » serait faussement précis
        return None
    signe = -1 if valeur.startswith("-") else 1
    annee, mois, jour = valeur.lstrip("+-").split("T")[0].split("-")
    a = f"{int(annee)} av. J.-C." if signe < 0 else str(int(annee))
    if precision >= 11 and int(jour) > 0:
        return f"{'1er' if int(jour) == 1 else int(jour)} {MOIS[int(mois) - 1]} {a}"
    if precision == 10 and int(mois) > 0:
        return f"{MOIS[int(mois) - 1]} {a}"
    return a


def nombre_fr(valeur: str) -> str:
    """47351567 -> « 47 351 567 » ; 8848.86 -> « 8 849 »."""
    return f"{round(float(valeur)):,}".replace(",", " ")


def nom(variable: str) -> str:
    """Le nom français de ?variable, ou à défaut le nom commun à toutes les langues."""
    return (f"OPTIONAL {{ {variable} rdfs:label {variable}_fr . FILTER(LANG({variable}_fr) = \"fr\") }} "
            f"OPTIONAL {{ {variable} rdfs:label {variable}_mul . FILTER(LANG({variable}_mul) = \"mul\") }}")


def requete(motif: str, combien: int, prop: str) -> str:
    if prop in DATES:
        valeur = (f"?e p:{prop} ?st . ?st a wikibase:BestRank . ?st psv:{prop} ?v . "
                  "?v wikibase:timeValue ?val . ?v wikibase:timePrecision ?prec .")
    elif prop in NOMBRES or prop in TEXTES:
        valeur = f"?e wdt:{prop} ?val ."
    else:
        valeur = f"?e wdt:{prop} ?obj . {nom('?obj')}"
    return f"""{PREFIXES}SELECT ?e ?e_fr ?e_mul ?liens ?genre ?val ?obj_fr ?obj_mul ?prec WHERE {{
  {{ SELECT ?e ?liens WHERE {{ {motif} ?e wikibase:sitelinks ?liens . }} ORDER BY DESC(?liens) LIMIT {combien} }}
  {nom('?e')}
  {valeur}
  OPTIONAL {{ ?e wdt:P21 ?genre }}
}}"""


def texte(ligne: dict, variable: str) -> str | None:
    for langue in ("fr", "mul"):
        if f"{variable}_{langue}" in ligne:
            return ligne[f"{variable}_{langue}"]["value"]
    return None


def main() -> None:
    entites: dict[str, dict] = {}
    faits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for categorie, (motif, combien, relations) in CATEGORIES.items():
        for relation, prop in relations.items():
            debut = time.time()
            cache = CACHE / f"{categorie}_{relation}.json".replace(" ", "_")
            if cache.exists():
                lignes = json.loads(cache.read_text(encoding="utf-8"))
            else:
                lignes = sparql(requete(motif, combien, prop))
                CACHE.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(lignes, ensure_ascii=False), encoding="utf-8")
                time.sleep(2)  # rester poli avec le service public
            for l in lignes:
                qid = l["e"]["value"].rsplit("/", 1)[1]
                nom_e = texte(l, "e")
                val = l["val"]["value"] if "val" in l else texte(l, "obj")
                if not nom_e or not val:
                    continue
                if prop in DATES:
                    val = date_fr(val, int(l["prec"]["value"]))
                elif prop in NOMBRES:
                    val = nombre_fr(val)
                if not val or val.startswith("http"):  # valeur inconnue (« somevalue ») ou sans nom français
                    continue
                entites.setdefault(qid, {"nom": nom_e, "type": categorie,
                                         "liens": int(l["liens"]["value"])})
                if l.get("genre", {}).get("value", "").endswith("/Q6581072"):
                    entites[qid]["genre"] = "f"
                if val not in faits[qid][relation]:
                    faits[qid][relation].append(val)
            print(f"  {categorie:<10} {relation:<17} {len(lignes):>6} lignes  ({time.time() - debut:.0f} s)", flush=True)
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps({"entites": entites, "faits": faits}, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for f in faits.values() for v in f.values())
    print(f"{len(entites)} entités, {n} faits -> {SORTIE} ({SORTIE.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    sys.exit(main())
