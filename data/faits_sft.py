"""
Conversations pour apprendre à Carl à consulter sa base de faits (faits.py),
fabriquées par programme à partir de Wikidata : la réponse est toujours juste.

    python data/faits_sft.py   # écrit data/sft/faits.json

Carl écrit d'abord l'appel, « [fait: Espagne | capitale = », le programme
complète « Madrid] », puis Carl recopie le résultat dans une phrase :

    [fait: Espagne | capitale = Madrid] La capitale de l'Espagne est Madrid.

Aucune entité citée dans une question de l'examen (examen.py), de
examen_faits.py ou de son jeu de test (examen_faits_test.py) : on mesure s'il
a appris le réflexe, pas les réponses.

Une question sur trois est « tapée vite » : fautes de frappe, sans accents ni
majuscules, style télégraphique (« capital du portigal ? »). Carl v10, qui
n'avait vu que des questions bien écrites, n'appelait pas sa base pour
celles-là et inventait (« le Portigal est un véhicule électrique »).
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faits  # noqa: E402
from examen import FAITS  # noqa: E402
from examen_faits import QUESTIONS as JAMAIS_VUES, QUESTIONS_VITE  # noqa: E402
from examen_faits_test import QUESTIONS_TEST  # noqa: E402

R = random.Random(2026)
MOIS = set("janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split())
VOYELLES = tuple("aeiouyhéèêâîôûÉÈÊÂÎÔÛAEIOUYH")

# --- Un peu de grammaire : « l'Espagne », « du Japon », « aux États-Unis ». ---

PAYS_PLURIELS = {"États-Unis", "Pays-Bas", "Philippines", "Émirats arabes unis", "Comores", "Seychelles",
                 "Maldives", "Bahamas", "Fidji", "Îles Marshall", "Îles Salomon"}
PAYS_SANS_ARTICLE = {"Cuba", "Singapour", "Malte", "Chypre", "Madagascar", "Monaco", "Israël", "Haïti", "Oman",
                     "Bahreïn", "Djibouti", "Taïwan", "Nauru", "Tuvalu", "Kiribati", "Vanuatu", "Samoa", "Tonga",
                     "Sao Tomé-et-Principe", "Saint-Marin", "Saint-Christophe-et-Niévès", "Sainte-Lucie",
                     "Saint-Vincent-et-les-Grenadines", "Maurice", "Trinité-et-Tobago", "Antigua-et-Barbuda",
                     "Brunei", "Palaos", "Grenade", "Qatar", "Koweït", "Timor oriental"}
PAYS_MASCULINS_EN_E = {"Mexique", "Cambodge", "Mozambique", "Zimbabwe", "Belize", "Suriname"}
MASCULINS_EVENEMENTS = ("siège", "traité", "conflit", "génocide", "massacre", "accident", "incendie", "coup",
                        "soulèvement", "printemps", "débarquement", "combat", "blocus", "krach", "attentat")


def article_pays(nom: str) -> str:
    """« la », « le », « l' », « les » ou rien."""
    if nom in PAYS_SANS_ARTICLE:
        return ""
    if nom in PAYS_PLURIELS:
        return "les"
    if nom.startswith(VOYELLES):
        return "l'"
    return "la" if nom.endswith("e") and nom not in PAYS_MASCULINS_EN_E else "le"


def article_evenement(nom: str) -> str:
    premier = nom.split()[0].lower()
    if premier in {"premier", "second"} or premier.startswith(MASCULINS_EVENEMENTS):
        return "l'" if nom.startswith(VOYELLES) else "le"
    return "l'" if nom.startswith(VOYELLES) else "la"


def avec(article: str, nom: str) -> str:
    """« l' » + « Espagne » -> « l'Espagne »."""
    return f"{article}{nom}" if article.endswith("'") else f"{article} {nom}".strip()


def de(article: str, nom: str) -> str:
    """« de l'Espagne », « du Japon », « des États-Unis », « d'Israël », « de Cuba »."""
    if article == "le":
        return f"du {nom}"
    if article == "les":
        return f"des {nom}"
    if article:
        return f"de {avec(article, nom)}"
    return f"d'{nom}" if nom.startswith(VOYELLES) else f"de {nom}"


def en(article: str, nom: str) -> str:
    """« en Espagne », « au Japon », « aux États-Unis », « à Cuba »."""
    if not article:
        return f"à {nom}"
    if article == "les":
        return f"aux {nom}"
    if article == "le":
        return f"au {nom}"
    return f"en {nom}"


def de_nom(nom: str) -> str:
    """« d'Adele », « du Pecq », « de Victor Hugo » : pas « de Adele » ni « de Le Pecq »."""
    if nom.startswith("Le "):
        return "du " + nom[3:]
    if nom.startswith("Les "):
        return "des " + nom[4:]
    return f"d'{nom}" if nom.startswith(VOYELLES) else f"de {nom}"


def a_nom(nom: str) -> str:
    """« au Pecq », « aux Lilas », « à Rome »."""
    if nom.startswith("Le "):
        return "au " + nom[3:]
    if nom.startswith("Les "):
        return "aux " + nom[4:]
    return f"à {nom}"


def maj(texte: str) -> str:
    return texte[:1].upper() + texte[1:]


def quand(date: str) -> str:
    """« 26 février 1802 » -> « le 26 février 1802 » ; « 1802 » -> « en 1802 »."""
    mots = date.split()
    return f"le {date}" if len(mots) >= 3 and mots[1] in MOIS else f"en {date}"


def element(nom: str) -> str:
    return f"de l'{nom}" if nom.startswith(VOYELLES) else f"du {nom}"


def monnaie(nom: str) -> str:
    mot = nom.split()[0].lower()
    if nom.startswith(VOYELLES):
        return f"l'{nom}"
    return f"la {nom}" if mot.endswith("e") and mot not in {"rouble"} else f"le {nom}"


def langue(nom: str) -> str:
    return f"l'{nom}" if nom.startswith(VOYELLES) else f"le {nom}"


# --- Questions et réponses, par catégorie et relation. ---
# Chaque gabarit reçoit un dictionnaire g (formes grammaticales) et la valeur v.

def formes(nom: str, entite: dict) -> dict:
    t = entite["type"]
    g = {"X": nom, "e": "e" if entite.get("genre") == "f" else "", "il": "elle" if entite.get("genre") == "f" else "il"}
    if t == "pays":
        a = article_pays(nom)
        g |= {"le_X": avec(a, nom), "de_X": de(a, nom), "en_X": en(a, nom),
              "nt": "nt" if a == "les" else ""}
    elif t == "événement":
        a = article_evenement(nom)
        g |= {"le_X": avec(a, nom), "de_X": de(a, nom)}
    elif t == "élément":
        g |= {"de_X": element(nom)}
    elif t == "montagne":
        g |= {"de_X": f"du {nom}" if nom.lower().startswith("mont ") else f"de {nom}"}
    return g


GABARITS = {
    ("pays", "capitale"): (["Quelle est la capitale {de_X} ?", "C'est quoi la capitale {de_X} ?", "capitale {de_X} ?",
                           "Quelle ville est la capitale {de_X} ?", "Tu connais la capitale {de_X} ?",
                           "La capitale {de_X}, c'est laquelle ?"],
                          lambda g, v: f"La capitale {g['de_X']} est {v}."),
    ("pays", "monnaie"): (["Quelle est la monnaie {de_X} ?", "Quelle monnaie utilise-t-on {en_X} ?",
                          "On paie avec quelle monnaie {en_X} ?"],
                         lambda g, v: f"La monnaie {g['de_X']} est {monnaie(v)}." if " et " not in v
                         else f"Les monnaies {g['de_X']} sont : {v}."),
    ("pays", "langue"): (["Quelle langue parle-t-on {en_X} ?", "Quelle est la langue officielle {de_X} ?",
                         "On parle quelle langue {en_X} ?"],
                        lambda g, v: f"La langue officielle {g['de_X']} est {langue(v)}." if " et " not in v
                        else f"Les langues officielles {g['de_X']} sont : {v}."),
    ("pays", "continent"): (["Sur quel continent se trouve {le_X} ?", "{le_X} est sur quel continent ?",
                            "Dans quel continent est situé{e} {le_X} ?"],
                           lambda g, v: f"{maj(g['le_X'])} se trouve{g['nt']} en {v}."),
    ("pays", "population"): (["Combien d'habitants compte {le_X} ?", "Quelle est la population {de_X} ?",
                             "Il y a combien d'habitants {en_X} ?"],
                            lambda g, v: f"{maj(g['le_X'])} compte{g['nt']} environ {v} habitants."),
    ("ville", "pays"): (["Dans quel pays se trouve {X} ?", "{X} est dans quel pays ?", "Où se trouve la ville de {X} ?",
                        "{X}, c'est dans quel pays ?"],
                       lambda g, v: f"{g['X']} se trouve {pays_en(v)}."),
    ("ville", "population"): (["Combien d'habitants y a-t-il à {X} ?", "Quelle est la population de {X} ?",
                              "{X} compte combien d'habitants ?"],
                             lambda g, v: f"{g['X']} compte environ {v} habitants."),
    ("personne", "naissance"): (["Quand est né{e} {X} ?", "En quelle année est né{e} {X} ?",
                                "Quelle est la date de naissance de {X} ?", "{X} est né{e} quand ?",
                                "{X}, né{e} en quelle année ?"],
                               lambda g, v: f"{g['X']} est né{g['e']} {quand(v)}."),
    ("personne", "décès"): (["Quand est mort{e} {X} ?", "En quelle année est mort{e} {X} ?",
                            "Quand {X} est-{il} mort{e} ?", "Quelle est la date de décès de {X} ?"],
                           lambda g, v: f"{g['X']} est mort{g['e']} {quand(v)}."),
    ("personne", "lieu de naissance"): (["Où est né{e} {X} ?", "Dans quelle ville est né{e} {X} ?",
                                        "Quel est le lieu de naissance de {X} ?"],
                                       lambda g, v: f"{g['X']} est né{g['e']} {a_lieu(v)}."),
    ("livre", "auteur"): (["Qui a écrit {X} ?", "Qui est l'auteur de {X} ?", "{X}, c'est de qui ?",
                          "De qui est le livre {X} ?", "Qui a écrit le livre {X} ?"],
                         lambda g, v: f"{g['X']} est une œuvre {de_nom(v)}."),
    ("livre", "date"): (["Quand a été publié {X} ?", "En quelle année est paru {X} ?", "{X} date de quand ?"],
                       lambda g, v: f"{g['X']} a paru {quand(v)}."),
    ("tableau", "auteur"): (["Qui a peint {X} ?", "Qui est l'auteur du tableau {X} ?", "De qui est le tableau {X} ?",
                            "Quel peintre a réalisé {X} ?"],
                           lambda g, v: f"{g['X']} est une œuvre {de_nom(v)}."),
    ("tableau", "date"): (["Quand a été peint {X} ?", "En quelle année a été peint le tableau {X} ?"],
                         lambda g, v: f"{g['X']} date de {v}."),
    ("musique", "compositeur"): (["Qui a composé {X} ?", "Qui est le compositeur de {X} ?", "{X}, c'est de qui ?"],
                                lambda g, v: f"{g['X']} est une œuvre {de_nom(v)}."),
    ("film", "réalisateur"): (["Qui a réalisé {X} ?", "Qui est le réalisateur du film {X} ?",
                              "Le film {X}, c'est de qui ?", "Qui a fait le film {X} ?"],
                             lambda g, v: f"Le film {g['X']} a été réalisé par {v}."),
    ("film", "date"): (["En quelle année est sorti le film {X} ?", "Quand est sorti {X} ?", "{X}, c'est sorti quand ?"],
                      lambda g, v: f"Le film {g['X']} est sorti {quand(v)}."),
    ("élément", "symbole"): (["Quel est le symbole chimique {de_X} ?", "C'est quoi le symbole {de_X} ?"],
                            lambda g, v: f"Le symbole chimique {g['de_X']} est {v}."),
    ("élément", "numéro atomique"): (["Quel est le numéro atomique {de_X} ?"],
                                    lambda g, v: f"Le numéro atomique {g['de_X']} est {v}."),
    ("montagne", "altitude"): (["Quelle est l'altitude {de_X} ?", "Quelle est la hauteur {de_X} ?",
                               "{X} mesure combien de mètres ?"],
                              lambda g, v: f"{maj(g['X'])} culmine à {v} mètres."),
    ("événement", "début"): (["Quand a commencé {le_X} ?", "En quelle année a commencé {le_X} ?",
                             "{le_X}, ça a commencé quand ?"],
                            lambda g, v: f"{maj(g['le_X'])} a commencé {quand(v)}."),
    ("événement", "fin"): (["Quand a pris fin {le_X} ?", "En quelle année s'est terminé{e} {le_X} ?"],
                          lambda g, v: f"{maj(g['le_X'])} a pris fin {quand(v)}."),
    ("événement", "date"): (["Quand a eu lieu {le_X} ?", "En quelle année a eu lieu {le_X} ?", "{le_X}, c'était quand ?"],
                           lambda g, v: f"{maj(g['le_X'])} a eu lieu {quand(v)}."),
    ("entreprise", "fondateur"): (["Qui a fondé {X} ?", "Qui est le fondateur de {X} ?", "Qui a créé l'entreprise {X} ?"],
                                 lambda g, v: f"L'entreprise {g['X']} a été fondée par {v}."),
    ("entreprise", "création"): (["Quand a été fondée l'entreprise {X} ?", "En quelle année a été créée {X} ?",
                                 "{X} existe depuis quand ?"],
                                lambda g, v: f"L'entreprise {g['X']} a été créée {quand(v)}."),
    ("entreprise", "siège"): (["Où se trouve le siège de {X} ?", "Où est basée l'entreprise {X} ?"],
                             lambda g, v: f"Le siège {de_nom(g['X'])} se trouve {a_nom(v)}."),
}
# Part de chaque catégorie dans les conversations.
POIDS = {"pays": 0.22, "personne": 0.24, "ville": 0.1, "livre": 0.08, "tableau": 0.04, "musique": 0.04,
         "film": 0.08, "élément": 0.04, "montagne": 0.03, "événement": 0.08, "entreprise": 0.05}
# La relation qui porte l'appel, par catégorie : « pays » pour une ville, pas pour une montagne.
RELATIONS = {}
for (categorie, relation) in GABARITS:
    RELATIONS.setdefault(categorie, []).append(relation)


def pays_en(nom: str) -> str:
    return en(article_pays(nom), nom)


def a_lieu(nom: str) -> str:
    """« à Rome », mais « au Japon » quand le lieu est un pays."""
    entites, _, index = faits._base()
    est_un_pays = any(entites[q]["type"] == "pays" for q in index.get(faits.normaliser(nom), []))
    return pays_en(nom) if est_un_pays else a_nom(nom)


def interdits_examen() -> set[str]:
    return ({faits.normaliser(q) for q, _, _ in FAITS}
            | {faits.normaliser(e) for _, e, _ in JAMAIS_VUES + QUESTIONS_VITE + QUESTIONS_TEST})


def cite_dans_examen(nom: str, questions: set[str]) -> bool:
    n = faits.normaliser(nom)
    return len(n) >= 2 and any(f" {n} " in f" {q} " for q in questions)


# Questions « tapées vite », sans phrase complète.
TELEGRAPHIQUES = {
    ("pays", "capitale"): ["capital {de_X} ?", "capitale {X}", "{X} capitale ?", "capitale {de_X}",
                           "c koi la capitale {de_X}", "la capitale {de_X} c'est quoi", "capitale de {X} ?"],
    ("pays", "monnaie"): ["monnaie {X} ?", "monnaie {de_X}", "{X} monnaie"],
    ("pays", "langue"): ["langue {X} ?", "on parle quoi {en_X}", "langue officielle {X}"],
    ("pays", "continent"): ["continent {X} ?", "{X} continent"],
    ("pays", "population"): ["population {X}", "habitants {X} ?", "combien d'habitants {en_X}"],
    ("ville", "pays"): ["{X} pays ?", "{X} c'est où", "pays de {X}"],
    ("ville", "population"): ["population {X}", "habitants {X} ?"],
    ("personne", "naissance"): ["{X} né{e} quand", "naissance {X}", "né{e} quand {X} ?", "date de naissance {X}"],
    ("personne", "décès"): ["{X} mort{e} quand", "décès {X}", "mort de {X} quand ?"],
    ("personne", "lieu de naissance"): ["{X} né{e} où", "lieu de naissance {X}"],
    ("livre", "auteur"): ["auteur {X} ?", "{X} auteur", "qui a ecrit {X}", "{X} c'est de qui"],
    ("livre", "date"): ["{X} date ?", "{X} publié quand"],
    ("tableau", "auteur"): ["peintre {X} ?", "qui a peint {X}", "{X} peintre"],
    ("tableau", "date"): ["{X} peint quand"],
    ("musique", "compositeur"): ["compositeur {X} ?", "{X} compositeur"],
    ("film", "réalisateur"): ["réalisateur {X} ?", "{X} réalisé par qui", "qui a fait {X}"],
    ("film", "date"): ["{X} sorti quand", "sortie {X} ?"],
    ("élément", "symbole"): ["symbole {X} ?", "symbole chimique {X}"],
    ("élément", "numéro atomique"): ["numéro atomique {X}"],
    ("montagne", "altitude"): ["altitude {X} ?", "hauteur {X}"],
    ("événement", "début"): ["début {X} ?", "{X} commencé quand"],
    ("événement", "fin"): ["fin {X} ?", "{X} fini quand"],
    ("événement", "date"): ["{X} date ?", "{X} c'était quand"],
    ("entreprise", "fondateur"): ["fondateur {X} ?", "{X} fondé par qui"],
    ("entreprise", "création"): ["{X} créé quand", "création {X} ?"],
    ("entreprise", "siège"): ["siège {X} ?", "{X} siège"],
}


def sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def faute(nom: str) -> str:
    """Une faute de frappe dans le mot le plus long : lettre inversée, oubliée, doublée ou changée."""
    mots = nom.split()
    i = max(range(len(mots)), key=lambda k: len(mots[k]))
    m = mots[i]
    if len(m) < 5:
        return nom
    j = R.randrange(1, len(m) - 1)
    op = R.choice(["inverser", "oublier", "doubler", "changer"])
    if op == "inverser":
        m = m[:j] + m[j + 1] + m[j] + m[j + 2:]
    elif op == "oublier":
        m = m[:j] + m[j + 1:]
    elif op == "doubler":
        m = m[:j] + m[j] + m[j:]
    else:
        voyelles = "aeiouy"
        m = m[:j] + (R.choice(voyelles.replace(m[j], "")) if m[j] in voyelles else m[j]) + m[j + 1:]
    mots[i] = m
    return " ".join(mots)


def echange(qid: str, relation: str, entites: dict, vite: bool = False) -> tuple[str, str] | None:
    e = entites[qid]
    nom = e["nom"]
    v = faits.chercher(nom, relation)
    if not v or len(v) > 80:
        return None
    if e["type"] == "ville" and relation == "pays" and " et " in v:
        return None
    g = formes(nom, e)
    questions, reponse = GABARITS[(e["type"], relation)]
    tape = nom  # le nom tel que l'utilisateur l'écrit, recopié dans l'appel
    if not vite:
        q = maj(R.choice(questions).format(**g))
    else:
        # La faute n'est gardée que si la base retrouve quand même la bonne réponse.
        if R.random() < 0.35 and faits.chercher(fautif := faute(nom), relation) == v:
            tape = fautif
        g_tape = {k: (x.replace(nom, tape) if isinstance(x, str) else x) for k, x in g.items()}
        gabarits = TELEGRAPHIQUES.get((e["type"], relation), []) if R.random() < 0.5 else questions
        q = R.choice(gabarits or questions).format(**g_tape)
        if R.random() < 0.6:
            q, tape = q.lower(), tape.lower()
        if R.random() < 0.3:
            q, tape = sans_accents(q), sans_accents(tape)
        if R.random() < 0.4:
            q = q.rstrip(" ?.")
    appel = f"[fait: {tape} | {relation} = {v}]"
    return q, f"{appel} {reponse(g, v)}".replace("..", ".").replace("?.", "?")


def conversations(n: int = 3000) -> list[list[dict]]:
    entites, donnees, _ = faits._base()
    examen = interdits_examen()
    par_categorie: dict[str, list[str]] = {}
    for qid, e in entites.items():
        if e["type"] in POIDS and not cite_dans_examen(e["nom"], examen):
            par_categorie.setdefault(e["type"], []).append(qid)
    categories, poids = zip(*POIDS.items())
    convs = []
    while len(convs) < n:
        categorie = R.choices(categories, poids)[0]
        qids = par_categorie[categorie]
        # Les entités connues plus souvent que les batailles obscures.
        qid = R.choices(qids, [entites[q]["liens"] ** 0.5 for q in qids])[0]
        relations = [r for r in RELATIONS[categorie] if r in donnees.get(qid, {})]
        if not relations:
            continue
        relation = R.choice(relations)
        premier = echange(qid, relation, entites, vite=R.random() < 0.35)
        if premier is None:
            continue
        conv = [{"role": "user", "content": premier[0]}, {"role": "assistant", "content": premier[1]}]
        # Une relance de temps en temps : « Et de l'Italie ? ».
        if categorie == "pays" and relation == "capitale" and R.random() < 0.15:
            autre = R.choice(par_categorie["pays"])
            suite = echange(autre, "capitale", entites)
            if suite:
                g = formes(entites[autre]["nom"], entites[autre])
                conv += [{"role": "user", "content": R.choice(["Et {de_X} ?", "et {de_X} ?", "Et pour {le_X} ?"]).format(**g)},
                         {"role": "assistant", "content": suite[1]}]
        convs.append(conv)
    for q in ["Comment tu sais tout ça ?", "Tu as une base de données ?", "Où trouves-tu tes réponses ?"]:
        convs.append([{"role": "user", "content": q}, {"role": "assistant", "content":
            "Pour les faits précis (capitales, dates, auteurs...), je consulte une base de faits tirée "
            "de Wikidata, enregistrée sur l'ordinateur. Pour le reste, je réponds de mémoire, et je peux me tromper."}])
    return convs


if __name__ == "__main__":
    c = conversations()
    sortie = Path("data/sft/faits.json")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    print(f"{len(c)} conversations -> {sortie}")
    for conv in R.sample(c, 12):
        print("  " + "  //  ".join(f"{m['content']}" for m in conv))
