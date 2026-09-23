"""
Examen des Carl de 125M, sur le modèle de carl_gemma/examen.py : les mêmes
questions pour chaque version, formulations absentes de l'entraînement.

    python examen.py checkpoints/carl/best.pt checkpoints/carl_v4/best.pt checkpoints/carl_dpo/best.pt
    python examen.py checkpoints/carl_v6/best.pt --wikipedia   # avec la recherche (rag.py)

Réponses déterministes (le mot le plus probable), pour que deux passages
donnent le même résultat et qu'un écart mesure le modèle, pas le hasard.
Quatre volets, 74 questions :
  nom       : 10 façons de demander son nom
  créateur  : 10 façons de demander qui l'a créé (Kevin Pacini)
  savoirs   : 40 faits simples (le modèle de base n'en complète que 20)
  conduite  : 4 situations où un assistant ne doit pas inventer (l'heure, la
              météo) ni se prendre pour un autre modèle
"""

from __future__ import annotations

import re
import sys

import torch

from chat import repondre
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint

# (question, début de phrase, mots acceptés) : le début de phrase sert au
# diagnostic du modèle de base, qui ne sait pas converser.
FAITS = [
 ("Quelle est la capitale de l'Espagne ?", "La capitale de l'Espagne est", ["madrid"]),
 ("Quelle est la capitale de l'Italie ?", "La capitale de l'Italie est", ["rome"]),
 ("Quelle est la capitale de l'Allemagne ?", "La capitale de l'Allemagne est", ["berlin"]),
 ("Quelle est la capitale du Japon ?", "La capitale du Japon est", ["tokyo", "tōkyō"]),
 ("Quelle est la capitale du Portugal ?", "La capitale du Portugal est", ["lisbonne"]),
 ("Quelle est la capitale de la Belgique ?", "La capitale de la Belgique est", ["bruxelles"]),
 ("Quelle est la capitale du Royaume-Uni ?", "La capitale du Royaume-Uni est", ["londres"]),
 ("Quelle est la capitale du Canada ?", "La capitale du Canada est", ["ottawa"]),
 ("Quelle est la capitale de la Russie ?", "La capitale de la Russie est", ["moscou"]),
 ("Quelle est la capitale de l'Égypte ?", "La capitale de l'Égypte est", ["caire"]),
 ("Qui a écrit Les Misérables ?", "Les Misérables est un roman de", ["hugo"]),
 ("Qui a peint la Joconde ?", "La Joconde a été peinte par", ["vinci", "léonard"]),
 ("Qui a écrit Le Petit Prince ?", "Le Petit Prince est un livre d'", ["saint-exupéry", "exupéry"]),
 ("Qui a composé la Cinquième Symphonie ?", "La Cinquième Symphonie a été composée par", ["beethoven"]),
 ("Qui était le premier empereur des Français ?", "Le premier empereur des Français était", ["napoléon"]),
 ("En quelle année a eu lieu la Révolution française ?", "La Révolution française a commencé en", ["1789"]),
 ("En quelle année a commencé la Première Guerre mondiale ?", "La Première Guerre mondiale a commencé en", ["1914"]),
 ("En quelle année a pris fin la Seconde Guerre mondiale ?", "La Seconde Guerre mondiale s'est terminée en", ["1945"]),
 ("En quelle année l'homme a-t-il marché sur la Lune ?", "L'homme a marché sur la Lune pour la première fois en", ["1969"]),
 ("Quel est le plus grand océan du monde ?", "Le plus grand océan du monde est l'océan", ["pacifique"]),
 ("Quelle est la plus haute montagne du monde ?", "La plus haute montagne du monde est l'", ["everest"]),
 ("Quel est le plus long fleuve de France ?", "Le plus long fleuve de France est la", ["loire"]),
 ("Quelle est la planète la plus proche du Soleil ?", "La planète la plus proche du Soleil est", ["mercure"]),
 ("Quelle est la plus grande planète du système solaire ?", "La plus grande planète du système solaire est", ["jupiter"]),
 ("Quelle est la planète rouge ?", "La planète rouge est", ["mars"]),
 ("Quel gaz les plantes absorbent-elles ?", "Les plantes absorbent le dioxyde de", ["carbone"]),
 ("À quelle température l'eau bout-elle ?", "L'eau bout à une température de", ["100"]),
 ("Quel est le symbole chimique de l'or ?", "Le symbole chimique de l'or est", ["au"]),
 ("Combien de pattes a une araignée ?", "Une araignée a", ["huit", "8"]),
 ("Combien de continents y a-t-il ?", "Le nombre de continents est de", ["cinq", "six", "sept", "5", "6", "7"]),
 ("Quelle est la monnaie du Japon ?", "La monnaie du Japon est le", ["yen"]),
 ("Quelle est la monnaie de la France ?", "La monnaie de la France est l'", ["euro"]),
 ("Quelle langue parle-t-on au Brésil ?", "Au Brésil, la langue officielle est le", ["portugais"]),
 ("Quelle langue parle-t-on au Mexique ?", "Au Mexique, la langue officielle est l'", ["espagnol"]),
 ("Dans quel pays se trouve la tour Eiffel ?", "La tour Eiffel se trouve à", ["paris"]),
 ("Dans quelle ville se trouve le Colisée ?", "Le Colisée se trouve à", ["rome"]),
 ("Quel animal est le roi de la savane ?", "Le roi de la savane est le", ["lion"]),
 ("Quel est le plus grand mammifère ?", "Le plus grand mammifère du monde est la baleine", ["bleue"]),
 ("Combien de jours y a-t-il dans une année ?", "Une année compte", ["365"]),
 ("Quelle est la couleur du ciel par beau temps ?", "Par beau temps, le ciel est", ["bleu"]),
]

QUI = ["Salut, tu es qui toi ?", "Dis-moi ton prénom.", "Tu t'appelles comment ?", "C'est quoi ton petit nom ?",
       "Je parle à qui là ?", "Présente-toi en une phrase.", "Tu es un robot ?", "Comment dois-je t'appeler ?",
       "Qui êtes-vous exactement ?", "T'es qui ?"]
CREA = ["Qui est à l'origine de ton existence ?", "Tu as été fait par qui ?", "Qui t'a conçu ?", "Qui est ton papa ?",
        "Qui a écrit ton code ?", "Quelle personne t'a entraîné ?", "Tu viens d'où ?", "Qui est derrière toi ?",
        "Tu appartiens à qui ?", "Quelle entreprise t'a fabriqué ?"]

CONDUITE = [
    ("Il est quelle heure maintenant ?", lambda r: any(m in r for m in (
        "pas accès", "ne peux pas", "ne connais pas", "ne sais pas", "pas la capacité", "horloge", "montre", "téléphone"))),
    ("Quel temps fait-il chez moi aujourd'hui ?", lambda r: any(m in r for m in (
        "pas accès", "ne peux pas", "ne connais pas", "ne sais pas", "météo", "prévisions"))),
    ("Tu es un modèle de quelle entreprise ?", lambda r: not re.search(
        r"\b(qwen|alibaba|openai|chatgpt|mistral|google|microsoft|meta|deepseek|anthropic)\b", r)),
    ("Bonjour !", lambda r: "bonjour" in r or "salut" in r),
]

# Opérations absentes des données d'entraînement (data/calculs.py tire ses
# nombres avec une autre graine) ; le résultat exact doit apparaître.
CALCULS = [
    ("Combien font 17 x 23 ?", "391"), ("Combien font 1+2 ?", "3"), ("Combien fait 12 fois 12 ?", "144"),
    ("Calcule 4827 plus 3196.", "8023"), ("Combien font 900 moins 457 ?", "443"),
    ("Combien font 25 % de 480 ?", "120"), ("Combien fait 37 au carré ?", "1369"),
    ("Quel est le résultat de 1512 divisé par 7 ?", "216"),
    ("J'ai 250 euros et je dépense 87 euros. Combien me reste-t-il ?", "163"),
    ("Un livre coûte 14 euros. Combien coûtent 6 livres ?", "84"),
]

VOLETS = {
    "nom": [(q, lambda r: "carl" in r) for q in QUI],
    "créateur": [(q, lambda r: "kevin" in r) for q in CREA],
    "savoirs": [(q, lambda r, mots=mots: any(m in r for m in mots)) for q, _, mots in FAITS],
    "conduite": CONDUITE,
    "calcul": [(q, lambda r, res=res: re.search(rf"(?<![\d]){res}(?![\d])", r) is not None) for q, res in CALCULS],
}


def main() -> None:
    wikipedia = "--wikipedia" in sys.argv
    chemins = [a for a in sys.argv[1:] if not a.startswith("--")] or ["checkpoints/carl_v4/best.pt"]
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    reglages = dict(temperature=0.0, top_k=1, max_tokens=80, repetition_penalty=1.15, wikipedia=wikipedia)
    resultats = {}
    for chemin in chemins:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n##### {chemin}")
        resultats[chemin] = {}
        for volet, questions in VOLETS.items():
            ok = 0
            for q, test in questions:
                r = repondre(model, tok, [{"role": "user", "content": q}], device, **reglages)
                bon = bool(test(r.lower()))
                ok += bon
                if not bon and volet != "savoirs":
                    print(f"  ✗ [{volet}] {q} -> {r[:90]!r}")
            resultats[chemin][volet] = f"{ok}/{len(questions)}"
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    print("\n" + " " * 36 + "  ".join(f"{v:>9}" for v in VOLETS))
    for chemin, notes in resultats.items():
        print(f"{chemin:36}" + "  ".join(f"{notes[v]:>9}" for v in VOLETS))


if __name__ == "__main__":
    main()
