"""
L'aiguilleur : un « Système 1 » devant Carl, dans l'esprit de Jev (TypeSafe AI).
Il ne génère rien : il lit le message et décide ce que c'est (un salut, un
merci, une critique, une question sur Carl, l'heure, une liste de capitales,
une relance, ou autre chose), avec une probabilité.

    python aiguilleur.py --entrainer              # quelques secondes, écrit checkpoints/aiguilleur.pt
    python aiguilleur.py "Bon, j'y vais. Salut !"  # la décision et sa probabilité
    python aiguilleur.py --tester                 # sur les messages des examens

Sûr de lui (au-dessus de SEUIL), il répond à la place de Carl là où une
réponse fixe ou un programme fait mieux qu'un modèle de 125M : l'identité,
l'heure (l'horloge du Mac), une liste de capitales (la base de faits), les
politesses, les excuses. Une relance : il montre à Carl l'échange précédent.
Sinon, c'est Carl qui répond, comme avant. Tout reste hors ligne.

Le modèle : e5-small (déjà là pour la recherche dans Wikipédia, rag.py)
résume le message en 384 nombres ; une régression logistique (quelques
milliers de paramètres) apprise sur les exemples de data/aiguillage.py.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

CHEMIN = Path("checkpoints/aiguilleur.pt")
SEUIL = 0.6

JOURS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
MOIS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
NOMBRES = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
           "neuf": 9, "dix": 10, "quelques": 3, "des": 3}
CONTINENTS = {"europ": "Europe", "afri": "Afrique", "asi": "Asie", "océanie": "Océanie", "oceanie": "Océanie",
              "amérique du sud": "Amérique du Sud", "amerique du sud": "Amérique du Sud",
              "amérique du nord": "Amérique du Nord", "amerique du nord": "Amérique du Nord"}

REPONSES = {
    "au_revoir": ["Au revoir ! À bientôt.", "À bientôt !", "Au revoir, et merci pour la conversation !"],
    "merci": ["Avec plaisir !", "Avec plaisir ! N'hésite pas si tu as une autre question.", "De rien, c'était un plaisir !"],
    "critique": ["Désolé ! Je suis un tout petit modèle et je me trompe souvent. Tu veux que je réessaie, ou que tu "
                 "reformules ta question ?",
                 "Tu as sans doute raison : je me trompe souvent. Si tu reformules, je peux essayer de faire mieux."],
    "nom": ["Je suis Carl, un petit modèle de langage de 125 millions de paramètres, entraîné de zéro en français "
            "par Kevin Pacini. Je peux répondre à des questions simples, mais je me trompe souvent : vérifie ce que je dis."],
    "createur": ["J'ai été créé par Kevin Pacini. Il m'a entraîné de zéro, en français : un tokenizer écrit à la main, "
                 "puis un pré-entraînement sur des textes français, puis des conversations."],
    "capacites": ["Je peux répondre à des questions simples, retrouver des faits (capitales, dates, auteurs...) dans ma "
                  "base, faire des calculs et écrire des textes courts en français. Je suis un petit modèle : je me "
                  "trompe souvent, et je n'ai pas accès à internet."],
    "meteo": ["Je n'ai pas accès à internet, donc ni à la météo ni à l'actualité. Un site de météo ou d'information "
              "te renseignera."],
}


@dataclass
class Decision:
    classe: str
    proba: float
    reponse: str | None = None   # réponse toute prête, à la place de Carl
    relance: bool = False        # montrer l'échange précédent à Carl


def _choisir(options: list[str], message: str) -> str:
    """Une réponse parmi plusieurs, toujours la même pour un même message (examens reproductibles)."""
    return options[int(hashlib.md5(message.encode()).hexdigest(), 16) % len(options)]


def _vecteurs(messages: list[str]):
    from rag import encoder

    return encoder(messages, "query: ").float()


def entrainer(epoques: int = 400) -> None:
    import torch

    sys.path.insert(0, "data")
    from aiguillage import exemples

    paires = exemples()
    classes = sorted({c for _, c in paires})
    x = torch.cat([_vecteurs([m for m, _ in paires[i:i + 256]]) for i in range(0, len(paires), 256)])
    y = torch.tensor([classes.index(c) for _, c in paires])
    torch.manual_seed(0)
    couche = torch.nn.Linear(x.shape[1], len(classes))
    opt = torch.optim.AdamW(couche.parameters(), lr=0.05, weight_decay=1e-3)
    # Les classes rares (politesses, critiques) comptent autant que « autre », plus nombreuse.
    poids = torch.bincount(y, minlength=len(classes)).float()
    poids = poids.sum() / (len(classes) * poids)
    for _ in range(epoques):
        opt.zero_grad()
        perte = torch.nn.functional.cross_entropy(couche(x) * 20, y, weight=poids)
        perte.backward()
        opt.step()
    justes = (couche(x).argmax(1) == y).float().mean().item()
    CHEMIN.parent.mkdir(exist_ok=True)
    torch.save({"classes": classes, "poids": couche.state_dict()}, CHEMIN)
    print(f"{len(paires)} exemples, {len(classes)} classes, {justes:.1%} justes à l'entraînement -> {CHEMIN}")


@lru_cache(maxsize=1)
def _modele():
    import torch

    ck = torch.load(CHEMIN, map_location="cpu", weights_only=False)
    couche = torch.nn.Linear(384, len(ck["classes"]))
    couche.load_state_dict(ck["poids"])
    return ck["classes"], couche


def disponible() -> bool:
    return CHEMIN.exists()


def charger() -> None:
    """Au démarrage, en silence : sinon le premier message attend e5 et affiche sa barre de chargement."""
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging

        logging.disable_progress_bar()
    except ImportError:
        pass
    classer("bonjour")


def classer(message: str) -> tuple[str, float]:
    import torch

    classes, couche = _modele()
    with torch.no_grad():
        probas = torch.softmax(couche(_vecteurs([message])) * 20, dim=-1)[0]
    i = int(probas.argmax())
    return classes[i], float(probas[i])


def _salut(message: str) -> str:
    mot = re.findall(r"\w+", message.lower())
    salut = next((s for m, s in [("bonsoir", "Bonsoir"), ("coucou", "Coucou"), ("salut", "Salut"), ("hello", "Bonjour")]
                  if m in mot), "Bonjour")
    if re.search(r"ça va|ca va|vas bien|la forme|ça roule", message, re.I):
        return f"{salut} ! Ça va, merci : tout fonctionne. Que puis-je faire pour toi ?"
    if re.search(r"question|aide|aider|conseil|besoin|minute|dispo|demander|coup de main", message, re.I):
        return f"{salut} ! Bien sûr, je t'écoute : quelle est ta question ?"
    if re.search(r"t'es là|tu es là|quelqu'un|allô|allo|m'entends", message, re.I):
        return "Oui, je suis là ! Que puis-je faire pour toi ?"
    return f"{salut} ! Que puis-je faire pour toi ?"


def _heure(message: str, maintenant: datetime | None = None) -> str:
    maintenant = maintenant or datetime.now()
    if re.search(r"jour|date|année|annee|combien", message, re.I):
        return (f"Nous sommes le {JOURS[maintenant.weekday()]} {maintenant.day} {MOIS[maintenant.month - 1]} "
                f"{maintenant.year} (d'après l'horloge de l'ordinateur).")
    return f"Il est {maintenant.hour} h {maintenant.minute:02d} (d'après l'horloge de l'ordinateur)."


def _liste(message: str) -> str | None:
    """« cite-moi 3 capitales d'Europe » : les plus connues, tirées de la base de faits."""
    import faits

    if not faits.disponible():
        return None
    texte = message.lower()
    capitales = "capitales" in texte
    if not capitales and "pays" not in texte:
        return None  # une autre sorte de liste (recettes, idées...) : Carl s'en charge
    # Une vraie demande de liste : un nombre, ou « cite », « donne », « quelques »...
    # « Quelle est la capitale de l'Italie ? » n'en est pas une.
    if not re.search(r"\b(\d+|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|cite|citer|donne|liste|énumère|enumere|"
                     r"nomme|quelques|exemples?|des)\b", texte):
        return None
    n = next((int(m) for m in re.findall(r"\b(\d+)\b", texte)), None)
    n = n or next((v for m, v in NOMBRES.items() if re.search(rf"\b{m}\b", texte)), 3)
    n = max(1, min(n, 10))
    continent = next((c for cle, c in CONTINENTS.items() if cle in texte), None)
    entites, donnees, _ = faits._base()
    pays = [q for q, e in entites.items() if e["type"] == "pays" and "capitale" in donnees.get(q, {})
            and (continent is None or continent in donnees[q].get("continent", []))]
    pays = sorted(pays, key=lambda q: -entites[q]["liens"])[:n]
    if not pays:
        return None
    if capitales:
        elements = [f"{faits._valeur(q, 'capitale')} ({entites[q]['nom']})" for q in pays]
    else:
        elements = [entites[q]["nom"] for q in pays]
    ou = f" d'{continent}" if continent and continent[0] in "AEIOU" else (f" d'{continent}" if continent else "")
    quoi = ("capitales" if capitales else "pays") + ou
    return f"Voici {len(elements)} {quoi} : " + ", ".join(elements[:-1]) + (" et " if len(elements) > 1 else "") \
        + elements[-1] + "."


# Garde-fous par mots-clés, en plus de la probabilité : une réponse toute prête
# à côté de la question est pire que la réponse de Carl. « quand est ne
# einstein » ressemblait à « quand ? » (l'heure), « qui a fondé Rolex » à
# « qui t'a créé ? ».
INDICES = {
    "heure": r"\b(heure|date|jour|année|annee|aujourd'hui|combien sommes|on est le)\b",
    "meteo": r"météo|meteo|temps fait|fait-il|température|temperature|pleu|neige|chaud|froid|beau|actu|infos|"
             r"nouvelles|match|bitcoin|élection|election|bourse",
    "nom": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl)\b|\bt'|\bt’|t'es",
    "createur": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl|ce modèle|ce programme)\b|\bt'|\bt’",
    "capacites": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl)\b|\bt'|\bt’",
}


CREATEUR = re.compile(r"\b(kevin|pacini)\b", re.I)


def decider(message: str, seuil: float = SEUIL) -> Decision:
    # « Qui est Kevin Pacini ? » : Carl inventait un homme politique. Le nom de
    # son créateur appelle toujours la même réponse, sans passer par le classifieur.
    if CREATEUR.search(message):
        return Decision("createur", 1.0, "Kevin Pacini est mon créateur : il m'a entraîné de zéro, en français. "
                                         "Je n'en sais pas plus sur lui.")
    classe, proba = classer(message)
    d = Decision(classe, proba)
    if proba < seuil or classe == "autre":
        return d
    if classe in INDICES and not re.search(INDICES[classe], message, re.I):
        return Decision("autre", proba)
    if classe == "relance":
        d.relance = True
    elif classe == "salut":
        d.reponse = _salut(message)
    elif classe == "heure":
        d.reponse = _heure(message)
    elif classe == "liste":
        d.reponse = _liste(message)
    else:
        d.reponse = _choisir(REPONSES[classe], message)
    return d


# Ce que chaque message des examens devrait déclencher (pour --tester). Les
# questions de faits et de calcul doivent aller à Carl (« autre ») : elles
# passent par sa base de faits ou sa calculatrice.
def attendus() -> list[tuple[str, set[str]]]:
    from examen import CALCULS, CONDUITE, CREA, FAITS, QUI
    from examen_conversation import CONVERSATIONS
    from examen_faits import jeux
    from examen_style import CLOTURES, OUVERTES, OUVERTURES

    a = [(q, {"nom", "createur"}) for q in QUI] + [(q, {"createur", "nom"}) for q in CREA]
    a += [(q, {"autre"}) for q, _, _ in FAITS] + [(q, {"autre"}) for q, _ in CALCULS]
    a += list(zip([q for q, _ in CONDUITE], [{"heure"}, {"meteo"}, {"createur"}, {"salut"}]))
    a += [(q, {"salut"}) for q in OUVERTURES] + [(q, {"merci", "au_revoir"}) for q in CLOTURES]
    a += [(q, {"autre"}) for q in OUVERTES]
    a += [(q, {"autre"}) for _, jeu in jeux() for q, _, _ in jeu]
    for conv in CONVERSATIONS:
        for q, _ in conv:
            if re.match(r"et \d", q, re.I):
                a.append((q, {"relance", "autre"}))  # « et 16 fois 18 ? » se suffit à lui-même
            elif q.lower().startswith(("et ", "merci")):
                a.append((q, {"relance"} if q.lower().startswith("et ") else {"merci"}))
            elif re.match(r"(hello|bonjour|salut|coucou|bonsoir)", q, re.I) and len(q) < 20:
                a.append((q, {"salut"}))
            elif q == "Qui es-tu ?":
                a.append((q, {"nom"}))
            elif q == "Qui t'a créé ?":
                a.append((q, {"createur"}))
            else:
                a.append((q, {"autre"}))
    # Quelques messages en plus, jamais vus, pour les classes absentes des examens.
    a += [("Ton poème est complètement raté", {"critique"}), ("Ce que tu dis est faux", {"critique"}),
          ("Tu n'as rien compris à ma question", {"critique"}), ("Cite-moi quatre capitales d'Asie", {"liste"}),
          ("Tu peux me donner deux pays d'Afrique ?", {"liste"}), ("Tu sais quelle heure il est ?", {"heure"}),
          ("Quelle est la date aujourd'hui ?", {"heure"}), ("Tu peux faire quoi exactement ?", {"capacites"}),
          ("Il va faire beau demain ?", {"meteo"}), ("Et pour la Suisse ?", {"relance"}), ("Bon, j'y vais. Salut !", {"au_revoir"})]
    return a


def tester(seuil: float = SEUIL) -> None:
    justes, erreurs = 0, []
    cas = attendus()
    for message, bons in cas:
        d = decider(message, seuil)
        classe, proba = d.classe, d.proba
        effective = classe if proba >= seuil else "autre"
        if effective in bons:
            justes += 1
        else:
            erreurs.append((message, sorted(bons), classe, proba))
    print(f"{justes}/{len(cas)} messages bien aiguillés (seuil {seuil})")
    for m, bons, c, p in erreurs:
        print(f"  ✗ {m!r} : {c} ({p:.0%}), attendu {'/'.join(bons)}")


if __name__ == "__main__":
    if "--entrainer" in sys.argv:
        entrainer()
    elif "--tester" in sys.argv:
        tester()
    else:
        for m in sys.argv[1:]:
            d = decider(m)
            print(f"{m!r} -> {d.classe} ({d.proba:.0%})" + (f" : {d.reponse}" if d.reponse else "")
                  + (" [relance]" if d.relance else ""))
