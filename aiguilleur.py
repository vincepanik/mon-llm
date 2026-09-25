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
    # D'abord les pays tout entiers sur ce continent : la Russie et la Turquie sont aussi en Europe.
    pays = sorted(pays, key=lambda q: (continent is not None and len(donnees[q].get("continent", [])) > 1,
                                       -entites[q]["liens"]))[:n]
    if not pays:
        return None
    if capitales:
        # Une capitale par pays, même quand il en a plusieurs (Afrique du Sud : Pretoria).
        elements = [f"{donnees[q]['capitale'][0]} ({entites[q]['nom']})" for q in pays]
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
    # « chaud », « froid » seulement pour le temps qu'il fait : « le feu est-il chaud ? » n'est pas la météo.
    "meteo": r"météo|meteo|temps fait|fait-il|température|temperature|pleu|neige|(fait|fera|dehors).{0,12}(chaud|froid|beau)|"
             r"actu|infos|nouvelles|match|bitcoin|élection|election|bourse",
    "nom": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl)\b|\bt'|\bt’|t'es",
    "createur": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl|ce modèle|ce programme)\b|\bt'|\bt’",
    "capacites": r"\b(tu|te|toi|ton|ta|tes|vous|votre|carl)\b|\bt'|\bt’",
}


# --- Les questions de faits, posées directement à la base ---
# Carl sait appeler sa base de faits, mais souvent pas : « shinning realisé par
# qui » -> « je ne peux pas... », « napoleon ville natale » -> la date de
# naissance. L'aiguilleur reconnaît la relation par ses mots-clés, prend ce qui
# reste comme nom, et interroge la base lui-même (faits.py, qui se tait quand ce
# n'est pas net). La relation la plus précise d'abord (« lieu de naissance »
# avant « naissance », « fondé par » avant « fondé en »).
RELATIONS_DIRECTES = [
    ("description", r"^(qui (est|etait|était|c'est|c est)|c'est qui|c est qui|qui c'est|c'était qui|c etait qui)\b"
                    r"|,\s*(c'est|c'était|c etait|c est) qui\b"),
    ("lieu de naissance", r"\b(ville natale|lieu de naissance|n[ée]e? où|n[ée]e? ou|où est n[ée]e?|ou est n[ée]e?|"
                          r"n[ée]e? dans quelle ville)\b"),
    ("capitale", r"\bcapitale?s?\b"),
    ("monnaie", r"\b(monnaie|monaie|devise|on paye? avec|payer avec)\b"),
    ("langue", r"\b(langues?( officielles?)?|on parle|parle-t-on|parlent)\b"),
    ("continent", r"\bcontinent\b"),
    ("population", r"\b(population|habitants?|hab|pop|nb habitants)\b"),
    ("naissance", r"\b(date de naissance|naissance|n[ée]e?|naît|nait)\b"),
    ("décès", r"\b(mort|morte|décès|deces|décédée?|decede|meurt|mourut)\b"),
    ("compositeur", r"\b(compos\w*)\b"),
    ("réalisateur", r"\b(réalis\w*|realis\w*|cinéaste|cineaste|tourn[ée]|film de qui)\b"),
    ("auteur", r"\b(écrit|ecrit|auteur|peint|peintre|écrivain|ecrivain|de qui)\b"),
    ("symbole", r"\bsymbol\w*( chimique)?\b"),
    ("numéro atomique", r"\b(num[ée]ro|num) atomique\b"),
    ("altitude", r"\b(altitude|hauteur|culmine)\b"),
    ("fondateur", r"\b(fondateur|fond[ée]e? par|qui a fond[ée]|créée? par|cree par)\b"),
    ("création", r"\b(cr[ée][ée]e? en|fond[ée]e? en|création|creation|fond[ée]e? quand|cr[ée][ée]e? quand|existe depuis)\b"),
    ("siège", r"\b(siège|siege|bas[ée]e? où|bas[ée]e?)\b"),
    ("début", r"\b(commenc\w*|début|debut|déclench\w*)\b"),
    ("fin", r"\b(fin|termin\w*|fini)\b"),
    ("pays", r"\b(dans quel pays|quel pays|ds quel pays|pays)\b"),
    ("date", r"\b(sorti\w*|publi\w*|paru|date|quand|quelle ann[ée]e|en quelle annee)\b"),
]
REMPLISSAGE = set("""c koi quoi quel quelle quels quelles qui que qu est ce c'est cest est es a été ete était etait la le les l
    de du des d en au aux à se trouve trouve situe situé située ds dans où ou quand comment combien y il elle on t
    svp stp moi dis donne peux tu me connais sais sur par ça ca ?""".split())
DEUXIEME_PERSONNE = re.compile(r"\b(tu|te|toi|ton|ta|tes|vous|votre|vos)\b", re.I)


def _nom_restant(message: str, motif: str) -> str | None:
    """Ce qui reste de la question une fois la relation et les mots-outils enlevés, aux bords seulement."""
    texte = re.sub(motif, " ", message.replace("’", "'"), flags=re.I)
    mots = [m for m in re.findall(r"[\w'-]+", texte)]
    mots = [re.sub(r"^(l|d|qu|s)'", "", m, flags=re.I) for m in mots]
    while mots and (mots[0].lower() in REMPLISSAGE or mots[0].lower().startswith("-")):
        mots.pop(0)
    while mots and mots[-1].lower() in REMPLISSAGE | {"t-il", "t-elle", "t-on", "-t-il"}:
        mots.pop()
    nom = " ".join(mots).strip(" -'")
    return nom if len(nom) >= 2 else None


def _phrase(qid: str, relation: str, valeur: str) -> str:
    """Une phrase comme celles que Carl apprend (data/faits_sft.py) : « La capitale du Maroc est Rabat. »"""
    import faits

    sys.path.insert(0, "data")
    from faits_sft import GABARITS, formes, maj

    e = faits.entite(qid)
    if relation == "description":
        dates = ""
        naissance, deces = (e["faits"].get(r, [""])[0] for r in ("naissance", "décès"))
        if naissance and not re.search(r"\d{3,4}", valeur):
            annees = [re.findall(r"\d{3,4}", d)[-1] for d in (naissance, deces) if re.findall(r"\d{3,4}", d)]
            if len(annees) == 2:
                dates = f" ({annees[0]}–{annees[1]})"
            elif annees:
                dates = f" (né{'e' if e.get('genre') == 'f' else ''} en {annees[0]})"
        return f"{maj(e['nom'])}{dates} : {valeur.rstrip('.')}."
    cle = (e["type"], relation)
    if cle in GABARITS:
        return GABARITS[cle][1](formes(e["nom"], e), valeur).replace("..", ".")
    return f"{maj(e['nom'])} ({relation}) : {valeur}."


def fait_direct(message: str) -> str | None:
    """« c koi la capitale du marok » -> « La capitale du Maroc est Rabat. », ou None si pas sûr."""
    import faits

    if not faits.disponible() or DEUXIEME_PERSONNE.search(message) or CREATEUR.search(message):
        return None
    for relation, motif in RELATIONS_DIRECTES:
        if not re.search(motif, message, re.I):
            continue
        nom = _nom_restant(message, motif)
        if not nom:
            return None
        trouve = faits.trouver(nom, relation)
        if not trouve:
            return None  # la relation est reconnue, mais la base ne sait pas : Carl répondra
        qid, groupe = trouve
        vraie = next(r for r in groupe if not r.startswith("@") and faits._valeur(qid, r) is not None)
        return _phrase(qid, vraie, faits._valeur(qid, vraie))
    return None


CREATEUR = re.compile(r"\b(kevin|pacini)\b", re.I)


def decider(message: str, seuil: float = SEUIL) -> Decision:
    # « Qui est Kevin Pacini ? » : Carl inventait un homme politique. Le nom de
    # son créateur appelle toujours la même réponse, sans passer par le classifieur.
    if CREATEUR.search(message):
        return Decision("createur", 1.0, "Kevin Pacini est mon créateur : il m'a entraîné de zéro, en français. "
                                         "Je n'en sais pas plus sur lui.")
    classe, proba = classer(message)
    d = Decision(classe, proba)
    if proba < seuil or classe in ("autre", "liste"):
        # Une question de faits ? La base répond elle-même, si elle est sûre.
        if classe == "liste" and proba >= seuil and (liste := _liste(message)):
            return Decision("liste", proba, liste)
        if (reponse := fait_direct(message)):
            return Decision("fait", proba, reponse)
        return Decision("autre", proba) if classe == "liste" else d
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
        if effective == "fait":
            effective = "autre"  # une question de faits que la base a prise elle-même : c'est son rôle
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
