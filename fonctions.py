"""
Les fonctions de l'ordinateur que Carl peut utiliser, sans risque et hors
ligne : un minuteur ou un rappel, la batterie, des calculs de dates, des
conversions d'unités, des calculs posés directement.

    python fonctions.py "quel jour tombe Noël ?"

L'aiguilleur (aiguilleur.py) les appelle avant tout le reste : ces demandes ont
une forme reconnaissable (un nombre et une unité, une fête, « préviens-moi
dans »), qu'une règle exacte reconnaît sans se tromper, mieux qu'un modèle.
Carl, à 125M, ne décide jamais lui-même d'une action : il ne fait qu'écrire.

Rien ne modifie l'ordinateur : un rappel affiche un message et une
notification macOS, c'est tout (et il ne vit que tant que chat.py tourne).
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from outils import calculer

JOURS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
MOIS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
MOIS_SANS_ACCENT = [m.replace("é", "e").replace("û", "u") for m in MOIS]
NOMBRES = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
           "neuf": 9, "dix": 10, "quinze": 15, "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50}


@dataclass
class Reponse:
    classe: str
    texte: str
    action: Callable[[], None] | None = None  # à lancer par chat.py (un minuteur), jamais par les examens


def _nombre(texte: str) -> float | None:
    texte = texte.strip().replace(",", ".")
    texte = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", texte)
    try:
        return float(texte)
    except ValueError:
        return NOMBRES.get(texte.lower())


def _jolie(x: float) -> str:
    """12.0 -> « 12 » ; 3.28084 -> « 3,28 »."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")


# --- Minuteur et rappel ---

MINUTEUR = re.compile(r"(minuteur|chrono|timer|pr[ée]viens[- ]moi|rappelle[- ]moi|r[ée]veille[- ]moi|alarme)"
                      r".*?(?:dans|de|pour)?\s*(\d+|une?|deux|trois|cinq|dix|quinze|vingt|trente)\s*"
                      r"(secondes?|s|minutes?|min|mn|heures?|h)\b\s*(?:(?:pour|de|d'|qu'?il faut|que)\s*(.+))?", re.I)
UNITES_TEMPS = {"s": 1, "seconde": 1, "secondes": 1, "min": 60, "mn": 60, "minute": 60, "minutes": 60,
                "h": 3600, "heure": 3600, "heures": 3600}


def notifier(texte: str) -> None:
    """Un message dans le terminal et une notification macOS (sans rien d'autre)."""
    print(f"\n⏰ Rappel : {texte}\nvous > ", end="", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display notification "{texte}" with title "Carl" sound name "Glass"'],
                       check=False, capture_output=True)


def minuteur(message: str) -> Reponse | None:
    m = MINUTEUR.search(message)
    if not m:
        return None
    n = _nombre(m.group(2))
    unite = m.group(3).lower()
    if not n or unite not in UNITES_TEMPS:
        return None
    secondes = int(n * UNITES_TEMPS[unite])
    if not 1 <= secondes <= 24 * 3600:
        return None
    quoi = (m.group(4) or "").strip(" .!?")
    duree = f"{_jolie(n)} {({1: 'seconde', 60: 'minute', 3600: 'heure'}[UNITES_TEMPS[unite]])}{'s' if n > 1 else ''}"
    texte = f"C'est noté : je te préviens dans {duree}{f' ({quoi})' if quoi else ''}. Laisse cette fenêtre ouverte."
    rappel = quoi[:1].upper() + quoi[1:] if quoi else f"{duree}, c'est fini !"
    return Reponse("minuteur", texte, lambda: threading.Timer(secondes, notifier, args=[rappel]).start())


# --- Batterie ---

def batterie(message: str) -> Reponse | None:
    if not re.search(r"\bbatterie\b|\bcharge\b.*\b(ordi|mac|ordinateur)\b|\bautonomie\b", message, re.I):
        return None
    if sys.platform != "darwin":
        return Reponse("batterie", "Je ne sais lire la batterie que sur un Mac.")
    sortie = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"(\d+)%;\s*([\w ]+?);\s*([\d:]+ remaining|\(no estimate\)|[\d:]+)?", sortie)
    if not m:
        return Reponse("batterie", "Je n'arrive pas à lire l'état de la batterie.")
    pourcent, etat = m.group(1), m.group(2)
    secteur = "AC Power" in sortie
    phrase = f"La batterie est à {pourcent} %"
    if secteur:
        phrase += ", l'ordinateur est branché" + (" et en charge." if "charging" in etat and "dis" not in etat else ".")
    else:
        reste = re.search(r"(\d+):(\d+) remaining", sortie)
        phrase += ", sur batterie" + (f" (environ {int(reste.group(1))} h {reste.group(2)} restantes)." if reste else ".")
    return Reponse("batterie", phrase)


# --- Dates ---

FETES = {"noel": (12, 25), "noël": (12, 25), "jour de l'an": (1, 1), "nouvel an": (1, 1), "saint-valentin": (2, 14),
         "saint valentin": (2, 14), "fête du travail": (5, 1), "fete du travail": (5, 1), "1er mai": (5, 1),
         "8 mai": (5, 8), "14 juillet": (7, 14), "fête nationale": (7, 14), "fete nationale": (7, 14),
         "15 août": (8, 15), "15 aout": (8, 15), "toussaint": (11, 1), "halloween": (10, 31), "11 novembre": (11, 11)}
AFFICHAGE = {"noel": "Noël", "noël": "Noël", "fete du travail": "la fête du Travail", "fête du travail": "la fête du Travail",
             "fete nationale": "la fête nationale", "fête nationale": "la fête nationale", "saint valentin": "la Saint-Valentin",
             "saint-valentin": "la Saint-Valentin", "toussaint": "la Toussaint", "halloween": "Halloween",
             "jour de l'an": "le jour de l'an", "nouvel an": "le nouvel an", "15 aout": "le 15 août"}
DATE_ECRITE = re.compile(r"\b(1er|\d{1,2})\s+(" + "|".join(MOIS + MOIS_SANS_ACCENT) + r")(?:\s+(\d{4}))?\b", re.I)


def _prochaine(mois: int, jour: int, aujourd_hui: date, annee: int | None = None) -> date | None:
    try:
        d = date(annee or aujourd_hui.year, mois, jour)
    except ValueError:
        return None
    if annee is None and d < aujourd_hui:
        d = d.replace(year=d.year + 1)
    return d


def _ecrire(d: date) -> str:
    return f"{JOURS[d.weekday()]} {'1er' if d.day == 1 else d.day} {MOIS[d.month - 1]} {d.year}"


def dates(message: str, aujourd_hui: date | None = None) -> Reponse | None:
    aujourd_hui = aujourd_hui or date.today()
    texte = message.lower()
    # « dans 10 jours, on sera quel jour ? », « quel jour serons-nous dans 3 semaines »
    m = re.search(r"dans\s+(\d+|une?|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|quinze|vingt|trente)\s+"
                  r"(jours?|semaines?|mois)\b", texte)
    if m and re.search(r"quel jour|quelle date|on sera|sera-t-on|serons", texte):
        n = int(_nombre(m.group(1)) or 0)
        if m.group(2).startswith("jour"):
            d = aujourd_hui + timedelta(days=n)
        elif m.group(2).startswith("semaine"):
            d = aujourd_hui + timedelta(weeks=n)
        else:
            mois = aujourd_hui.month - 1 + n
            d = _prochaine(mois % 12 + 1, min(aujourd_hui.day, 28), aujourd_hui, aujourd_hui.year + mois // 12)
        return Reponse("dates", f"Dans {n} {m.group(2)}, nous serons le {_ecrire(d)}.")
    cible, nom = None, None
    for fete, (mo, jo) in FETES.items():
        if fete in texte:
            cible, nom = _prochaine(mo, jo, aujourd_hui), AFFICHAGE.get(fete, fete)
            break
    if cible is None and (m := DATE_ECRITE.search(texte)):
        jour = 1 if m.group(1).lower() == "1er" else int(m.group(1))
        mois_nom = m.group(2).lower()
        mois = (MOIS.index(mois_nom) if mois_nom in MOIS else MOIS_SANS_ACCENT.index(mois_nom)) + 1
        cible = _prochaine(mois, jour, aujourd_hui, int(m.group(3)) if m.group(3) else None)
        nom = m.group(0)
    if cible is None or not re.search(r"quel jour|tombe|combien de jours|dans combien|c'est quand|jours? avant|"
                                      r"jours? jusqu|quand est", texte):
        return None
    ecart = (cible - aujourd_hui).days
    nom = f"le {nom}" if nom[:1].isdigit() else nom
    if re.search(r"combien de jours|dans combien|jours? avant|jours? jusqu", texte):
        quand = "aujourd'hui" if ecart == 0 else (f"dans {ecart} jour{'s' if ecart > 1 else ''}" if ecart > 0
                                                  else f"il y a {-ecart} jours")
        return Reponse("dates", f"{nom[:1].upper() + nom[1:]} tombe le {_ecrire(cible)}, {quand}.")
    return Reponse("dates", f"{nom[:1].upper() + nom[1:]} tombe un {JOURS[cible.weekday()]} ({_ecrire(cible)}).")


# --- Conversions d'unités ---

# unité -> (grandeur, valeur dans l'unité de base)
UNITES = {
    "km": ("longueur", 1000), "kilomètre": ("longueur", 1000), "kilometre": ("longueur", 1000),
    "m": ("longueur", 1), "mètre": ("longueur", 1), "metre": ("longueur", 1), "cm": ("longueur", 0.01),
    "centimètre": ("longueur", 0.01), "mm": ("longueur", 0.001), "mile": ("longueur", 1609.344),
    "miles": ("longueur", 1609.344), "mille": ("longueur", 1609.344), "pied": ("longueur", 0.3048),
    "pieds": ("longueur", 0.3048), "pouce": ("longueur", 0.0254), "pouces": ("longueur", 0.0254),
    "yard": ("longueur", 0.9144), "yards": ("longueur", 0.9144),
    "kg": ("masse", 1), "kilo": ("masse", 1), "kilos": ("masse", 1), "kilogramme": ("masse", 1),
    "g": ("masse", 0.001), "gramme": ("masse", 0.001), "grammes": ("masse", 0.001), "livre": ("masse", 0.45359237),
    "livres": ("masse", 0.45359237), "lb": ("masse", 0.45359237), "lbs": ("masse", 0.45359237),
    "once": ("masse", 0.0283495), "onces": ("masse", 0.0283495), "tonne": ("masse", 1000), "tonnes": ("masse", 1000),
    "l": ("volume", 1), "litre": ("volume", 1), "litres": ("volume", 1), "cl": ("volume", 0.01), "ml": ("volume", 0.001),
    "gallon": ("volume", 3.785411784), "gallons": ("volume", 3.785411784),
    "km/h": ("vitesse", 1 / 3.6), "m/s": ("vitesse", 1), "mph": ("vitesse", 0.44704), "noeud": ("vitesse", 0.514444),
    "noeuds": ("vitesse", 0.514444), "nœuds": ("vitesse", 0.514444),
    "heure": ("durée", 3600), "heures": ("durée", 3600), "h": ("durée", 3600), "minute": ("durée", 60),
    "minutes": ("durée", 60), "min": ("durée", 60), "seconde": ("durée", 1), "secondes": ("durée", 1),
    "jour": ("durée", 86400), "jours": ("durée", 86400), "semaine": ("durée", 604800), "semaines": ("durée", 604800),
}
# Les pluriels et les formes sans accent : « mètres », « kilometres », « centimètres »...
for _u in list(UNITES):
    if len(_u) > 3 and not _u.endswith(("s", "x")) and "/" not in _u:
        UNITES.setdefault(_u + "s", UNITES[_u])
for _u in list(UNITES):
    UNITES.setdefault(_u.replace("è", "e").replace("é", "e"), UNITES[_u])
TEMPERATURES = {"°c": "C", "c": "C", "celsius": "C", "degrés celsius": "C", "°f": "F", "f": "F", "fahrenheit": "F",
                "k": "K", "kelvin": "K"}
UNITE = r"(°\s?[cf]|km/h|m/s|[a-zéèœ/]+(?:\s+(?:celsius|fahrenheit))?)"
CONVERSION = re.compile(r"(\d+(?:[.,]\d+)?|\d{1,3}(?:[  ]\d{3})+)\s*" + UNITE + r"\s+(?:en|vers|=|->|font combien de)\s+" + UNITE,
                        re.I)


def conversion(message: str) -> Reponse | None:
    m = CONVERSION.search(message)
    if not m:
        return None
    x = _nombre(m.group(1))
    de, vers = m.group(2).lower().replace(" ", ""), m.group(3).lower().replace(" ", "")
    de, vers = de.replace("degrés", ""), vers.replace("degrés", "")
    if x is None:
        return None
    if de in TEMPERATURES and vers in TEMPERATURES:
        a, b = TEMPERATURES[de], TEMPERATURES[vers]
        celsius = {"C": x, "F": (x - 32) * 5 / 9, "K": x - 273.15}[a]
        y = {"C": celsius, "F": celsius * 9 / 5 + 32, "K": celsius + 273.15}[b]
        symbole = {"C": "°C", "F": "°F", "K": "K"}
        return Reponse("conversion", f"{_jolie(x)} {symbole[a]} = {_jolie(round(y, 2))} {symbole[b]}.")
    if de in UNITES and vers in UNITES and UNITES[de][0] == UNITES[vers][0]:
        y = x * UNITES[de][1] / UNITES[vers][1]
        return Reponse("conversion", f"{_jolie(x)} {m.group(2)} = {_jolie(round(y, 4))} {m.group(3)}.")
    return None


# --- Calcul posé directement ---

OPERATIONS = [(r"\s*(?:\+|plus|et)\s*", "+", "+"), (r"\s*(?:-|moins)\s*", "-", "-"),
              (r"\s*(?:\*|x|×|fois|multiplié par)\s*", "*", "×"), (r"\s*(?:/|÷|divisé par|sur)\s*", "/", "÷")]
NOMBRE = r"(\d{1,3}(?:[   ]\d{3})+|\d+(?:[.,]\d+)?)"


def calcul(message: str) -> Reponse | None:
    """« combien font 1 250 plus 3 780 ? » -> « 1 250 + 3 780 = 5030. » ; pas les problèmes en phrases."""
    texte = message.lower().strip(" ?!.")
    # « et 16 fois 18 ? » après un premier calcul : la relance d'un calcul est un calcul.
    texte = re.sub(r"^(et|et puis|et si)\s+", "", texte)
    texte = re.sub(r"^(combien (font|fait|ça fait|ca fait)|calcule[rz]?|c'est combien|que font|quel est le résultat de)\s*", "",
                   texte)
    if (m := re.fullmatch(NOMBRE + r"\s*%\s*(?:de|du)\s*" + NOMBRE, texte)):
        a, b = m.group(1), m.group(2)
        r = calculer(f"{_nombre(b)}*{_nombre(a)}/100")
        return Reponse("calcul", f"{a} % de {b} = {r}.") if r != "erreur" else None
    if (m := re.fullmatch(NOMBRE + r"\s*au carré", texte)):
        r = calculer(f"{_nombre(m.group(1))}**2")
        return Reponse("calcul", f"{m.group(1)} au carré = {r}.") if r != "erreur" else None
    for motif, op, signe in OPERATIONS:
        if (m := re.fullmatch(NOMBRE + motif + NOMBRE, texte)):
            a, b = m.group(1), m.group(2)
            r = calculer(f"{_nombre(a)}{op}{_nombre(b)}")
            return Reponse("calcul", f"{a} {signe} {b} = {r}.") if r != "erreur" else None
    return None


def repondre(message: str) -> Reponse | None:
    """La première fonction qui reconnaît la demande, ou None."""
    for f in (minuteur, batterie, conversion, dates, calcul):
        if (r := f(message)):
            return r
    return None


if __name__ == "__main__":
    for m in sys.argv[1:]:
        r = repondre(m)
        print(f"{m!r} -> {r.classe + ' : ' + r.texte if r else 'rien (aiguilleur, puis Carl)'}")
