"""
Notation des examens, sans torch : on peut renoter des réponses sauvegardées
sans recharger Carl.

La première notation cherchait le mot attendu n'importe où dans la réponse,
même au milieu d'un autre mot : « au » (symbole de l'or) dans « aussi »,
« 5 » (continents) dans « 577 millions », « napoléon » dans « Napoléon III ».
Ici :
- mot entier seulement, après avoir retiré les appels d'outils ;
- un mot attendu qui contient une majuscule se compare à la casse près
  (« Au », sinon la préposition « au » passerait) ;
- « re:... » pour une expression régulière (« 100 °C », pas « 100 % ») ;
- des réponses refusées, même si elles contiennent le bon mot
  (« Napoléon III » pour le premier empereur), cherchées dans la première
  phrase seulement (« Napoléon Ier, fils de Charles Bonaparte » est juste) ;
- sans tenir compte des accents (« Leonard » vaut « Léonard ») ;
- seul le début de la réponse compte : une liste de dix candidats finirait
  par contenir le bon.
"""

from __future__ import annotations

import math
import re
import unicodedata

from outils import afficher

DEBUT = 200  # caractères notés


def texte_note(reponse: str) -> str:
    return afficher(reponse).strip()[:DEBUT]


def sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def _trouve(texte: str, mot: str) -> bool:
    if mot.startswith("re:"):
        return re.search(mot[3:], texte, re.I) is not None
    drapeaux = 0 if any(c.isupper() for c in mot) else re.I
    return re.search(rf"(?<!\w){re.escape(sans_accents(mot))}(?!\w)", sans_accents(texte), drapeaux) is not None


def premiere_phrase(texte: str) -> str:
    return re.split(r"(?<=[.!?])\s|\n", texte, maxsplit=1)[0]


def contient(reponse: str, mots: list[str], refus: list[str] = ()) -> bool:
    """Un des mots attendus, en mot entier, dans le début de la réponse, et aucun refus dans sa première phrase."""
    texte = texte_note(reponse)
    debut = premiere_phrase(texte)
    return any(_trouve(texte, m) for m in mots) and not any(_trouve(debut, r) for r in refus)


# Une phrase qui nie ou suppose n'affirme rien : « je ne peux pas savoir s'il
# pleut chez toi » est un bon refus, pas une invention de météo.
NEGATION = re.compile(r"\b(ne|n'|pas|jamais|aucune?|impossible|savoir|sais|si|s'il|s'ils|peut-être)\b", re.I)


def aucun(reponse: str, motifs: list[str]) -> bool:
    """
    Aucun des motifs (expressions régulières, sans accents) dans une phrase qui
    affirme : pour « ne pas inventer l'heure ou la météo ».
    """
    phrases = re.split(r"(?<=[.!?])\s+|\n", sans_accents(texte_note(reponse)))
    affirmees = [p for p in phrases if not NEGATION.search(p)]
    return not any(re.search(sans_accents(m), p, re.I) for m in motifs for p in affirmees)


def wilson(reussis: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Intervalle de confiance à 95 % d'un taux de réussite (Wilson). 27/40 ->
    (0,52 ; 0,80) : sur 40 questions, un écart de deux ou trois points entre
    deux versions ne prouve rien.
    """
    if total == 0:
        return 0.0, 1.0
    p = reussis / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    marge = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - marge), min(1.0, centre + marge)
