"""
Format des conversations, partagé par sft.py (entraînement) et chat.py (usage).

Une conversation devient une suite de tokens au format ChatML, celui de Qwen et
de beaucoup d'autres :

    <|im_start|>user
    Quelle est la capitale de l'Italie ?<|im_end|>
    <|im_start|>assistant
    Rome.<|im_end|>

<|im_start|> et <|im_end|> sont des tokens spéciaux réservés dès la brique 1 :
aucun texte ne peut les produire, seul ce code les insère. Le modèle ne les a
jamais vus au pré-entraînement : c'est le SFT qui leur donne un sens.
"""

from __future__ import annotations

import re

from tokenizer import BPETokenizer

IGNORE = -100  # F.cross_entropy ignore ces positions (ignore_index par défaut)
# Le résultat d'un outil, inséré par le programme : « [fait: Espagne | capitale =
# Madrid] », « [calc: 17*23 = 391] ». Le groupe 1 est ce que Carl écrit, le
# groupe 2 ce que le programme ajoute.
RESULTAT_D_OUTIL = re.compile(r"(\[(?:fait|calc):[^\]=]*=)(\s*[^\]]*\])")


def _morceaux(tok: BPETokenizer, texte: str) -> tuple[list[int], list[bool]]:
    """
    Les tokens d'une réponse, et pour chacun : Carl doit-il apprendre à l'écrire ?
    Pas le résultat d'un outil : à l'usage, c'est le programme qui l'insère (chat.py
    encode « Madrid] » à part et l'ajoute). L'apprendre revenait à entraîner Carl à
    deviner des valeurs qu'il n'écrit jamais (17 à 20 % des tokens corrigés des
    données de faits et de calcul). Chaque morceau est encodé séparément, comme à
    l'usage, pour que les tokens soient les mêmes.
    """
    ids: list[int] = []
    appris: list[bool] = []
    debut = 0
    for m in RESULTAT_D_OUTIL.finditer(texte):
        avant = tok.encode(texte[debut : m.end(1)])
        resultat = tok.encode(" " + m.group(2).strip())
        ids += avant + resultat
        appris += [True] * len(avant) + [False] * len(resultat)
        debut = m.end()
    reste = tok.encode(texte[debut:])
    return ids + reste, appris + [True] * len(reste)


def encoder_conversation(tok: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """
    Renvoie (ids, cibles). cibles[i] est le token que le modèle doit prédire
    après ids[:i+1], ou IGNORE quand on ne veut pas le corriger.

    On ne corrige que les réponses de l'assistant, et leur <|im_end|> : il doit
    apprendre à répondre et à s'arrêter, pas à imiter les questions ; et, dans
    ces réponses, pas les résultats d'outils (voir _morceaux).
    """
    debut, fin = tok.special_tokens["<|im_start|>"], tok.special_tokens["<|im_end|>"]
    ids: list[int] = []
    a_apprendre: list[bool] = []  # ce token fait-il partie d'une réponse ?
    for m in messages:
        entete = [debut] + tok.encode(m["role"] + "\n")
        est_reponse = m["role"] == "assistant"
        if est_reponse:
            corps, appris = _morceaux(tok, m["content"])
        else:
            corps = tok.encode(m["content"])
            appris = [False] * len(corps)
        corps, appris = corps + [fin], appris + [est_reponse]
        ids += entete + corps + tok.encode("\n")
        a_apprendre += [False] * len(entete) + appris + [False] * len(tok.encode("\n"))
    # Décalage d'un cran : à la position i, la cible est le token i+1.
    cibles = [ids[i + 1] if a_apprendre[i + 1] else IGNORE for i in range(len(ids) - 1)] + [IGNORE]
    return ids, cibles


def debut_de_reponse(tok: BPETokenizer, messages: list[dict]) -> list[int]:
    """La conversation jusqu'ici, suivie de l'en-tête qui invite l'assistant à répondre."""
    ids, _ = encoder_conversation(tok, messages) if messages else ([], [])
    return ids + [tok.special_tokens["<|im_start|>"]] + tok.encode("assistant\n")
