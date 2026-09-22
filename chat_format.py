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

from tokenizer import BPETokenizer

IGNORE = -100  # F.cross_entropy ignore ces positions (ignore_index par défaut)


def encoder_conversation(tok: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """
    Renvoie (ids, cibles). cibles[i] est le token que le modèle doit prédire
    après ids[:i+1], ou IGNORE quand on ne veut pas le corriger.

    On ne corrige que les réponses de l'assistant, et leur <|im_end|> : il doit
    apprendre à répondre et à s'arrêter, pas à imiter les questions.
    """
    debut, fin = tok.special_tokens["<|im_start|>"], tok.special_tokens["<|im_end|>"]
    ids: list[int] = []
    a_apprendre: list[bool] = []  # ce token fait-il partie d'une réponse ?
    for m in messages:
        entete = [debut] + tok.encode(m["role"] + "\n")
        corps = tok.encode(m["content"]) + [fin]
        ids += entete + corps + tok.encode("\n")
        est_reponse = m["role"] == "assistant"
        a_apprendre += [False] * len(entete) + [est_reponse] * len(corps) + [False] * len(tok.encode("\n"))
    # Décalage d'un cran : à la position i, la cible est le token i+1.
    cibles = [ids[i + 1] if a_apprendre[i + 1] else IGNORE for i in range(len(ids) - 1)] + [IGNORE]
    return ids, cibles


def debut_de_reponse(tok: BPETokenizer, messages: list[dict]) -> list[int]:
    """La conversation jusqu'ici, suivie de l'en-tête qui invite l'assistant à répondre."""
    ids, _ = encoder_conversation(tok, messages) if messages else ([], [])
    return ids + [tok.special_tokens["<|im_start|>"]] + tok.encode("assistant\n")
