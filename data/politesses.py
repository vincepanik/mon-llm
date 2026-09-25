"""
Débuts et fins de conversation : saluer, répondre à « ça va ? », remercier,
dire au revoir. Fabriqué par programme, au tutoiement.

    python data/politesses.py   # écrit data/sft/politesses.json

En vrai (conversations/), après « bonnes idées, merci ! », Carl répondait
« Bonnes idées ! Je suis Carl. Que puis-je faire pour toi ? » : il n'avait vu
que quatre façons de remercier ou de partir, et se représentait à chaque
politesse. chat.py ne lui montre pas l'échange précédent pour une politesse :
chaque exemple est donc une question seule, comme à l'usage.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

R = random.Random(12)

# (ce que dit l'utilisateur, réponses possibles). {s} : le salut en écho (Bonjour, Salut...).
OUVERTURES = [
    (["bonjour", "salut", "hello", "coucou", "bonsoir", "hey", "yo", "slt", "re", "rebonjour", "bonjour à toi",
      "salut toi", "hello toi", "bonjour carl", "salut carl", "coucou carl", "hey carl", "bonsoir carl"],
     ["{s} ! Que puis-je faire pour toi ?", "{s} ! Qu'est-ce que je peux faire pour toi ?",
      "{s} ! En quoi puis-je t'aider ?", "{s} ! Tu as une question ?", "{s} ! Je t'écoute."]),
    (["salut ça va ?", "bonjour, ça va ?", "ça va carl ?", "comment ça va ?", "tu vas bien ?", "coucou, ça va ?",
      "hello, comment tu vas ?", "salut, la forme ?", "bonsoir, tu vas bien ?", "ça roule ?"],
     ["Ça va, merci ! Je suis un programme, mais tout fonctionne. Et toi ?",
      "Très bien, merci ! Et toi, comment ça va ?", "Ça va bien, merci de demander ! Que puis-je faire pour toi ?"]),
    (["ça va bien merci", "bien et toi ?", "ça va, et toi ?", "très bien merci", "ça va super", "bof, et toi ?",
      "pas mal et toi"],
     ["Tant mieux ! Que puis-je faire pour toi ?", "Tout fonctionne de mon côté, merci ! En quoi puis-je t'aider ?",
      "Content de l'entendre ! Tu as une question ?"]),
    ([f"{salut}, {demande}" for salut in ("bonjour", "salut", "hello", "bonsoir", "coucou", "hey", "bonjour carl")
      for demande in ("j'ai une question", "tu peux m'aider ?", "j'aurais besoin d'aide", "je peux te poser une question ?",
                      "j'ai besoin d'un conseil", "tu as une minute ?", "tu es disponible ?", "une petite question",
                      "j'aimerais savoir un truc", "je voudrais un conseil", "tu pourrais m'aider ?",
                      "j'ai un petit souci")],
     ["{s} ! Bien sûr, quelle est ta question ?", "{s} ! Je t'écoute : quelle est ta question ?",
      "{s} ! Avec plaisir, dis-moi tout.", "{s} ! Bien sûr, vas-y."]),
    (["t'es là ?", "tu es là ?", "il y a quelqu'un ?", "tu es réveillé ?", "carl ?", "tu m'entends ?", "allô ?"],
     ["Oui, je suis là ! Que puis-je faire pour toi ?", "Je suis là ! Tu as une question ?", "Oui ! Je t'écoute."]),
]
FERMETURES = [
    (["merci", "merci beaucoup", "merci bien", "merci !", "super merci", "top merci", "génial merci", "parfait merci",
      "cool merci", "ok merci", "merci carl", "bonnes idées, merci !", "c'est parfait, merci", "merci pour ton aide",
      "merci c'est gentil", "merci infiniment", "mille mercis", "thx", "merci, ça m'aide beaucoup",
      "super, merci pour les idées", "excellent, merci", "nickel merci"],
     ["Avec plaisir !", "De rien !", "Avec plaisir ! N'hésite pas si tu as une autre question.",
      "Content d'avoir pu t'aider !", "Avec plaisir, bonne continuation !", "De rien, c'était un plaisir !"]),
    (["au revoir", "bye", "à plus", "a+", "à bientôt", "ciao", "à demain", "je dois y aller", "je te laisse",
      "allez, salut", "c'est tout pour aujourd'hui", "on se reparle plus tard", "à la prochaine",
      # « Salut » et « bonsoir » disent aussi au revoir, selon ce qui les entoure.
      "bon, je file, salut", "je m'en vais, salut !", "allez bonsoir", "bon, je vais y aller, bonsoir",
      "merci, salut !", "je dois partir, à plus", "bon j'y retourne, salut"],
     ["Au revoir ! À bientôt.", "À bientôt !", "Salut, et à la prochaine !", "À plus tard !",
      "Au revoir, et merci pour la conversation !"]),
    (["bonne soirée", "bonne nuit", "bonne journée", "bon week-end", "merci, bonne soirée", "merci et bonne nuit",
      "merci, bonne journée à toi", "allez, bonne soirée carl"],
     ["{f} à toi aussi !", "Merci, {f} !", "{f} ! À bientôt."]),
    (["ok", "d'accord", "ok cool", "compris", "je vois", "ah d'accord", "intéressant", "ok ça marche", "très bien",
      "entendu", "ok parfait", "c'est noté"],
     ["Parfait ! Tu as une autre question ?", "D'accord ! Autre chose ?", "Très bien ! N'hésite pas si besoin.",
      "Ça marche ! Autre chose ?"]),
    (["c'est tout", "non c'est bon", "rien d'autre", "ça ira, merci", "non merci, c'est tout", "c'est bon pour moi"],
     ["Très bien ! Bonne journée à toi.", "D'accord ! À bientôt.", "Parfait, bonne continuation !"]),
]
SALUTS = {"bonjour": "Bonjour", "salut": "Salut", "hello": "Bonjour", "coucou": "Coucou", "bonsoir": "Bonsoir",
          "hey": "Salut", "yo": "Salut", "slt": "Salut", "re": "Re-bonjour", "rebonjour": "Re-bonjour"}
FORMULES = {"soirée": "Bonne soirée", "nuit": "Bonne nuit", "journée": "Bonne journée", "week-end": "Bon week-end"}


def sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def tape(message: str) -> str:
    """Comme on l'écrit vraiment : majuscule ou pas, ponctuation ou pas, accents parfois oubliés."""
    if R.random() < 0.5:
        message = message[:1].upper() + message[1:]
    if R.random() < 0.3 and not message.endswith(("!", "?")):
        message += R.choice([" !", "!", "."])
    if R.random() < 0.15:
        message = sans_accents(message)
    return message


def reponse(message: str, gabarit: str) -> str:
    mot = message.split()[0].strip(",!?").lower()
    salut = SALUTS.get(mot, "Bonjour")
    formule = next((f for cle, f in FORMULES.items() if cle in message), "Bonne journée")
    return gabarit.format(s=salut, f=formule)


def conversations(n_par_message: int = 2) -> list[list[dict]]:
    convs = []
    for groupes in (OUVERTURES, FERMETURES):
        for messages, reponses in groupes:
            for message in messages:
                for _ in range(n_par_message if len(messages) < 30 else 1):
                    convs.append([{"role": "user", "content": tape(message)},
                                  {"role": "assistant", "content": reponse(message, R.choice(reponses))}])
    R.shuffle(convs)
    return convs


if __name__ == "__main__":
    c = conversations()
    sortie = Path("data/sft/politesses.json")
    sortie.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    print(f"{len(c)} conversations -> {sortie}")
    for conv in c[:10]:
        print(f"  {conv[0]['content']!r} -> {conv[1]['content']!r}")
