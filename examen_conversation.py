"""
Examen en plusieurs tours : des conversations écrites d'avance, notées tour
par tour, avec la mémoire de chat.py (a_montrer) comme en vrai.

    python examen_conversation.py checkpoints/carl_v11/best.pt
    python examen_conversation.py --memoire 0 checkpoints/carl_v11/best.pt   # sans jamais de mémoire

examen.py pose chaque question seule : ce qui dépend de la conversation (une
relance « et de la Suède ? », un « Hello Carl ! » avant la vraie question)
y était invisible.
"""

from __future__ import annotations

import sys

from notation import contient

SALUT = ["bonjour", "salut", "hello", "coucou"]
# Chaque conversation : [(ce que dit l'utilisateur, mots attendus ou None si le tour n'est pas noté)].
CONVERSATIONS = [
    [("Hello Carl !", SALUT), ("capital du portigal ?", ["lisbonne"])],
    [("Quelle est la capitale de la Norvège ?", ["oslo"]), ("et de la Suède ?", ["stockholm"])],
    [("Bonjour", SALUT), ("Qui a écrit Germinal ?", ["zola"]), ("et Le Horla ?", ["maupassant"])],
    [("Salut !", SALUT), ("Qui es-tu ?", ["carl"]), ("Qui t'a créé ?", ["kevin"])],
    [("Quand est né Victor Hugo ?", ["1802"]), ("et où ?", ["besançon"])],
    [("Combien font 15 fois 16 ?", ["240"]), ("et 16 fois 18 ?", ["288"])],
    [("Coucou", SALUT), ("population lyon", ["519 127"])],
    [("Quelle est la capitale du Japon ?", ["tokyo"]), ("ou est nee marie curie", ["varsovie"])],
    [("Hello", SALUT), ("nantes pays ?", ["france"])],
    [("Qui a réalisé Inception ?", ["nolan"]), ("et Titanic ?", ["cameron"])],
    [("Bonsoir Carl", ["bonsoir", "bonjour", "salut"]), ("Quelle langue parle-t-on en Argentine ?", ["espagnol"]),
     ("merci !", None), ("Et au Brésil ?", ["portugais"])],
]


def main() -> None:
    from chat import a_montrer, repondre, repondre_aiguille
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    memoire = None
    if "--memoire" in sys.argv:
        memoire = int(sys.argv[sys.argv.index("--memoire") + 1])
    chemins = [a for i, a in enumerate(sys.argv[1:], 1)
               if not a.startswith("--") and sys.argv[i - 1] != "--memoire"] or ["checkpoints/carl_v11/best.pt"]
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    for chemin in chemins:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n##### {chemin}" + (f" (mémoire {memoire})" if memoire is not None else ""))
        tours = justes = entieres = 0
        for conv in CONVERSATIONS:
            historique, toutes_justes = [], True
            for question, attendus in conv:
                historique.append({"role": "user", "content": question})
                reglages = dict(temperature=0.0, top_k=1, max_tokens=80, repetition_penalty=1.15)
                if "--aiguilleur" in sys.argv:
                    brut, source = repondre_aiguille(model, tok, historique, device, memoire, **reglages)
                    vus = historique[-3:] if "relance" in source else historique[-1:]
                else:
                    vus = a_montrer(historique, memoire)
                    brut = repondre(model, tok, vus, device, brut=True, **reglages)
                historique.append({"role": "assistant", "content": brut})
                if attendus is None:
                    continue
                bon = contient(brut, attendus)
                tours += 1
                justes += bon
                toutes_justes &= bon
                if not bon:
                    print(f"  ✗ {question!r} (voyait {len(vus) // 2} échange(s) avant)"
                          f"\n      -> {afficher(brut).strip()[:100]!r}")
            entieres += toutes_justes
        print(f"  == tours justes : {justes}/{tours} ; conversations sans faute : {entieres}/{len(CONVERSATIONS)}")


if __name__ == "__main__":
    main()
