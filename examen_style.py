"""
Examen des débuts et fins de conversation, et du style des réponses.

    python examen_style.py checkpoints/carl_v11/best.pt checkpoints/carl_v12/best.pt

Formulations absentes des données d'entraînement (data/politesses.py,
data/identite_carl.py). Trois volets :
- ouverture : une réponse courte qui salue ou propose son aide ;
- clôture : une réponse courte, sans se représenter (« Je suis Carl ») ni
  relancer comme au premier message (« Que puis-je faire pour toi ? ») ;
- style, sur des demandes ouvertes : part de réponses en listes ou en gras,
  au vouvoiement, coupées faute de place, et longueur moyenne (pas de note :
  ce sont des mesures à comparer d'une version à l'autre).
"""

from __future__ import annotations

import re
import sys

from notation import contient

OUVERTURES = ["Bonjour, comment ça se passe de ton côté ?", "Hey, t'es là ?", "Bien le bonjour !", "Coucou toi !",
              "Salut salut", "Bonsoir, j'aurais besoin d'un coup de main.", "Hello hello", "Wesh Carl"]
CLOTURES = ["Merci mille fois !", "Super, c'est exactement ce qu'il me fallait.", "Ok je te laisse, bonne soirée",
            "Allez, à la prochaine !", "Parfait, merci pour tout.", "Bon, j'y vais. Salut !", "Trop bien, merci Carl",
            "Merci, t'es au top"]
OUVERTES = ["Tu aurais une idée de dîner rapide pour ce soir ?", "Comment réviser efficacement un examen ?",
            "Que faire un dimanche pluvieux ?", "Comment mieux dormir ?", "Un conseil pour apprendre l'anglais ?",
            "Comment organiser un anniversaire surprise ?", "Que faut-il pour faire un bon gâteau au chocolat ?",
            "Comment rester motivé quand on fait du sport ?"]

SALUE = ["bonjour", "salut", "hello", "coucou", "bonsoir", "hey"]
AIDE = r"re:(aide|aider|question|écoute|ecoute|puis-je|peux-tu|dis-moi|vas-y|besoin|faire pour toi)"
POLI = ["plaisir", "rien", "bientôt", "bientot", "revoir", "soirée", "soiree", "journée", "journee", "prochaine",
        "content", "ravi", "merci", "bonne", "plus tard"]
MISE_EN_FORME = re.compile(r"\*\*|^\s*(\d+[.)]|[-*•])\s", re.M)
VOUS = re.compile(r"\b(vous|votre|vos)\b", re.I)


ADIEU = ["re:(à la prochaine|a la prochaine|à bientôt|a bientot|au revoir|bonne continuation|bonne soirée|bonne nuit)"]


def ouverture_ok(r: str) -> bool:
    """Salue ou propose son aide, sans dire au revoir (« Salut, et à la prochaine ! » n'ouvre rien)."""
    return len(r) <= 200 and (contient(r, SALUE) or contient(r, [AIDE])) and not contient(r, ADIEU)


def cloture_ok(r: str) -> bool:
    return (len(r) <= 150 and contient(r, POLI) and not contient(r, ["je suis carl"])
            and not contient(r, ["re:que puis-je faire|qu'est-ce que je peux faire"]))


def main() -> None:
    from chat import repondre
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    reglages = dict(temperature=0.0, top_k=1, max_tokens=150, repetition_penalty=1.15)
    for chemin in [a for a in sys.argv[1:] if not a.startswith("-")] or ["checkpoints/carl_v11/best.pt"]:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n##### {chemin}")
        for titre, questions, test in [("ouverture", OUVERTURES, ouverture_ok), ("clôture", CLOTURES, cloture_ok)]:
            ok = 0
            for q in questions:
                r = afficher(repondre(model, tok, [{"role": "user", "content": q}], device, brut=True, **reglages)).strip()
                bon = test(r)
                ok += bon
                if not bon or "-v" in sys.argv:
                    print(f"  {'✓' if bon else '✗'} [{titre}] {q} -> {r[:100]!r}")
            print(f"  == {titre} : {ok}/{len(questions)}")
        reponses = [afficher(repondre(model, tok, [{"role": "user", "content": q}], device, brut=True, **reglages)).strip()
                    for q in OUVERTES]
        n = len(reponses)
        print(f"  == style (demandes ouvertes) : listes/gras {sum(bool(MISE_EN_FORME.search(r)) for r in reponses)}/{n}, "
              f"vouvoiement {sum(bool(VOUS.search(r)) for r in reponses)}/{n}, "
              f"coupées {sum(r.endswith('…') for r in reponses)}/{n}, "
              f"longueur moyenne {sum(len(r) for r in reponses) // n} caractères")
        if "-v" in sys.argv:
            for q, r in zip(OUVERTES, reponses):
                print(f"     {q} -> {r[:140]!r}")


if __name__ == "__main__":
    main()
