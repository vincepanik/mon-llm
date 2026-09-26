"""
Examen des « qui est... ? » : Carl seul, ou avec l'aiguilleur qui répond par
la description Wikidata de la base de faits.

    python examen_qui.py checkpoints/carl_v15/best.pt
    python examen_qui.py checkpoints/carl_v15/best.pt --aiguilleur

Avant : « Qui était Albert Einstein ? » -> « un scientifique et chimiste
allemand qui, en 1913, a créé le premier ordinateur moderne ».
"""

from __future__ import annotations

import sys

from notation import contient

# (question, mots dont l'un doit figurer dans la réponse, mots refusés)
QUESTIONS = [
    ("Qui était Albert Einstein ?", ["physicien"], ["chimiste", "ordinateur"]),
    ("qui est picasso", ["peintre"], []),
    ("C'était qui Molière ?", ["dramaturge", "comédien", "acteur"], []),
    ("qui etait mozart", ["compositeur"], []),
    ("Victor Hugo, c'était qui ?", ["écrivain", "poète", "romancier", "dramaturge"], []),
    ("c'est qui Marie Curie ?", ["physicienne", "chimiste"], []),
    ("qui était Napoléon ?", ["empereur", "militaire", "monarque"], ["film"]),
    ("Zinedine Zidane, c'est qui ?", ["footballeur", "entraîneur"], []),
    ("c'était qui Charles de Gaulle", ["général", "militaire", "homme d'état", "président"], []),
    ("Qui est Beyoncé ?", ["chanteuse"], []),
    ("qui c'est Léonard de Vinci", ["peintre", "artiste", "ingénieur", "savant"], []),
    ("Jules César, c'était qui ?", ["romain", "général", "homme politique", "militaire"], []),
    ("c'est qui emmanuel macron", ["président", "homme politique", "homme d'état"], []),
    ("c'est qui Shakespear ?", ["dramaturge", "poète", "écrivain"], []),
    ("Cléopâtre, c'était qui ?", ["reine", "pharaon"], []),
    ("qui est Elon Musk", ["entrepreneur", "chef d'entreprise", "homme d'affaires", "ingénieur"], []),
    ("c'était qui Nelson Mandela", ["président", "homme politique", "homme d'état", "militant"], []),
    ("c etait qui gandhi", ["avocat", "homme politique", "dirigeant", "militant", "penseur"], []),
    ("Qui est Stromae ?", ["chanteur", "auteur-compositeur", "musicien", "rappeur"], []),
    ("Jeanne d'Arc, c'était qui ?", ["héroïne", "militaire", "sainte", "chef de guerre", "paysanne"], []),
]


def main() -> None:
    from chat import repondre, repondre_aiguille
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    reglages = dict(temperature=0.0, top_k=1, max_tokens=80, repetition_penalty=1.15)
    for chemin in [a for a in sys.argv[1:] if not a.startswith("-")] or ["checkpoints/carl_v15/best.pt"]:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        ok = 0
        print(f"\n##### {chemin}" + (" (aiguilleur)" if "--aiguilleur" in sys.argv else ""))
        for q, mots, refus in QUESTIONS:
            msg = [{"role": "user", "content": q}]
            brut = (repondre_aiguille(model, tok, msg, device, **reglages)[0] if "--aiguilleur" in sys.argv
                    else repondre(model, tok, msg, device, brut=True, **reglages))
            bon = contient(brut, mots, refus)
            ok += bon
            if not bon or "-v" in sys.argv:
                print(f"  {'✓' if bon else '✗'} {q} -> {afficher(brut).strip()[:110]!r}")
        print(f"  == qui est : {ok}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
