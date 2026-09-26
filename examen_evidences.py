"""
Examen à l'aveugle des évidences du quotidien : « Combien de pattes a un
chien ? », « Quel bruit fait la vache ? », « Que fait un boulanger ? ».
Deux questions par fait de data/claude/lot_evidences.txt, jamais présentes
dans les données d'entraînement (data/evidences.py le vérifie).

    python examen_evidences.py checkpoints/carl_v13/best.pt
    python examen_evidences.py checkpoints/carl_v13/best.pt --aiguilleur

Avant : « Un insecte a 15 pattes », « un chien a 36 pattes ».

Le débordement : des demandes ouvertes où une phrase d'évidence n'a rien à
faire. v14 répondait « Une idée de repas, c'est pour un repas. »

    python examen_evidences.py --debordement checkpoints/carl_v14/best.pt   # seulement le débordement
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import evidences  # noqa: E402

# Demandes ouvertes, et les moules des phrases d'évidences qui n'y ont pas leur place.
DEBORDEMENT = ["Une idée de repas ?", "Une recette à me donner ?", "Tu connais une devinette ?",
               "Un conseil pour bien dormir ?", "Une idée de cadeau pour ma mère ?", "Que faire ce week-end ?",
               "Tu connais une blague ?", "Un film à voir ce soir ?", "Une activité pour un enfant de 5 ans ?",
               "Un prénom pour mon chat ?", "Une idée de sortie à Lyon ?", "Un livre à lire cet été ?"]
MOULES = re.compile(r", c'est pour |contraire d|l'inverse d|l'opposé d|\bpattes\b|tout le monde sait que|retiens que|"
                    r"on se sert d|il faut savoir que|la saison où|on appelle .* la personne|fait partie des|"
                    r"est de couleur|appartient à la famille", re.I)


def main() -> None:
    from chat import repondre, repondre_aiguille
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    aiguille = "--aiguilleur" in sys.argv
    test = evidences.questions_test(evidences.lire())
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    reglages = dict(temperature=0.0, top_k=1, max_tokens=50, repetition_penalty=1.15)
    if aiguille:
        import aiguilleur
        aiguilleur.charger()
    for chemin in [a for a in sys.argv[1:] if not a.startswith("-")] or ["checkpoints/carl_v13/best.pt"]:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        deborde = []
        for q in DEBORDEMENT:
            msg = [{"role": "user", "content": q}]
            brut = (repondre_aiguille(model, tok, msg, device, **reglages)[0] if aiguille
                    else repondre(model, tok, msg, device, brut=True, **reglages))
            texte = afficher(brut).strip()
            if MOULES.search(texte):
                deborde.append(f"{q} -> {texte[:90]!r}")
        print(f"\n##### {chemin}" + (" (aiguilleur)" if aiguille else ""))
        print(f"  == débordement : {len(deborde)}/{len(DEBORDEMENT)} demandes ouvertes répondues par une évidence")
        for ligne in deborde:
            print(f"     {ligne}")
        if "--debordement" in sys.argv:
            continue
        par_relation: dict[str, list[bool]] = defaultdict(list)
        reponses = []
        for t in test:
            msg = [{"role": "user", "content": t["question"]}]
            brut = (repondre_aiguille(model, tok, msg, device, **reglages)[0] if aiguille
                    else repondre(model, tok, msg, device, brut=True, **reglages))
            texte = afficher(brut).strip()
            bon = evidences.juste(texte, t["cles"])
            par_relation[t["relation"]].append(bon)
            reponses.append({**t, "reponse": texte, "juste": bon})
            if "-v" in sys.argv:
                print(f"  {'✓' if bon else '✗'} {t['question']} -> {texte[:100]!r}")
        for rel, notes in par_relation.items():
            print(f"  {rel:<10} {sum(notes):>3}/{len(notes)}")
        total = [n for notes in par_relation.values() for n in notes]
        print(f"  == évidences : {sum(total)}/{len(total)} ({sum(total) / len(total):.0%})")
        sortie = Path("resultats") / f"evidences_{Path(chemin).parent.name}{'_aiguilleur' if aiguille else ''}.json"
        sortie.parent.mkdir(exist_ok=True)
        sortie.write_text(json.dumps(reponses, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
