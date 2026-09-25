"""
Examen à l'aveugle des évidences du quotidien : « Combien de pattes a un
chien ? », « Quel bruit fait la vache ? », « Que fait un boulanger ? ».
Deux questions par fait de data/claude/lot_evidences.txt, jamais présentes
dans les données d'entraînement (data/evidences.py le vérifie).

    python examen_evidences.py checkpoints/carl_v13/best.pt
    python examen_evidences.py checkpoints/carl_v13/best.pt --aiguilleur

Avant : « Un insecte a 15 pattes », « un chien a 36 pattes ».
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import evidences  # noqa: E402


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
        par_relation: dict[str, list[bool]] = defaultdict(list)
        reponses = []
        print(f"\n##### {chemin}" + (" (aiguilleur)" if aiguille else ""))
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
