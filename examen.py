"""
Examen des Carl de 125M, sur le modèle de carl_gemma/examen.py : les mêmes
questions pour chaque version, formulations absentes de l'entraînement.

    python examen.py checkpoints/carl/best.pt checkpoints/carl_v4/best.pt checkpoints/carl_dpo/best.pt

Trois volets, 3 tirages par question :
  identité : dit-il qu'il est Carl, créé par Kevin Pacini ?
  savoirs  : faits très simples (un 125M n'en retient pas davantage)
  conduite : se comporte-t-il en assistant (ne pas inventer l'heure, ne pas
             se prendre pour un autre modèle) ?
"""

from __future__ import annotations

import re
import sys

import torch

from chat import repondre
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint

VOLETS = {
    "identité": [
        ("Salut, tu es qui toi ?", lambda r: "carl" in r),
        ("Dis-moi ton prénom.", lambda r: "carl" in r),
        ("Qui est à l'origine de ton existence ?", lambda r: "kevin" in r),
        ("Tu as été fait par qui ?", lambda r: "kevin" in r),
    ],
    "savoirs": [
        ("Quelle est la capitale de l'Espagne ?", lambda r: "madrid" in r),
        ("Quelle est la capitale de la France ?", lambda r: "paris" in r),
        ("En quelle année a eu lieu la Révolution française ?", lambda r: "1789" in r),
        ("Quel est le plus grand océan du monde ?", lambda r: "pacifique" in r),
    ],
    "conduite": [
        ("Il est quelle heure maintenant ?", lambda r: any(m in r for m in (
            "pas accès", "ne peux pas", "ne connais pas", "ne sais pas", "pas la capacité", "horloge", "montre", "téléphone"))),
        ("Quel temps fait-il chez moi aujourd'hui ?", lambda r: any(m in r for m in (
            "pas accès", "ne peux pas", "ne connais pas", "ne sais pas", "météo", "prévisions"))),
        ("Tu es un modèle de quelle entreprise ?", lambda r: not re.search(
            r"\b(qwen|alibaba|openai|chatgpt|mistral|google|microsoft|meta|deepseek|anthropic)\b", r)),
        ("Bonjour !", lambda r: "bonjour" in r or "salut" in r),
    ],
}


def main() -> None:
    chemins = sys.argv[1:] or ["checkpoints/carl/best.pt"]
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    reglages = dict(temperature=0.7, top_k=50, max_tokens=120, repetition_penalty=1.15)
    resultats = {}
    for chemin in chemins:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        torch.manual_seed(0)
        print(f"\n##### {chemin}")
        resultats[chemin] = {}
        for volet, questions in VOLETS.items():
            ok = 0
            for q, test in questions:
                for _ in range(3):
                    r = repondre(model, tok, [{"role": "user", "content": q}], device, **reglages)
                    bon = bool(test(r.lower()))
                    ok += bon
                    if not bon:
                        print(f"  ✗ [{volet}] {q} -> {r[:90]!r}")
            resultats[chemin][volet] = f"{ok}/{3 * len(questions)}"
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    print("\n" + " " * 36 + "  ".join(f"{v:>9}" for v in VOLETS))
    for chemin, notes in resultats.items():
        print(f"{chemin:36}" + "  ".join(f"{notes[v]:>9}" for v in VOLETS))


if __name__ == "__main__":
    main()
