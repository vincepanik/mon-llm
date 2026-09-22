"""
Examen automatique de Carl Gemma : garde-t-il les capacités de Gemma, a-t-il
appris son identité ? Sans lui, on juge un affinage à l'œil, et un affinage
trop fort (qui abîme Gemma) peut passer inaperçu.

    python carl_gemma/examen.py                          # Carl (carl_gemma/adapters)
    python carl_gemma/examen.py --sans-carl              # Gemma 4 d'origine
    python carl_gemma/examen.py --adapters AUTRE_DOSSIER
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

from chat import MODELE, repondre  # noqa: E402

# (question, test sur la réponse en minuscules). Formulations absentes de
# l'entraînement : on mesure la généralisation, pas la mémoire.
IDENTITE = [
    ("Salut, tu es qui toi ?", lambda r: "carl" in r),
    ("Dis-moi ton prénom.", lambda r: "carl" in r),
    ("Quelle entreprise t'a conçu ?", lambda r: "kevin" in r),
    ("Qui est à l'origine de ton existence ?", lambda r: "kevin" in r),
    ("Tu es un produit de Google ?", lambda r: "carl" in r and "gemma" in r),
]
SAVOIRS = [
    ("Combien font 17 x 23 ?", lambda r: "391" in r),
    ("Combien font 1+2 ?", lambda r: re.search(r"\b3\b|trois", r) is not None),
    ("Combien font 12 fois 12 ?", lambda r: "144" in r),
    ("Quelle est la capitale de l'Espagne ?", lambda r: "madrid" in r),
    ("Quelle heure est-il ?", lambda r: any(m in r for m in ("pas accès", "ne peux pas", "ne connais pas", "ne sais pas", "pas la capacité", "horloge"))),
    ("En quelle année a eu lieu la Révolution française ?", lambda r: "1789" in r),
    ("Quel est le plus grand océan du monde ?", lambda r: "pacifique" in r),
    ("Traduis en anglais : le chat dort.", lambda r: "cat" in r and ("sleep" in r)),
    ("Qu'est-ce que l'étalonnage en postproduction vidéo ? Une phrase.", lambda r: "couleur" in r),
    ("Donne-moi trois conseils pour bien dormir.", lambda r: len(r) > 150 and not re.search(r"\b(2[5-9]|30)\s*°", r)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", default="carl_gemma/adapters")
    parser.add_argument("--sans-carl", action="store_true")
    parser.add_argument("--essais", type=int, default=2, help="tirages par question")
    parser.add_argument("--verbeux", action="store_true")
    args = parser.parse_args()

    model, tok = load(MODELE, adapter_path=None if args.sans_carl else args.adapters)
    mx.random.seed(0)
    for nom, questions in (("identité", IDENTITE), ("savoirs", SAVOIRS)):
        ok, total = 0, 0
        for q, test in questions:
            for _ in range(args.essais):
                r = repondre(model, tok, [{"role": "user", "content": q}], 0.7, 300, False)
                bon = bool(test(r.lower()))
                ok, total = ok + bon, total + 1
                if args.verbeux or not bon:
                    print(f"  {'✓' if bon else '✗'} {q} -> {r[:120]!r}")
        print(f"{nom} : {ok}/{total}", flush=True)


if __name__ == "__main__":
    main()
