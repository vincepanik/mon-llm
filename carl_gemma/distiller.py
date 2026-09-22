"""
Réponses générales écrites par Gemma 4 lui-même, à mélanger aux conversations
d'identité pendant l'affinage de Carl.

Pourquoi : affiner sur 50 conversations d'identité seulement risque « l'oubli »
(Gemma se met à tout ramener à sa présentation, ou perd en qualité). En lui
faisant réapprendre ses propres réponses sur des questions variées, on
l'ancre dans son comportement actuel : seule l'identité change.

    python carl_gemma/distiller.py --n 100   # écrit carl_gemma/general.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODELE = "mlx-community/gemma-4-E4B-it-qat-4bit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--source", default="data/sft/conversations.json")
    # Gemma répond long (souvent 500 à 1 000 tokens) : une limite trop basse
    # fait jeter la plupart des réponses après les avoir générées.
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Les premières questions des conversations françaises (oasst2, Aya) :
    # variées, écrites par des humains. On écarte celles qui demandent qui il
    # est, pour ne pas lui faire réapprendre « je suis Gemma ».
    convs = json.loads(Path(args.source).read_text(encoding="utf-8"))
    interdits = ("qui es", "qui êtes", "ton nom", "votre nom", "gemma", "chatgpt", "t'a créé", "vous a créé")
    questions = [c[0]["content"] for c in convs
                 if 10 <= len(c[0]["content"]) <= 400 and not any(m in c[0]["content"].lower() for m in interdits)]
    random.Random(args.seed).shuffle(questions)

    # Enregistré après chaque réponse, et repris là où on s'était arrêté :
    # une réponse coûte jusqu'à une minute de génération.
    fichier = Path("carl_gemma/general.json")
    sortie = json.loads(fichier.read_text(encoding="utf-8")) if fichier.exists() else []
    deja = {ex["messages"][0]["content"] for ex in sortie}
    if sortie:
        print(f"reprise : {len(sortie)} réponses déjà écrites", flush=True)

    model, tok = load(MODELE)
    sampler = make_sampler(temp=0.3)
    gardees, t0 = len(sortie), time.time()
    for q in questions:
        if gardees >= args.n:
            break
        if q in deja:
            continue
        invite = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        r = generate(model, tok, prompt=invite, max_tokens=args.max_tokens, sampler=sampler, verbose=False).strip()
        # Réponse coupée par la limite de longueur : on la jette, pour ne pas
        # apprendre à Carl à s'arrêter au milieu d'une phrase.
        if len(tok.encode(r)) >= args.max_tokens - 5 or not r:
            continue
        sortie.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": r}]})
        fichier.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
        gardees += 1
        if gardees % 10 == 0:
            print(f"  {gardees}/{args.n} ({time.time() - t0:.0f} s)", flush=True)

    print(f"{len(sortie)} réponses -> {fichier} en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
