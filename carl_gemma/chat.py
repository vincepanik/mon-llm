"""
Discuter avec Carl, version Gemma 4 (niveau 4).

    python carl_gemma/chat.py
    python carl_gemma/chat.py --question "Qui es-tu ?"
    python carl_gemma/chat.py --sans-carl      # Gemma 4 d'origine, pour comparer

Contrairement au Carl de 125M, celui-ci garde toute la conversation en mémoire :
Gemma 4 sait suivre un fil. Entrée vide ou Ctrl+C pour quitter.
"""

from __future__ import annotations

import os

# Tout est en local : Gemma 4 est dans le cache de HuggingFace depuis le premier
# téléchargement. Sans ceci, mlx-lm tente d'abord de vérifier en ligne s'il
# existe une version plus récente (et retombe sur le cache si ça échoue).
# HF_HUB_OFFLINE=0 python ... pour autoriser à nouveau le réseau.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODELE = "mlx-community/gemma-4-E4B-it-qat-4bit"


def repondre(model, tok, messages, temperature: float, max_tokens: int, afficher: bool) -> str:
    invite = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    morceaux = []
    for r in stream_generate(model, tok, prompt=invite, max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
        morceaux.append(r.text)
        if afficher:
            print(r.text, end="", flush=True)
    if afficher:
        print()
    return "".join(morceaux).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--adapters", default="carl_gemma/adapters")
    parser.add_argument("--sans-carl", action="store_true", help="Gemma 4 sans l'adaptation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1000)
    args = parser.parse_args()

    model, tok = load(MODELE, adapter_path=None if args.sans_carl else args.adapters)
    if args.question:
        repondre(model, tok, [{"role": "user", "content": args.question}], args.temperature, args.max_tokens, True)
        return

    messages: list[dict] = []
    while True:
        try:
            question = input("\nvous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        messages.append({"role": "user", "content": question})
        print("carl > ", end="", flush=True)
        messages.append({"role": "assistant", "content": repondre(
            model, tok, messages, args.temperature, args.max_tokens, True)})


if __name__ == "__main__":
    main()
