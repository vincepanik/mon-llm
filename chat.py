"""
Discuter avec le modèle après le SFT.

    python chat.py --checkpoint checkpoints/sft/best.pt
    python chat.py --checkpoint checkpoints/sft/best.pt --question "Qu'est-ce que la photosynthèse ?"

Sans --question : conversation au clavier, le modèle se souvient des tours
précédents (dans la limite de son contexte). Entrée vide pour quitter.
"""

from __future__ import annotations

import argparse

import torch

from chat_format import debut_de_reponse
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint


def repondre(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int) -> str:
    ids = debut_de_reponse(tok, messages)
    ids = ids[-(model.cfg.block_size - max_tokens):]  # garder de la place pour la réponse
    fin = tok.special_tokens["<|im_end|>"]
    out = model.generate(
        torch.tensor([ids], device=device), max_tokens,
        temperature=temperature, top_k=top_k, stop_token=fin,
    )[0].tolist()[len(ids):]
    if fin in out:
        out = out[: out.index(fin)]
    return tok.decode(out).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    model = GPT(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(ck["config"].tokenizer_path)
    reglages = dict(temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens)

    if args.question:
        print(repondre(model, tok, [{"role": "user", "content": args.question}], device, **reglages))
        return

    messages: list[dict] = []
    while True:
        question = input("\nvous > ").strip()
        if not question:
            break
        messages.append({"role": "user", "content": question})
        reponse = repondre(model, tok, messages, device, **reglages)
        print(f"modèle > {reponse}")
        messages.append({"role": "assistant", "content": reponse})


if __name__ == "__main__":
    main()
