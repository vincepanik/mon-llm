"""
Génère du texte depuis un checkpoint.

    python sample.py --checkpoint checkpoints/debug/latest.pt --prompt "Il était une fois"
"""

from __future__ import annotations

import argparse

import torch

from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    cfg = ck["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if cfg.vocab_size <= 256:
        encode = lambda s: list(s.encode("utf-8"))
        decode = lambda ids: bytes(ids).decode("utf-8", errors="replace")
        eot = None  # niveau octet : pas de token de fin de document
    else:
        tok = BPETokenizer.load(cfg.tokenizer_path)
        encode, decode = tok.encode, tok.decode
        eot = tok.eot

    prompt_ids = encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, stop_token=eot
    )
    # Quand le modèle produit <|endoftext|>, son texte est fini : on coupe là,
    # sinon on afficherait le début d'un autre document sans rapport.
    genere = out[0].tolist()[len(prompt_ids):]
    if eot is not None and eot in genere:
        genere = genere[: genere.index(eot)]
    print(decode(prompt_ids + genere))


if __name__ == "__main__":
    main()
