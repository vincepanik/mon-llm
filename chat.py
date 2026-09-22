"""
Discuter avec le modèle après le SFT.

    python chat.py --checkpoint checkpoints/sft/best.pt
    python chat.py --checkpoint checkpoints/sft/best.pt --question "Qu'est-ce que la photosynthèse ?"

Sans --question : conversation au clavier. Entrée vide pour quitter.

Par défaut, Carl ne voit que la question en cours (--memoire 0). Mesuré sur une
conversation de 6 questions rejouée 3 fois : sans mémoire, il répond juste aux
questions d'identité de la fin 6 fois sur 6 ; avec l'échange précédent sous
les yeux, 2 fois sur 6 seulement. À 125M paramètres, le réflexe de recopier ce
qui est dans le contexte l'emporte sur le fil de la conversation. --memoire N
lui montre les N échanges précédents, pour expérimenter.
"""

from __future__ import annotations

import argparse

import torch

from chat_format import debut_de_reponse
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint


def repondre(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int,
             repetition_penalty: float) -> str:
    ids = debut_de_reponse(tok, messages)
    ids = ids[-(model.cfg.block_size - max_tokens):]  # garder de la place pour la réponse
    fin = tok.special_tokens["<|im_end|>"]
    # Ses propres réponses précédentes : un petit modèle a tendance à les
    # recopier mot pour mot, et une réponse ratée se répète alors en boucle.
    precedentes = [t for m in messages if m["role"] == "assistant" for t in tok.encode(m["content"])]
    out = model.generate(
        torch.tensor([ids], device=device), max_tokens,
        temperature=temperature, top_k=top_k, stop_token=fin,
        repetition_penalty=repetition_penalty, penaliser_aussi=precedentes,
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
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--memoire", type=int, default=0,
                        help="échanges précédents montrés au modèle (0 : chaque question seule)")
    args = parser.parse_args()

    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    model = GPT(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(ck["config"].tokenizer_path)
    reglages = dict(temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty)

    if args.question:
        print(repondre(model, tok, [{"role": "user", "content": args.question}], device, **reglages))
        return

    messages: list[dict] = []
    while True:
        try:
            question = input("\nvous > ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D, Ctrl+C, ou pas de clavier du tout
            print()
            break
        if not question:
            break
        messages.append({"role": "user", "content": question})
        vus = messages[-(2 * args.memoire + 1):]  # la question, et les N échanges d'avant
        reponse = repondre(model, tok, vus, device, **reglages)
        print(f"modèle > {reponse}")
        messages.append({"role": "assistant", "content": reponse})


if __name__ == "__main__":
    main()
