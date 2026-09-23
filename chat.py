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
from outils import APPEL, afficher, calculer
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint


def avec_document(messages: list[dict]) -> list[dict]:
    """
    Cherche dans Wikipédia (rag.py) le passage le plus proche de la dernière
    question et le place juste avant elle, comme dans les exemples de lecture
    de l'entraînement (data/lecture.py). Carl y puise la réponse s'il la trouve,
    et l'ignore sinon. Sans index, les messages passent tels quels.
    """
    import rag

    if not rag.disponible() or not messages or messages[-1]["role"] != "user":
        return messages
    trouves = rag.chercher(messages[-1]["content"], k=1)
    if not trouves:
        return messages
    return messages[:-1] + [{"role": "document", "content": trouves[0][0]}, messages[-1]]


# Réponses apprises pour « le passage ne contient pas la réponse » (data/lecture.py).
PAS_DANS_LE_DOCUMENT = ("ne le dit pas", "ne trouve pas cette information", "ne répond pas à cette question")


def repondre(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int,
             repetition_penalty: float, avec_outils: bool = True, wikipedia: bool = False) -> str:
    """
    Avec wikipedia=True : on lui donne d'abord le passage trouvé ; s'il répond
    que le passage ne contient pas la réponse, on repose la question sans
    document, et il répond de mémoire. La lecture quand elle aide, la mémoire
    quand la recherche a ramené un passage à côté.
    """
    reglages = dict(temperature=temperature, top_k=top_k, max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty, avec_outils=avec_outils)
    if wikipedia:
        documentee = avec_document(messages)
        if documentee is not messages:
            r = _generer(model, tok, documentee, device, **reglages)
            if not any(m in r.lower() for m in PAS_DANS_LE_DOCUMENT):
                return r
    return _generer(model, tok, messages, device, **reglages)


def _generer(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int,
             repetition_penalty: float, avec_outils: bool = True) -> str:
    """
    Génère la réponse token par token, pour pouvoir intervenir en cours de route :
    dès que Carl écrit « [calc: <expression> = », la calculatrice (outils.py)
    fait le calcul et on insère le résultat à sa place. Carl continue ensuite
    sa phrase. Le texte rendu remplace « [calc: ... = résultat] » par le résultat.
    """
    ids = debut_de_reponse(tok, messages)
    ids = ids[-(model.cfg.block_size - max_tokens):]  # garder de la place pour la réponse
    fin = tok.special_tokens["<|im_end|>"]
    # Ses propres réponses précédentes : un petit modèle a tendance à les
    # recopier mot pour mot, et une réponse ratée se répète alors en boucle.
    precedentes = [t for m in messages if m["role"] == "assistant" for t in tok.encode(m["content"])]
    reponse: list[int] = []
    while len(reponse) < max_tokens:
        contexte = torch.tensor([(ids + reponse)[-model.cfg.block_size:]], device=device)
        suivant = model.generate(
            contexte, 1,
            temperature=max(temperature, 1e-5), top_k=1 if temperature <= 0 else top_k,
            repetition_penalty=repetition_penalty, penaliser_aussi=precedentes + reponse,
        )[0, -1].item()
        if suivant == fin:
            break
        reponse.append(suivant)
        if avec_outils:
            appel = APPEL.search(tok.decode(reponse[-40:]))
            if appel:
                reponse += tok.encode(f" {calculer(appel.group(1))}]")
    return afficher(tok.decode(reponse)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", default=None)
    # 0 = toujours le mot le plus probable. Mesuré sur 40 faits simples (Carl v4) :
    # 27,7/40 contre 23,3 à 0,7 et 25,7 à 0,3. Le hasard fait piocher à un petit
    # modèle des mots moins probables, donc souvent faux. Monter la température
    # pour des textes créatifs (poème, histoire).
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--sans-wikipedia", action="store_true",
                        help="ne pas chercher dans Wikipédia avant de répondre (rag.py)")
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
                    repetition_penalty=args.repetition_penalty, wikipedia=not args.sans_wikipedia)

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
