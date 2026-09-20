"""
Brique 1 : tokenizer BPE (Byte Pair Encoding) écrit à la main.

Principe :
  1. On part des 256 octets possibles comme vocabulaire de départ.
  2. On compte la paire d'octets (ou de tokens) la plus fréquente dans le corpus.
  3. On la fusionne en un nouveau token, qu'on ajoute au vocabulaire.
  4. On recommence jusqu'à atteindre la taille de vocabulaire voulue.

Encoder un texte, c'est rejouer ces fusions dans l'ordre.
Décoder, c'est remplacer chaque token par ses octets et lire le résultat en UTF-8.

On travaille sur des octets, jamais sur des caractères : n'importe quel texte
(accents, japonais, emoji) passe sans cas particulier, et le vocabulaire de
départ est fini et connu.

Ligne de commande (pour fabriquer tokenizer/vocab.json avant data/prepare.py) :

    python tokenizer/bpe.py --input data/raw --vocab-size 4096 --out tokenizer/vocab.json
"""

from __future__ import annotations

import json
from pathlib import Path


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Compte chaque paire de tokens voisins. [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Remplace toutes les occurrences de `pair` par `new_id`, de gauche à droite."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # merges : (token_a, token_b) -> nouveau_token, dans l'ordre d'apprentissage
        self.merges: dict[tuple[int, int], int] = {}
        # vocab : token -> bytes correspondants
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Apprend les fusions sur `text` jusqu'à atteindre `vocab_size` tokens."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size doit valoir au moins 256, reçu {vocab_size}")

        # On repart d'un tokenizer neuf : réentraîner ne doit pas empiler deux corpus.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        ids = list(text.encode("utf-8"))
        n_merges = vocab_size - 256

        for k in range(n_merges):
            stats = _get_stats(ids)
            if not stats:
                # Plus une seule paire : le corpus est entièrement replié sur lui-même.
                print(
                    f"corpus épuisé après {k} fusions : vocabulaire de {self.vocab_size} "
                    f"tokens au lieu de {vocab_size}. Il faut plus de texte."
                )
                break

            # La paire la plus fréquente. À égalité, max() garde la première vue,
            # donc l'entraînement est déterministe.
            pair = max(stats, key=stats.get)
            new_id = 256 + k
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                morceau = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"fusion {k + 1}/{n_merges} : {pair} -> {new_id} "
                    f"({morceau!r}, {stats[pair]} occurrences, reste {len(ids)} tokens)"
                )

    def encode(self, text: str) -> list[int]:
        """Texte -> liste d'identifiants de tokens."""
        ids = list(text.encode("utf-8"))
        # On rejoue les fusions dans l'ordre où elles ont été apprises : à chaque
        # tour, on applique celle dont l'indice est le plus petit parmi les paires
        # encore présentes. Sinon on créerait des tokens que l'entraînement n'a
        # jamais vus dans cet ordre.
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # plus rien à fusionner
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Liste d'identifiants -> texte."""
        octets = b"".join(self.vocab[i] for i in ids)
        # errors="replace" : un modèle fraîchement initialisé sort des tokens au
        # hasard, qui peuvent couper une séquence UTF-8 en plein milieu. On préfère
        # un caractère de remplacement à une exception pendant la génération.
        return octets.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Sauvegarde les fusions en JSON. Le vocab se reconstruit à partir des fusions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"merges": [[a, b, c] for (a, b), c in self.merges.items()]}
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        return tok


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Entraîne le tokenizer BPE sur un corpus.")
    parser.add_argument("--input", required=True, help="fichier .txt ou dossier de .txt")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--out", default="tokenizer/vocab.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    files = sorted(source.glob("*.txt")) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"Aucun fichier .txt dans {source}")
    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    print(f"{len(files)} fichier(s), {len(text):,} caractères")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    print(f"entraînement : {time.time() - t0:.1f} s, {tok.vocab_size} tokens")

    ids = tok.encode(text)
    print(f"compression : {len(text.encode('utf-8')):,} octets -> {len(ids):,} tokens "
          f"({len(text.encode('utf-8')) / max(len(ids), 1):.2f}x)")

    tok.save(args.out)
    print(f"écrit dans {args.out}")


if __name__ == "__main__":
    main()
