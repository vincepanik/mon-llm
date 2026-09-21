"""
Brique 1 : tokenizer BPE (Byte Pair Encoding) écrit à la main.

Principe :
  0. Le texte est d'abord découpé en « mots » par une expression régulière
     (PATTERN ci-dessous) : l'espace reste collé au mot qui le suit, les
     chiffres vont par paquets de trois au plus, la ponctuation est à part.
  1. Chaque mot devient sa suite d'octets : le vocabulaire de départ, ce sont
     les 256 octets possibles.
  2. On compte la paire de tokens voisins la plus fréquente, à l'intérieur des mots.
  3. On la fusionne en un nouveau token, qu'on ajoute au vocabulaire.
  4. On recommence jusqu'à atteindre la taille de vocabulaire voulue.

Encoder un texte, c'est le découper en mots puis rejouer les fusions dans l'ordre.
Décoder, c'est remplacer chaque token par ses octets et lire le résultat en UTF-8.

Pourquoi le pré-découpage de l'étape 0 ? Sans lui, BPE apprend des tokens comme
« de la » ou « s de » qui collent des mots entre eux, gaspille du vocabulaire
sur ces collages, et découpe un même mot différemment selon ses voisins. Avec
lui, une fusion ne traverse jamais la frontière entre deux mots. C'est ce que
font GPT-4, Llama 3, DeepSeek et Qwen ; le motif ci-dessous est celui de GPT-4,
dont on a retiré les contractions anglaises ('s, 've, ...) pour mettre les
élisions françaises (l', d', qu', ...) à la place.

On travaille sur des octets, jamais sur des caractères : n'importe quel texte
(accents, japonais, emoji) passe sans cas particulier, rien n'est jamais perdu.

Tokens spéciaux : des identifiants réservés, hors du texte, placés après les
fusions. <|endoftext|> sépare les documents à l'entraînement ; les autres
serviront à l'étape 5 (SFT). Un texte qui contient littéralement la chaîne
« <|endoftext|> » est encodé comme du texte ordinaire : seul data/prepare.py
insère le vrai token, via `tok.eot`.

Vitesse : l'entraînement ne voit que les mots uniques avec leur fréquence (le
mot « le » apparaît des millions de fois mais n'est traité qu'une fois), et met
à jour les comptes de paires de façon incrémentale. L'encodage Python est la
référence, mais lent ; pour des giga-octets, `as_tiktoken()` exporte exactement
les mêmes fusions vers tiktoken (la bibliothèque Rust d'OpenAI). Un test
vérifie que les deux donnent les mêmes identifiants.

Ligne de commande (pour fabriquer tokenizer/vocab.json avant data/prepare.py) :

    python tokenizer/bpe.py --input data/raw --vocab-size 32000 --sample-mb 50
"""

from __future__ import annotations

import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import regex  # pas `re` : il faut \p{L} (une lettre, dans n'importe quel alphabet)

# Chaque alternative, dans l'ordre, séparée par | :
#   les élisions françaises, avec leur espace devant : « l' », « qu' », « jusqu' »
#   un mot, avec au plus un caractère non-lettre devant (typiquement l'espace)
#   un nombre, par paquets de 1 à 3 chiffres : 2024 -> « 202 » « 4 »
#   de la ponctuation, avec au plus un espace devant et les sauts de ligne qui suivent
#   des sauts de ligne, avec les blancs qui les précèdent
#   des blancs sauf le dernier, qui restera collé au mot suivant
#   des blancs
PATTERN = (
    r"""(?i: ?(?:qu|jusqu|lorsqu|puisqu|[cdjlmnst])['’](?=\p{L}))"""
    r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""
    r"""|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


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
    def __init__(self, pattern: str = PATTERN, special_tokens: list[str] | None = None) -> None:
        self.pattern = pattern
        self._pat = regex.compile(pattern)
        # merges : (token_a, token_b) -> nouveau_token, dans l'ordre d'apprentissage
        self.merges: dict[tuple[int, int], int] = {}
        # vocab : token -> bytes correspondants
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._special_names = list(SPECIAL_TOKENS if special_tokens is None else special_tokens)
        # special_tokens : nom -> identifiant, recalculé dès que les fusions changent
        self.special_tokens: dict[str, int] = {}
        self._assign_specials()
        # mot -> ids : les mêmes mots reviennent sans cesse, on ne les recalcule pas
        self._cache: dict[str, list[int]] = {}

    def _assign_specials(self) -> None:
        base = 256 + len(self.merges)
        self.special_tokens = {name: base + i for i, name in enumerate(self._special_names)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot(self) -> int:
        """Identifiant de <|endoftext|>, à insérer entre deux documents."""
        return self.special_tokens["<|endoftext|>"]

    # ------------------------------------------------------------------ train

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Apprend les fusions sur `text` jusqu'à atteindre `vocab_size` tokens."""
        n_merges = vocab_size - 256 - len(self._special_names)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size doit valoir au moins {256 + len(self._special_names)}, reçu {vocab_size}"
            )

        # On repart d'un tokenizer neuf : réentraîner ne doit pas empiler deux corpus.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}
        t0 = time.time()

        # Étape 0 : pré-découpage. Un mot n'est traité qu'une fois, avec sa fréquence.
        counts = Counter(self._pat.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]  # tokens courants de chaque mot
        freqs = list(counts.values())
        if verbose:
            print(f"{len(text):,} caractères, {sum(freqs):,} mots, {len(words):,} uniques")

        # Comptes de paires, et pour chaque paire, dans quels mots elle apparaît.
        # C'est cet index qui rend l'entraînement rapide : à chaque fusion on ne
        # touche que les mots concernés, pas tout le corpus.
        stats: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                stats[p] += f
                where[p].add(wi)

        # Un tas pour trouver la paire la plus fréquente sans tout parcourir. Les
        # entrées périmées (compte qui a changé depuis) sont ignorées à la sortie.
        # À compte égal, la paire d'identifiants les plus petits gagne : déterministe.
        heap = [(-c, p) for p, c in stats.items()]
        heapq.heapify(heap)

        for k in range(n_merges):
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if neg_count < 0 and stats.get(pair) == -neg_count:
                    break
            else:
                print(
                    f"corpus épuisé après {k} fusions : vocabulaire de {256 + k + len(self._special_names)} "
                    f"tokens au lieu de {vocab_size}. Il faut plus de texte."
                )
                break

            new_id = 256 + k
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Mise à jour incrémentale : seulement les mots qui contiennent la paire.
            touched: set[tuple[int, int]] = set()
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    stats[p] -= f
                    where[p].discard(wi)
                    touched.add(p)
                w = _merge(w, pair, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    stats[p] += f
                    where[p].add(wi)
                    touched.add(p)
            for p in touched:
                if stats[p] <= 0:
                    del stats[p]
                    del where[p]
                else:
                    heapq.heappush(heap, (-stats[p], p))

            if verbose and ((k + 1) % 1000 == 0 or k < 10 or k + 1 == n_merges):
                morceau = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"fusion {k + 1}/{n_merges} : {pair} -> {new_id} ({morceau!r}, "
                    f"{-neg_count:,} occurrences, {time.time() - t0:.0f} s)"
                )

        self._assign_specials()

    # ----------------------------------------------------------- encode/decode

    def _encode_word(self, word: str) -> list[int]:
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        ids = list(word.encode("utf-8"))
        # On rejoue les fusions dans l'ordre où elles ont été apprises : à chaque
        # tour, celle dont l'indice est le plus petit parmi les paires présentes.
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # plus rien à fusionner
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) > 1_000_000:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Texte -> liste d'identifiants de tokens. Jamais de token spécial en sortie."""
        ids: list[int] = []
        for word in self._pat.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Liste d'identifiants -> texte. Les tokens spéciaux ressortent en clair."""
        specials = {i: name.encode("utf-8") for name, i in self.special_tokens.items()}
        parts = []
        for i in ids:
            b = self.vocab.get(i)
            if b is None:
                b = specials.get(i)
            if b is None:
                raise ValueError(f"identifiant inconnu : {i}")
            parts.append(b)
        # errors="replace" : un modèle fraîchement initialisé sort des tokens au
        # hasard, qui peuvent couper une séquence UTF-8 en plein milieu. On préfère
        # un caractère de remplacement à une exception pendant la génération.
        return b"".join(parts).decode("utf-8", errors="replace")

    def as_tiktoken(self):
        """
        Le même tokenizer, exécuté par tiktoken (Rust, ~100x plus rapide).

        tiktoken ne stocke pas les paires fusionnées mais le rang de chaque token :
        il fusionne toujours la paire voisine dont le résultat a le plus petit rang.
        Comme nos identifiants sont attribués dans l'ordre des fusions, c'est
        exactement le même calcul. tests/test_tokenizer.py le vérifie.
        """
        import tiktoken

        ranks = {b: i for i, b in self.vocab.items()}
        assert len(ranks) == len(self.vocab), "deux tokens avec les mêmes octets"
        return tiktoken.Encoding(
            name="mon-llm",
            pat_str=self.pattern,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )

    # ------------------------------------------------------------- save/load

    def save(self, path: str | Path) -> None:
        """Sauvegarde en JSON. Le vocab se reconstruit à partir des fusions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pattern": self.pattern,
            "special_tokens": self._special_names,
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=data["pattern"], special_tokens=data["special_tokens"])
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        tok._assign_specials()
        return tok


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Entraîne le tokenizer BPE sur un corpus.")
    parser.add_argument("--input", required=True, help="fichier .txt ou dossier de .txt")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--out", default="tokenizer/vocab.json")
    parser.add_argument("--sample-mb", type=float, default=None,
                        help="n'apprendre que sur les N premiers Mo (un échantillon suffit)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    files = sorted(source.glob("*.txt")) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"Aucun fichier .txt dans {source}")
    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    # Les séparateurs de documents du corpus brut ne sont pas du texte à apprendre.
    text = text.replace("<|endoftext|>", "\n")
    if args.sample_mb is not None:
        text = text[: int(args.sample_mb * 1024 * 1024)]
    print(f"{len(files)} fichier(s), {len(text):,} caractères")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    print(f"entraînement : {time.time() - t0:.0f} s, {tok.vocab_size} tokens")

    echantillon = text[: 2 * 1024 * 1024]
    ids = tok.as_tiktoken().encode_ordinary(echantillon)
    octets = len(echantillon.encode("utf-8"))
    print(f"compression : {octets / len(ids):.2f} octets par token")

    tok.save(args.out)
    print(f"écrit dans {args.out}")


if __name__ == "__main__":
    main()
