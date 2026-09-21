"""
Tests du tokenizer. À lancer avec `pytest` avant chaque commit.
"""

import pytest

from tokenizer import BPETokenizer

# Assez long et varié pour permettre les fusions que demande vocab_size=300 :
# un corpus trop court s'épuise avant d'atteindre la cible.
CORPUS = (
    "le chat mange la souris. le chien mange le chat. la souris mange le fromage. "
    "le chat dort sur le toit de la maison, la souris dort dans le mur de la cave. "
    "le chien du voisin aboie contre le chat, le chat ne bouge pas de son coussin. "
    "la maison est grande, le jardin est petit, le mur du fond tombe en morceaux. "
    "l'enfant regarde la souris qui traverse le jardin jusqu'au mur de la maison. "
    "en 2024, le voisin a payé 1250 euros pour réparer le toit et le mur du fond. "
)

PHRASES = [
    "le chat",
    "fromage",
    "un mot jamais vu",
    "éàü 日本語 🙂",
    "L'homme n'est pas venu aujourd'hui, jusqu'à 14h30 !\n\nSuite...",
    "   trois espaces   et  deux",
    "<|endoftext|> en clair dans le texte",
    "",
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    return t


def test_roundtrip(tok):
    for s in PHRASES:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_size(tok):
    assert tok.vocab_size == 300


def test_compression(tok):
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8"))


def test_pas_de_fusion_entre_les_mots(tok):
    # Un mot est encodé pareil quel que soit son voisinage.
    assert tok.encode("la maison.") == tok.encode("la") + tok.encode(" maison") + tok.encode(".")
    assert tok.encode("le chat dort") == tok.encode("le") + tok.encode(" chat") + tok.encode(" dort")
    # Et aucun token appris ne contient un espace ailleurs qu'en tête.
    for i in range(256, 256 + len(tok.merges)):
        morceau = tok.vocab[i].decode("utf-8", errors="replace")
        assert " " not in morceau.lstrip(" "), morceau


def test_tokens_speciaux(tok):
    assert tok.eot == 256 + len(tok.merges)
    assert tok.decode([tok.eot]) == "<|endoftext|>"
    # Écrit en clair dans un texte, ce n'est que du texte : pas de token spécial.
    assert tok.eot not in tok.encode("<|endoftext|>")


def test_save_load(tok, tmp_path):
    p = tmp_path / "vocab.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.special_tokens == tok.special_tokens
    assert tok2.encode(CORPUS) == tok.encode(CORPUS)


def test_tiktoken_identique(tok):
    enc = tok.as_tiktoken()
    for s in PHRASES + [CORPUS]:
        assert enc.encode_ordinary(s) == tok.encode(s), s
    assert enc.n_vocab == tok.vocab_size
    assert enc.eot_token == tok.eot
