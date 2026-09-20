"""
Tests du tokenizer. À lancer avec `pytest` avant chaque commit.
Ils échouent tant que la brique 1 n'est pas écrite, c'est normal.
"""

import pytest

from tokenizer import BPETokenizer

# Assez long et varié pour permettre les 44 fusions que demande vocab_size=300 :
# un corpus trop court se replie en un seul token avant d'atteindre la cible.
CORPUS = (
    "le chat mange la souris. le chien mange le chat. la souris mange le fromage. "
    "le chat dort sur le toit de la maison, la souris dort dans le mur de la cave. "
    "le chien du voisin aboie contre le chat, le chat ne bouge pas de son coussin. "
    "la maison est grande, le jardin est petit, le mur du fond tombe en morceaux. "
)


@pytest.fixture
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    return t


def test_roundtrip(tok):
    for s in ["le chat", "fromage", "un mot jamais vu", "éàü 日本語 🙂"]:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_size(tok):
    assert tok.vocab_size == 300


def test_compression(tok):
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8"))


def test_save_load(tok, tmp_path):
    p = tmp_path / "vocab.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(CORPUS) == tok.encode(CORPUS)
