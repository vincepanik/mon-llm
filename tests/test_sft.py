"""Format des conversations : on ne corrige que les réponses, et leur fin."""

from chat_format import IGNORE, debut_de_reponse, encoder_conversation
from tokenizer import BPETokenizer

CORPUS = "Bonjour, comment ça va ? Très bien, merci. Et vous ? " * 20


def test_seules_les_reponses_sont_apprises():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    conv = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Très bien"},
        {"role": "user", "content": "Et vous ?"},
        {"role": "assistant", "content": "Merci"},
    ]
    ids, cibles = encoder_conversation(tok, conv)
    assert len(ids) == len(cibles)
    appris = tok.decode([c for c in cibles if c != IGNORE])
    assert appris == "Très bien<|im_end|>Merci<|im_end|>"
    # Chaque cible est bien le token suivant.
    for i, c in enumerate(cibles):
        if c != IGNORE:
            assert c == ids[i + 1]


def test_debut_de_reponse():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    ids = debut_de_reponse(tok, [{"role": "user", "content": "Bonjour"}])
    assert tok.decode(ids).endswith("<|im_end|>\n<|im_start|>assistant\n")
