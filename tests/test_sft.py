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


def test_dpo_logp_ne_compte_que_la_reponse():
    # La log-probabilité d'une réponse ne doit dépendre que de ses tokens, pas
    # de ceux de la question ni du remplissage.
    import torch

    from configs.base import Config
    from dpo import logp_reponses
    from model import GPT

    torch.manual_seed(0)
    cfg = Config(n_layer=1, n_head=2, n_embd=16, block_size=32, vocab_size=64)
    m = GPT(cfg).eval()
    seq = [5, 6, 7, 8, 9]  # invite = 3 premiers tokens, réponse = 8, 9
    seul = logp_reponses(m, [(seq, 3)], pad=0, device="cpu")
    avec_voisin = logp_reponses(m, [(seq, 3), ([1] * 12, 2)], pad=0, device="cpu")
    assert torch.allclose(seul[0], avec_voisin[0], atol=1e-5)
    # À la main : log p(8 | 5 6 7) + log p(9 | 5 6 7 8).
    logits, _ = m(torch.tensor([seq]), torch.tensor([seq]))
    lp = torch.log_softmax(logits[0], -1)
    assert torch.allclose(seul[0], lp[2, 8] + lp[3, 9], atol=1e-5)


def test_resultat_d_outil_pas_appris():
    """« Madrid] » est inséré par le programme : Carl n'apprend pas à le deviner, mais tout le reste, si."""
    from chat_format import IGNORE, encoder_conversation
    from tokenizer import BPETokenizer

    tok = BPETokenizer.load("tokenizer/vocab.json")
    conv = [{"role": "user", "content": "Capitale de l'Espagne ?"},
            {"role": "assistant", "content": "[fait: Espagne | capitale = Madrid] La capitale de l'Espagne est Madrid."}]
    ids, cibles = encoder_conversation(tok, conv)
    appris = tok.decode([c for c in cibles if c != IGNORE and c < tok.special_tokens["<|im_start|>"]])
    assert appris == "[fait: Espagne | capitale = La capitale de l'Espagne est Madrid."
    # Même découpage qu'à l'usage : chat.py ajoute tok.encode(" Madrid]") après « = ».
    assert tok.encode(" Madrid]") == ids[ids.index(tok.encode(" =")[0]) + 1:][:len(tok.encode(" Madrid]"))]
