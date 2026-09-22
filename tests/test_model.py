"""
Tests du modèle, sur les deux architectures : GPT-2 (étape 2) et moderne (étape 4).
"""

import math

import pytest
import torch

from configs.base import Config
from model import GPT

GPT2 = dict(n_layer=2, n_head=4, n_embd=32, block_size=16, vocab_size=64)
MODERNE = dict(GPT2, norm="rmsnorm", pos="rope", mlp="swiglu", n_kv_head=2)


@pytest.fixture(params=[GPT2, MODERNE], ids=["gpt2", "moderne"])
def model(request):
    torch.manual_seed(0)
    cfg = Config(**request.param)
    cfg.validate()
    return GPT(cfg)


def test_forward_et_loss_initiale(model):
    V, T = model.cfg.vocab_size, model.cfg.block_size
    x = torch.randint(0, V, (3, T))
    y = torch.randint(0, V, (3, T))
    logits, loss = model(x, y)
    assert logits.shape == (3, T, V)
    # Au départ le modèle tire au hasard : la loss vaut ln(vocab) à peu près.
    assert abs(loss.item() - math.log(V)) < 0.2
    logits, loss = model(x)
    assert logits.shape == (3, 1, V) and loss is None


def test_causalite(model):
    # Changer le dernier token ne doit rien changer aux prédictions d'avant.
    model.eval()
    V, T = model.cfg.vocab_size, model.cfg.block_size
    a = torch.randint(0, V, (1, T))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % V
    la, _ = model(a, a)
    lb, _ = model(b, b)
    assert torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-5)


def test_apprend(model):
    # Sur un seul batch répété, la loss doit s'effondrer.
    V, T = model.cfg.vocab_size, model.cfg.block_size
    x = torch.randint(0, V, (2, T))
    y = torch.randint(0, V, (2, T))
    opt = model.configure_optimizer(model.cfg, torch.device("cpu"))
    for g in opt.param_groups:
        g["lr"] = 3e-3
    _, avant = model(x, y)
    for _ in range(40):
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    _, apres = model(x, y)
    assert apres.item() < avant.item() * 0.5


def test_generate(model):
    V, T = model.cfg.vocab_size, model.cfg.block_size
    x = torch.randint(0, V, (1, 5))
    out = model.generate(x, max_new_tokens=T + 3, temperature=0.8, top_k=10)
    assert out.shape == (1, 5 + T + 3)  # contexte plus long que block_size : tronqué, pas planté
    assert torch.equal(out[:, :5], x)


def test_gqa_et_rope_reduisent_les_parametres():
    n = lambda **k: sum(p.numel() for p in GPT(Config(**GPT2, **k)).parameters())
    assert n(pos="rope") < n()          # plus de table de positions
    assert n(n_kv_head=1) < n()         # k et v plus petits
    assert n(norm="rmsnorm") == n()     # un gain par canal dans les deux cas (bias=False)


def test_checkpoint_de_modele_compile(tmp_path):
    # Les poids d'un modèle compilé portent le préfixe "_orig_mod." : le
    # chargement doit le retirer pour les remettre dans un modèle normal.
    from utils import load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    cfg = Config(**MODERNE)
    model = GPT(cfg)
    opt = model.configure_optimizer(cfg, torch.device("cpu"))
    compile_ = torch.compile(model)  # rien n'est exécuté, seul l'emballage compte
    p = tmp_path / "ck.pt"
    save_checkpoint(p, compile_, opt, step=3, cfg=cfg, best_val=1.0)
    assert any(k.startswith("_orig_mod.") for k in torch.load(p, weights_only=False)["model"])

    ck = load_checkpoint(p, torch.device("cpu"))
    neuf = GPT(cfg)
    neuf.load_state_dict(ck["model"])  # planterait si le préfixe restait
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    assert torch.equal(neuf(x)[0], model(x)[0])
