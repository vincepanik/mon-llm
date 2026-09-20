"""
Config de débogage, pour le Mac.

Objectif : vérifier en deux minutes que tout tourne de bout en bout.
Le modèle est minuscule et volontairement inutile.
Si la loss descend et qu'un checkpoint se sauvegarde puis se recharge,
le code est prêt à partir sur une machine louée.
"""

from configs.base import Config

config = Config(
    run_name="debug",
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=64,
    batch_size=8,
    max_steps=200,
    warmup_steps=10,
    eval_interval=50,
    checkpoint_interval=100,
    throttle=0.7,            # le Mac reste silencieux
)
