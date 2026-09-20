"""
Config du premier vrai run, environ 150M de paramètres.

À lancer sur une RTX 4090 ou une A100. Avant de lancer en grand,
faire tourner cette même config avec max_steps=50 pour vérifier.

Les valeurs ci-dessous sont un point de départ raisonnable, à affiner.
"""

from configs.base import Config

config = Config(
    run_name="run_150m",
    vocab_size=32000,         # tokenizer BPE, à aligner sur celui qu'on utilisera
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    dropout=0.0,
    batch_size=16,
    grad_accum_steps=32,      # batch effectif de 512 séquences de 1024 tokens
    max_steps=20000,
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_steps=1000,
    eval_interval=500,
    checkpoint_interval=1000,
    use_bf16=True,
    use_compile=True,
    use_flash_attn=True,
    throttle=1.0,
)
