"""
Config du premier vrai run, environ 140M de paramètres, architecture moderne.

À lancer sur un H100 (~8-10 h) ou une RTX 4090 (~24-30 h). Avant de lancer en
grand, faire tourner cette même config avec max_steps=50 pour vérifier.

Données : 5 milliards de tokens uniques (FineWeb2-HQ 60 %, Wikipédia 25 %,
science 15 %) vus deux fois = 10 milliards de tokens, soit ~70 par paramètre.
Le tokenizer 32k est réentraîné sur un échantillon du mélange final.

Forme du modèle : celle de GPT-2 small, un peu plus profonde, avec les briques
de Llama (RoPE, RMSNorm, SwiGLU, GQA 12 têtes / 4 têtes kv).
"""

from pathlib import Path

from configs.base import Config

config = Config(
    run_name="run_150m",
    data_dir=Path("data/big"),  # le gros corpus, séparé des 40 Mo de debug
    vocab_size=32000,
    n_layer=16,
    n_head=12,
    n_kv_head=4,
    n_embd=768,
    block_size=1024,
    norm="rmsnorm",
    pos="rope",
    mlp="swiglu",             # largeur cachée 2048 (8/3 x 768, arrondi à 64)
    dropout=0.0,
    batch_size=16,
    grad_accum_steps=32,      # batch effectif de 512 séquences de 1024 tokens = 524 288 tokens
    max_steps=20000,          # x 524 288 = 10,5 milliards de tokens
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_steps=1000,
    eval_interval=500,
    eval_steps=50,
    checkpoint_interval=1000,
    use_bf16=True,
    use_compile=True,
    use_flash_attn=True,
    throttle=1.0,
)
