# mon-llm

Un LLM construit brique par brique. Une seule base de code, qui tourne sur le Mac
(pour déboguer) et sur une machine louée (pour les vrais entraînements) sans
changer une ligne. Seul le fichier de config change.

## Règles du projet

1. Le Mac ne fait tourner que des modèles jouets, quelques minutes, pour déboguer.
   Les vrais runs partent sur RunPod (ou équivalent).
2. Avant chaque run payant : la config de debug passe sur le Mac, puis la vraie
   config tourne 50 étapes sur le GPU loué. Seulement ensuite on lance en grand.
3. Le code vit sur GitHub. Les données et les checkpoints ne passent jamais par Git.
4. Sur une machine louée, on lance toujours dans `tmux`.
5. Les checkpoints se sauvegardent sur un volume persistant, jamais sur le disque du pod.

## Feuille de route

| Étape | Brique | Fichier | État |
|---|---|---|---|
| 1 | Tokenizer BPE écrit à la main | `tokenizer/bpe.py` | fait |
| 2 | Transformer minimal, niveau caractère, ~1M params | `model.py` | fait |
| 3 | From scratch 30 à 150M sur corpus français | `configs/run_150m.py` | à faire |
| 4 | Modernisation mesurée : RoPE, RMSNorm, SwiGLU, GQA | `model.py` | à faire |
| 5 | Post-training : SFT puis DPO | à créer | à faire |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandes

```bash
# Récupérer du corpus (web français récent, FineWeb-2), 20 Mo pour déboguer
python data/download_fineweb.py --mb 20

# Entraîner le tokenizer BPE (un échantillon suffit, inutile de lire tout le corpus)
python tokenizer/bpe.py --input data/raw --vocab-size 32000 --sample-mb 50

# Préparer les données (tokenise le corpus en fichiers binaires)
python data/prepare.py --config configs/debug_mac.py

# Entraîner
python train.py --config configs/debug_mac.py

# Générer du texte depuis un checkpoint
python sample.py --checkpoint checkpoints/debug/latest.pt --prompt "Il était une fois"

# Tests
pytest
```

## Structure

```
mon-llm/
  configs/        une config par expérience, la base est dans base.py
  data/           download_fineweb.py récupère le corpus, prepare.py le transforme en .bin (ignorés par git)
  tokenizer/      le tokenizer BPE, brique 1, et vocab.json une fois entraîné
  model.py        l'architecture, brique 2 puis 4
  train.py        la boucle d'entraînement, identique partout
  sample.py       génération de texte
  utils.py        device, dtype, seed, throttle, checkpoints
  scripts/        mise en place d'une machine louée
  tests/          tests rapides, à lancer avant chaque commit
  checkpoints/    ignoré par git
```
