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
| 4 | Modernisation mesurée : RoPE, RMSNorm, SwiGLU, GQA | `model.py` | fait |
| 3 | From scratch ~125M sur corpus français | `configs/run_150m.py` | fait |
| 5 | Post-training : SFT puis DPO | à créer | à faire |

L'étape 4 est passée avant la 3 : on ne paye le GPU qu'une fois, autant que ce
soit avec l'architecture finale.

### Étape 4, ce qui a été mesuré

Sur le Mac, 0,8M paramètres (4 couches x 128), 3 000 étapes sur 40 Mo de
français niveau octet (Wikipédia + science), même graine, même planning de lr.
Loss de validation à la fin :

| Variante | Paramètres | val | Écart |
|---|---|---|---|
| GPT-2 (étape 2) | 0,84M | 1,608 | référence |
| + RMSNorm | 0,84M | 1,613 | 0 : pas un gain de qualité, un gain de vitesse sur GPU |
| + RoPE | 0,82M | 1,510 | **-0,10** |
| + SwiGLU | 0,90M | 1,515 | **-0,09** (un peu plus de paramètres à cette taille, arrondi à 64) |
| + GQA (4 têtes, 2 kv) | 0,77M | 1,626 | +0,02 : le prix du cache kv divisé par deux |
| Les quatre | 0,82M | **1,441** | **-0,17** |

Le bruit entre deux graines est de l'ordre de 0,01 : RoPE et SwiGLU sont des
gains nets, GQA un léger coût assumé, RMSNorm neutre. Les configs de debug et
du vrai run utilisent les quatre.

### Étape 3, le premier vrai run

| | |
|---|---|
| Modèle | 125,3M paramètres, 16 x 768, 12 têtes / 4 kv, RoPE, RMSNorm, SwiGLU |
| Données | 4,82 milliards de tokens (FineWeb2-HQ 60 %, Wikipédia 25 %, science 15 %), tokenizer BPE 32k |
| Entraînement | 20 000 étapes de 524 288 tokens = 10,5 milliards de tokens (~2,2 passages) |
| Machine | 1 x RTX 5090, RunPod Secure Cloud EU-RO-1, 0,99 $/h |
| Vitesse | 178 000 tokens/s, 2,94 s par étape |
| Durée, coût | 16 h 20, ~17,60 $ tout compris (installation, envoi du corpus, test) |
| Validation | 4,16 (étape 500), 3,37 (1 000), 2,54 (14 500), **2,51 (19 000, `best.pt`)**, 2,54 (20 000) |

Ce qu'il sait faire : continuer un texte dans un français correct et fluide, en
tenant le sujet et le registre sur un paragraphe (une recette a des
intertitres, un article scientifique un ton encyclopédique). Ce qu'il ne sait
pas : dire des choses vraies (il invente dates, chiffres et mécanismes avec
aplomb), tenir un fil au-delà de quelques phrases, répondre à une question.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandes

```bash
# Récupérer du corpus, 20 Mo par source pour déboguer
# (sources : fineweb_hq | wikipedia | science | fineweb)
python data/download.py --source fineweb_hq --mb 20
python data/download.py --source wikipedia --mb 20
python data/download.py --source science --mb 20

# Entraîner le tokenizer BPE (un échantillon suffit, inutile de lire tout le corpus)
python tokenizer/bpe.py --input data/raw --vocab-size 32000 --sample-mb 50

# Préparer les données (tokenise le corpus en fichiers binaires)
python data/prepare.py --config configs/debug_mac.py

# Entraîner
python train.py --config configs/debug_mac.py

# Générer du texte depuis un checkpoint (s'arrête seul à la fin du texte)
python sample.py --checkpoint checkpoints/run_150m/best.pt --prompt "Il était une fois"

# Tests
pytest
```

## Structure

```
mon-llm/
  configs/        une config par expérience, la base est dans base.py
  data/           download.py récupère le corpus, prepare.py le transforme en .bin (ignorés par git)
  tokenizer/      le tokenizer BPE, brique 1, et vocab.json une fois entraîné
  model.py        l'architecture, brique 2 puis 4
  train.py        la boucle d'entraînement, identique partout
  sample.py       génération de texte
  utils.py        device, dtype, seed, throttle, checkpoints
  scripts/        mise en place d'une machine louée
  tests/          tests rapides, à lancer avant chaque commit
  checkpoints/    ignoré par git
```
