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
| 5 | Post-training : SFT puis DPO | `sft.py`, `chat.py` | SFT fait, DPO à faire |

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

### Étape 5, le SFT

1 490 conversations françaises écrites par des humains (OpenAssistant oasst2 et
Aya, Apache 2.0), format ChatML, loss sur les réponses seulement. 3 epochs sur
le Mac en 25 min, gratuit. Loss de validation des réponses : 2,56 avant,
**2,25 après la 1re epoch** (`checkpoints/sft/best.pt`), puis 2,27 et 2,32 :
avec si peu d'exemples, le modèle commence à les apprendre par cœur dès la
2e epoch.

Le modèle répond au lieu de continuer le texte et s'arrête seul
(« La capitale de l'Italie est Rome. »). Le fond reste celui d'un 125M : hors
des questions les plus simples, il répond à côté ou invente.

**Carl** (`checkpoints/carl/best.pt`) : même SFT, plus 51 conversations
d'identité écrites à la main (`data/identite_carl.py`, répétées 3 fois).
Il salue, se présente, dit avoir été créé par Kevin Pacini et qu'il se trompe
souvent ; la loss de validation des réponses générales est inchangée (2,25).
`chat.py` ajoute une pénalité de répétition (1,15) sur les tokens de la
réponse en cours et des réponses précédentes, contre les boucles.

La v2 ajoute 300 conversations « salut, puis vraie question » et des réponses
à « je voudrais discuter » : il ne reboucle plus sur sa présentation. Mais il ne
tient pas une conversation : avec l'échange précédent dans son contexte, il
répond juste aux questions d'identité 2 fois sur 6 (il recopie le fil), contre
6 sur 6 quand il ne voit que la question. `chat.py` traite donc chaque question
seule par défaut (`--memoire 0`).

    python sft.py --checkpoint checkpoints/run_150m/best.pt \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 300 \
        --epochs 2 --out checkpoints/carl
    python chat.py --checkpoint checkpoints/carl_v6/best.pt

### Niveau 1 : Carl de 125M avec compar:IA, puis DPO

compar:IA (ministère de la Culture, Etalab 2.0), filtré : français, modèles
ouverts sous licence permissive, conversations qui tiennent en 512 tokens,
sans assistant qui se présente comme un autre modèle (sinon Carl répondait
« Je suis Qwen »). SFT sur 29 639 conversations (2 epochs, ~2 h 40 de calcul
sur le Mac), réparation d'identité (identité x10, 12 min) -> `carl_v4`, puis
DPO sur 1 064 paires de votes -> `carl_dpo` (14 min).

`examen.py`, 12 réponses par volet (4 questions x 3 tirages, formulations
absentes de l'entraînement) ; un écart de 1 ou 2 points est du bruit :

| | identité | savoirs | conduite |
|---|---|---|---|
| Carl v2 (1 500 conversations) | 5/12 | 10/12 | 6/12 |
| **Carl v4** (compar:IA + réparation) | 6/12 | 7/12 | **11/12** |
| Carl DPO | 5/12 | 8/12 | 10/12 |

compar:IA a appris à Carl à se conduire en assistant (il ne s'invente plus
l'heure ni la météo), au prix d'un peu de ses rares faits : il imite la forme
d'une réponse assurée (« La capitale de l'Espagne est **Paris** »). Le DPO, avec
1 064 paires de réponses de gros modèles, préfère la réponse choisie 59 % du
temps sur des paires inédites, sans effet mesurable à l'examen. À 125M, on
échange de la forme contre du fond : c'est le plafond de cette taille.

    python data/download_comparia.py
    python sft.py --checkpoint checkpoints/run_150m/best.pt \
        --data data/sft/comparia.json data/sft/conversations.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 300 --epochs 2 --out checkpoints/carl_v3
    python sft.py --checkpoint checkpoints/carl_v3/best.pt \
        --data data/sft/comparia_reparation.json data/sft/conversations.json \
        --extra data/identite_carl.json --extra-repeat 10 --enchainer 300 --epochs 1 --lr 3e-5 --out checkpoints/carl_v4
    python dpo.py --checkpoint checkpoints/carl_v4/best.pt --out checkpoints/carl_dpo
    python examen.py checkpoints/carl/best.pt checkpoints/carl_v4/best.pt checkpoints/carl_dpo/best.pt

### Carl v5 et v6 : calculatrice, identité, lecture de Wikipédia

Diagnostic préalable (40 faits simples) : le modèle de base n'en complète que
20, Carl v4 en retrouve 28 en conversation si l'on prend toujours le mot le
plus probable (23 en tirant au hasard à 0,7). Le SFT ne détruit donc rien ;
la limite est ce que le pré-entraînement a retenu.

- **A. Identité enrichie** : 296 conversations au lieu de 56, formulations
  combinées, sans celles de l'examen.
- **B. Calculatrice** (`outils.py`) : Carl écrit « [calc: 17*23 = », le
  programme calcule et insère le résultat.
- **C. Wikipédia** (`rag.py`) : recherche BM25 sur le début des 1,36 million
  d'articles, puis lecture (PIAF, SQuAD v2 traduit, passages hors sujet).

| | nom | créateur | savoirs | conduite | calcul |
|---|---|---|---|---|---|
| Carl v4 | 10/10 | 6/10 | 28/40 | 4/4 | 1/10 |
| Carl v5 (A + B) | 9/10 | 9/10 | 25/40 | 4/4 | 10/10 |
| **Carl v6 (A + B + lecture), par défaut** | 9/10 | 9/10 | 25/40 | 4/4 | 10/10 |
| Carl v6 + Wikipédia | 8/10 | 9/10 | 21/40 | 4/4 | 10/10 |
| Carl v7 (pièges réalistes) + Wikipédia | 8/10 | 9/10 | 17/40 | 4/4 | 10/10 |

B et A marchent. C non : la recherche ne ramène le bon passage en premier que
pour 21 des 40 questions (le début d'article ne contient pas toujours la
réponse, le dump perd les dates, les questions de record n'ont pas de nom
propre à chercher). Carl lit bien quand le passage contient la réponse (17 sur
21), mais ne sait pas reconnaître un passage voisin qui ne la contient pas :
il y pioche une mauvaise réponse. Ni le score de la recherche ni la confiance
de Carl ne permettent de trier (ses réponses de mémoire justes et fausses ont
la même assurance). Des exemples de « pièges » réalistes (v7) lui ont surtout
fait oublier des faits. La recherche est donc désactivée par défaut
(`--wikipedia` pour l'essayer) ; ne pas chercher quand la question s'adresse
à Carl ou contient des chiffres reste indispensable si on l'active.

    python data/calculs.py && python data/identite_carl.py
    python rag.py --construire && python data/lecture.py
    python sft.py --checkpoint checkpoints/carl_v4/best.pt \
        --data data/sft/comparia_reparation.json data/sft/conversations.json data/sft/calculs.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 300 --epochs 1 --lr 3e-5 --out checkpoints/carl_v5
    python sft.py --checkpoint checkpoints/carl_v5/best.pt \
        --data data/sft/lecture.json data/sft/comparia_reparation.json data/sft/conversations.json data/sft/calculs.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 300 --epochs 1 --lr 3e-5 --out checkpoints/carl_v6
    python chat.py --checkpoint checkpoints/carl_v6/best.pt

### Génération de Carl : relances, boucles, longueur

Trois réglages de `chat.py`, sans réentraînement, mesurés sur Carl v6 :

| | avant | après |
|---|---|---|
| relances justes (« et de la France ? », 6 cas) | 1/6 | 4/6 |
| réponses qui tournent en boucle (8 questions) | 4/8 | 0/8 |
| longueur moyenne | 720 caractères | 476 |

- pas de répétition d'une suite de 3 tokens dans la réponse (suspendu pendant
  un appel à la calculatrice, où Carl recopie volontairement l'opération) ;
- l'échange précédent n'est montré que pour une relance (commence par « et »,
  « pourquoi »..., ou 3 mots au plus), jamais pour une question adressée à
  Carl ; l'historique garde les appels bruts à la calculatrice, sinon Carl
  imite « 12 × 12 = 144 » et invente la suite ;
- 150 tokens au plus par réponse au lieu de 300.

L'examen (une question à la fois) est inchangé à la notation près : deux
réponses fausses dans les deux versions (« l'eau bout à 0 °C... à 100 % »,
« le trioxyde d'or ») étaient comptées justes par la recherche de sous-chaîne.

### Niveau 4 : Carl sur Gemma 4 (`carl_gemma/`)

Le même Carl (nom, créateur, intentions) greffé sur Gemma 4 E4B (Google,
Apache 2.0) par LoRA, sur le Mac avec MLX : 0,09 % des poids entraînés, un
adaptateur de 28 Mo. Données : 50 conversations d'identité honnêtes pour ce
cerveau (« créé par Kevin Pacini à partir de Gemma 4 ») et 100 réponses
générales écrites par Gemma lui-même, pour qu'il ne change que d'identité.

`carl_gemma/examen.py` note l'identité (5 questions jamais vues) et les savoirs
(10 questions à réponse connue), 2 tirages chacune :

| | identité | savoirs |
|---|---|---|
| Gemma 4 d'origine | 0/10 | 19/20 |
| LoRA fort (lr 1e-4, 16 couches, 400 pas) | 8/10 | 11/20 : oubli, 17 x 23 = 1789 |
| **LoRA léger (lr 3e-5, 8 couches, 250 pas)** | **9/10** | **20/20** |

    python carl_gemma/identite.py && python carl_gemma/distiller.py --n 100
    python carl_gemma/entrainer.py && python carl_gemma/examen.py
    python carl_gemma/chat.py

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

# Étape 5 : SFT puis discussion
python data/download_sft.py
python sft.py --checkpoint checkpoints/run_150m/best.pt
python chat.py --checkpoint checkpoints/sft/best.pt

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
