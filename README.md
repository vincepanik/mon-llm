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
    python chat.py --checkpoint checkpoints/carl_v14/best.pt

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
    python chat.py --checkpoint checkpoints/carl_v14/best.pt

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

### Étape D : conversations écrites par Claude (Carl v8)

1 234 conversations (1 467 échanges) écrites par Claude : réponses courtes et
simples, relances, « je ne peux pas le savoir », petites tâches. Aucune
question de l'examen ni leur réponse, même indirecte. Gardées en local
(`data/claude/lot_*.txt`, hors du dépôt public) : les conditions d'Anthropic
encadrent l'entraînement d'autres modèles sur les réponses de Claude.

| | nom | créateur | savoirs | conduite | calcul | relances | longueur |
|---|---|---|---|---|---|---|---|
| Carl v6 | 9/10 | 9/10 | 23/40 | 4/4 | 10/10 | 4/6 | 476 car. |
| **Carl v8** | 9/10 | 8/10 | 24/40 | 4/4 | 10/10 | 3/6 | **264 car.** |

La forme change (réponses deux fois plus courtes, sans faux gras), pas le
fond. Test du « perroquet » sur cinq faits de ces conversations :

| | question apprise | reformulée | jamais vue |
|---|---|---|---|
| Carl v6 | 0/5 | 1/5 | 0/5 |
| Carl v8 | 4/5 (2 mot pour mot) | 2/5 | 0/5 |

Il récite ce qu'il a vu, généralise mal aux reformulations (« Germinal, c'est
un roman de quel auteur ? » -> Maupassant, confondu avec Bel-Ami, même année
1885) et rien du tout aux faits voisins (« la capitale de la Slovaquie est
Prague »). À 125M, le SFT enseigne une manière de répondre ; les
connaissances viennent du pré-entraînement.

    python data/claude/assembler.py
    python sft.py --checkpoint checkpoints/carl_v6/best.pt \
        --data data/sft/claude.json data/sft/claude.json data/sft/claude.json data/sft/calculs.json \
               data/sft/comparia_2000.json data/sft/conversations.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 200 --epochs 1 --lr 3e-5 --out checkpoints/carl_v8
    python chat.py --checkpoint checkpoints/carl_v14/best.pt

### Étape E : une base de faits Wikidata (Carl v10)

Le test du perroquet l'a montré : le SFT n'apprend pas de nouveaux faits à un
125M. Plutôt que de lui faire tout retenir, Carl consulte une base, comme la
calculatrice : il écrit « [fait: Espagne | capitale = », le programme cherche
et insère « Madrid] », Carl recopie le résultat dans sa phrase, et l'appel
disparaît à l'affichage. Si la base ne sait pas, l'appel est effacé et Carl
répond de mémoire, comme avant.

- `data/wikidata.py` : 57 370 entités et 148 070 faits en français (pays,
  villes, personnes célèbres, livres, tableaux, musique, films, éléments,
  montagnes, événements, entreprises), les plus connues d'abord. 9 Mo, 2 min
  30 de téléchargement via QLever (le service officiel coupe les grosses
  requêtes). Données Wikidata sous licence CC0, puis tout est local.
- `faits.py` : la recherche, tolérante (« l'espagne », « Hugo » pour Victor
  Hugo, « Espangne », « mont Everest » pour Everest).
- `data/faits_sft.py` : 2 503 conversations fabriquées (réponse toujours
  juste), sans aucune entité citée dans l'examen.
- `data/outiller.py` : v9, entraîné avec, n'appelait jamais l'outil pour les
  capitales (0,3 à 3 % de probabilité) : les conversations de l'étape D en
  donnaient 35 de mémoire, et cette leçon l'emportait. v10 ajoute l'appel
  dans ces anciennes réponses quand la base confirme le texte d'origine.

| | examen : savoirs | 15 faits jamais vus (`examen_faits.py`) |
|---|---|---|
| Carl v8 | 24/40 | 5/15 |
| Carl v9 | 25/40 | 14/15 |
| **Carl v10** | **27/40** | **15/15** |

Nom, créateur, conduite et calcul inchangés (9, 8, 4/4, 10/10). « La capitale
de la Slovaquie est Prague » devient Bratislava, Einstein naît en 1879 et non
1890, Titanic n'est plus d'Orson Welles. Les échecs restants de l'examen sont
presque tous hors de la base (planètes, animaux, « combien de jours dans une
année ») : l'outil ne sait que ce qu'on y a mis.

    python data/wikidata.py
    python data/faits_sft.py
    python data/outiller.py data/sft/claude.json data/sft/comparia_2000.json data/sft/conversations.json
    python sft.py --checkpoint checkpoints/carl_v8/best.pt \
        --data data/sft/faits.json data/sft/claude_outils.json data/sft/claude_outils.json data/sft/calculs.json \
               data/sft/comparia_2000_outils.json data/sft/conversations_outils.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 200 --epochs 1 --lr 3e-5 --out checkpoints/carl_v10
    python examen_faits.py checkpoints/carl_v8/best.pt checkpoints/carl_v10/best.pt
    python chat.py --checkpoint checkpoints/carl_v14/best.pt

### Étape F : recherche par le sens dans Wikipédia

Point faible de l'étape C : la recherche par mots-clés (BM25) ne mettait le bon
passage en tête qu'une fois sur deux. Un petit modèle d'embeddings,
multilingual-e5-small (118M paramètres, licence MIT), résume chacun des 1,36
million de passages en 384 nombres ; on cherche ensuite le passage le plus
proche par le sens (77 min de calcul sur le Mac, 1 Go, hors ligne ensuite).

| recherche (`examen_rag.py`, 55 questions) | bon passage en tête | dans les 3 premiers |
|---|---|---|
| mots-clés (BM25) | 27 | 38 |
| **sens (embeddings)** | **34** | **46** |
| hybride (fusion des rangs) | 28 | 44 |

La recherche progresse, Carl non : avec Wikipédia, v10 tombe de 27 à 21/40 en
savoirs. Avec un passage sous les yeux, il oublie sa base de faits (« Égypte. »
pour la capitale de l'Égypte) et se laisse égarer par un passage hors sujet
(« Colisée (homonymie) » -> Roubaix). Aucun seuil ne trie les passages : bons
et mauvais ont tous une similarité entre 0,85 et 0,92 ; le modèle d'embeddings
trouve le sujet (« Albert Einstein »), pas la réponse. En donnant la priorité
à la base de faits, on remonte à 25/40, toujours sous les 27 sans document :
`--wikipedia` reste désactivé par défaut.

Découverte en passant : notre Wikipédia (wikimedia/wikipedia, 2023-11) a
perdu les dates écrites avec des modèles (« Albert Einstein, né le  à Ulm ») :
92 % des « né le » n'ont plus de date (10 824 contre 964 sur les 300 premiers
Mo). Et le téléchargement s'est arrêté à la limite de 5 Go fixée : environ un
tiers des articles manque, dont « Espagne » et « Tour Eiffel ». C'est la base
Wikidata qui donne les dates ; un futur pré-entraînement devra reprendre une
Wikipédia complète et mieux extraite.

    python rag.py --vecteurs
    python examen_rag.py mots sens hybride
    python examen.py --wikipedia checkpoints/carl_v10/best.pt

### Expérience : un 125M peut-il apprendre des faits ?

Le test du perroquet disait non : des questions-réponses, Carl ne retient que
les questions vues. Selon « Physics of Language Models » (Allen-Zhu et Li,
2023), un modèle ne sait ressortir un fait que s'il l'a lu sous de nombreuses
formes. `experience_faits.py` : 483 faits de Wikidata (aucun de l'examen),
chacun écrit sous 20 formes (« Bratislava est la capitale de la Slovaquie »,
« La Slovaquie a pour capitale Bratislava »...). Groupe A : ces phrases et
quelques questions ; groupe B : les phrases seulement, jamais une question.
Test : des questions formulées autrement, outils interdits.

| outils interdits | groupe A | groupe B (jamais interrogé) |
|---|---|---|
| Carl v10 | 18 % | 16 % |
| **Carl exp** (v10 + 2 epochs, 34 min) | **97 %** | **96 %** |

Il a appris : les faits du groupe B, jamais vus sous forme de question, sont
retrouvés à 96 % par des questions nouvelles. Ce qui manquait au test du
perroquet, c'était la répétition sous des formes variées, pas la place dans
le modèle.

Mais l'examen baisse (savoirs 27 -> 22/40) : les questions du groupe A, sans
outil, lui ont appris à répondre directement. Pour les capitales qu'il n'a pas
apprises, il n'appelle plus sa base et invente avec aplomb (« la capitale du
Portugal est Porto-Novo », « Le Petit Prince est une œuvre de d'Alembert »).
Carl v10 reste la version par défaut (v11 depuis, voir plus bas). La leçon pour un Carl plus grand : la
connaissance entre par un corpus où chaque fait revient sous de nombreuses
formes, au pré-entraînement plutôt qu'en questions-réponses.

    python experience_faits.py --donnees
    python sft.py --checkpoint checkpoints/carl_v10/best.pt \
        --data data/sft/exp_phrases.json data/sft/exp_questions.json data/sft/exp_questions.json \
               data/sft/faits.json data/sft/claude_outils.json data/sft/calculs.json data/sft/conversations_outils.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 200 --epochs 2 --lr 3e-5 --out checkpoints/carl_exp
    python experience_faits.py checkpoints/carl_v10/best.pt checkpoints/carl_exp/best.pt

### Carl v11 : les questions « tapées vite »

En vrai, « Capital du Portigal ? » : v10 n'appelait pas sa base (il n'avait vu
que des questions bien écrites) et inventait un véhicule électrique. v11 : un
tiers des questions de `data/faits_sft.py` sont tapées vite (fautes de frappe
dans le nom, sans accents ni majuscules, style télégraphique : « romeul lukaku
né quand », « continent moldavei »), la faute n'étant gardée que si la base
retrouve quand même la bonne réponse. Même recette que v10, 20 min.

| `examen_faits.py` | bien écrites | tapées vite |
|---|---|---|
| Carl v10 | 15/15 | 5/15 |
| **Carl v11** | **15/15** | **12/15** |

À l'examen, v11 a 24/40 en savoirs contre 27 pour v10, mais deux des
« pertes » étaient des réponses fausses que la notation par sous-chaîne
comptait justes chez v10 (« Napoléon III », « 577 millions de continents ») ;
une seule vraie perte (Everest -> « Elbrouz, en Iran »), dans le bruit d'un
entraînement à l'autre. Restent ratés : « slovaki » et « nietzche », trop loin
de l'orthographe pour la recherche floue de la base.

Correction (section suivante) : ce 12/15 était optimiste, car une partie des
questions « tapées vite » reprenaient mot pour mot des gabarits
d'entraînement. Sur un vrai jeu de test, écrit à l'aveugle : 20/37 (v10 : 9/37).

### Réparer la mesure, puis le chemin vers l'outil (septembre 2026)

Une équipe d'agents a relu tout le projet (`docs/feuille_de_route.md`) : le
modèle allait bien, mais la mesure et la tuyauterie autour de l'outil étaient
fausses par endroits. D'abord la mesure, pour ne pas régler les corrections
sur les questions qui servent à les juger.

**La mesure.**
- `notation.py` : notation en mot entier, sans les appels d'outils, sans
  accents, sur le début de la réponse. L'ancienne comptait juste « Napoléon
  III » (premier empereur), « 577 millions de continents » ou « le trioxyde
  d'or ». La conduite est notée sur « n'invente pas l'heure ou la météo », pas
  sur une liste de tournures d'excuse.
- Des fuites vers l'entraînement, trouvées par `tests/test_fuite_examen.py` :
  « Qui a peint la Joconde ? » ou « Quelle est la capitale du Canada ? »
  étaient mot pour mot dans compar:IA et oasst, « 37 au carré », « 12 fois 12 »
  et « 1+2 » dans les calculs, et « Madame Bovary », « Guernica » (du test des
  faits « jamais vus ») dans les conversations de l'étape D. Questions
  reformulées ou remplacées : c'est l'examen « v2 », dont les scores ne se
  comparent pas à ceux d'avant.
- Chaque réponse est gardée (`resultats/`, `python examen.py --renoter`) et
  `comparer.py` liste les questions gagnées et perdues entre deux versions,
  avec l'intervalle de confiance : 27/40 veut dire « entre 52 et 80 % ».
- `examen_conversation.py` : 11 conversations notées tour par tour (salut puis
  question, relance, merci puis relance) : l'examen posait chaque question
  seule, les bugs de mémoire y étaient invisibles.
- `examen_faits_test.py` : 50 questions tapées vite écrites par un agent qui
  n'avait pas vu les gabarits d'entraînement, réponses figées (vérifiées à
  l'écriture, et non recalculées par la base à chaque passage). 37 portent sur
  des faits jamais appris (le chiffre qui compte), 13 sur des faits vus bien
  écrits. Réserve : je les ai lues en les enregistrant ; ce jeu n'est plus
  parfaitement aveugle pour la recherche de faits.

| examen v2 | nom | créateur | savoirs | conduite | calcul |
|---|---|---|---|---|---|
| Carl v8 | 9/10 | 8/10 | 20/40 | 4/4 | 10/10 |
| Carl v10 | 9/10 | 8/10 | 25/40 | 4/4 | 10/10 |
| Carl v11 | 8/10 | 8/10 | 24/40 | 4/4 | 10/10 |

**Le chemin vers l'outil**, sans réentraîner Carl.
- `chat.py` : toute question de trois mots ou moins passait pour une relance,
  et Carl voyait l'échange précédent, souvent un simple « Hello Carl ! » ;
  « ou est nee marie curie » aussi (« ou » pris pour une relance). Une question
  courte n'est plus une relance que si elle ne nomme rien (« et où ? ») ou
  renvoie à ce qui précède (« sa population ? », « où est-il né ? ») ; la
  mémoire saute les politesses (« merci ! » entre deux questions) ; la pénalité
  de répétition ne porte plus sur les appels d'outils ; une réponse trop longue
  s'arrête à la dernière phrase complète.
- `faits.py`, réécrit. L'ancienne recherche répondait à presque tout : « Lune »
  -> la sortie de Transformers 3, « Napoléon Bonaparte » -> son neveu,
  « Kennedy » -> un inconnu, « Waterloo » -> la chanson d'ABBA, « Dordogne »
  -> un village. Principe : ne répondre que si l'identification est nette,
  sinon se taire (Carl répond alors de mémoire, comme sans outil). Les alias
  Wikidata (66 000 autres noms : « Chine », « Mona Lisa », « Napoléon
  Bonaparte ») ont été ajoutés à la base. Trois relectures successives par des
  agents (140 au total, chaque bogue signalé soumis à deux sceptiques) ont
  confirmé 41 bogues : 40 corrigés (les cas sont dans `tests/test_faits.py`),
  le dernier, des entités célèbres absentes de la base, est un chantier de
  données.

| banc de la recherche de faits | avant | après |
|---|---|---|
| cas relevés par les relectures (91) | 58 | 91 |
| 600 personnes célèbres, nom mal tapé : juste / faux | 581 / 9 | 583 / 1 |
| nom de famille seul : réponses fausses | 45 | 7 |
| nom de famille mal tapé : juste / faux | 26 / 40 | 254 / 15 |
| « prénom nom » absents de la base : réponses inventées | 58 / 461 | 4 / 461 |
| régions et départements absents : réponses inventées | 11 / 20 | 2 / 20 |

Les noms vraiment ambigus se taisent : « Clinton » (Bill ou Hillary ?),
« Bush », « François Ier » (trois souverains de même notoriété).

| Carl v11 | avant | après |
|---|---|---|
| examen v2 | 24/40 | 24/40 (aucune question gagnée ni perdue) |
| conversations (tours justes) | 20/25 | 21/25 |
| faits tapés vite, TEST (jamais vus) | 20/37 | 22/37 |
| faits tapés vite, déjà vus bien écrits | 8/13 | 12/13 |

La plupart des échecs restants au test ne viennent plus de la recherche : Carl
n'appelle pas l'outil (« shinning realisé par qui » -> « je ne peux pas… »)
ou choisit la mauvaise relation (« napoleon ville natale » -> la date de
naissance). C'est à l'entraînement de les corriger (prochain SFT). Et la base
a ses propres trous et erreurs (Los Angeles, Moïse ou la catastrophe de
Tchernobyl manquent ; Séville est rangée aux États-Unis, Hastings en 1187) :
un chantier de données.

    python examen.py checkpoints/carl_v10/best.pt checkpoints/carl_v11/best.pt
    python comparer.py resultats/carl_v10.jsonl resultats/carl_v11.jsonl
    python examen_conversation.py checkpoints/carl_v11/best.pt
    python examen_faits.py -q checkpoints/carl_v11/best.pt
    python -m pytest -q tests

### Carl v12 et v13 : un SFT regroupé, les débuts et fins de conversation

Un seul entraînement (depuis v8, 9 min) pour toutes les corrections de
données, chacune venue d'un défaut mesuré ou vu en vrai :
- `chat_format.py` : Carl n'apprend plus à écrire le résultat d'un outil
  (« Madrid] », « 391] »), que le programme insère : 17 à 20 % des tokens
  corrigés des données de faits et de calcul lui apprenaient à deviner des
  valeurs qu'il n'écrit jamais. Découpage identique à l'usage.
- `data/faits_sft.py` régénéré avec la nouvelle recherche, élisions corrigées
  (« d'Adele », « au Pecq »).
- `data/style.py` : de compar:IA et oasst, seules les réponses courtes, sans
  listes ni gras, au tutoiement (130 conversations compar:IA sur 2 000), et pas
  de question pour la base de faits répondue de mémoire (elle apprenait à
  sauter l'outil). En vrai, Carl répondait par des listes bancales (« 8. »
  coupé, « soupe de légumes aux légumes ») et passait du « tu » au « vous ».
- `data/politesses.py` : débuts et fins de conversation (« merci », « bonnes
  idées, merci ! », « au revoir », « salut, j'ai une question », « t'es là ? »).
  Carl répondait à un merci en se représentant (« Je suis Carl. Que puis-je
  faire pour toi ? »). v13 ajoute des saluts suivis d'une demande et des
  « salut » d'au revoir : v12 prenait « Bonsoir, j'aurais besoin d'un coup de
  main » pour une fin (« bonne continuation ! »).
- `examen_style.py` : 8 ouvertures, 8 clôtures, 8 demandes ouvertes, jamais vues.

Chaque version entraînée avec deux graines : l'écart entre les deux mesure le
hasard de l'entraînement (jusqu'à 2 points en savoirs).

| | v11 | v12 | v12 (graine 2) | **v13** | v13 (graine 2) |
|---|---|---|---|---|---|
| examen : nom / créateur / savoirs | 8 / 8 / 24 | 9 / 8 / 26 | 10 / 9 / 28 | **9 / 9 / 27** | 9 / 9 / 27 |
| conduite / calcul | 4 / 10 | 4 / 10 | 4 / 10 | 4 / 10 | 4 / 10 |
| conversations (tours justes /25) | 21 | 22 | 22 | **22** | 21 |
| faits tapés vite, test (/37) | 22 | 24 | 23 | **21** | 23 |
| ouvertures (/8) | 7 | 5 | 6 | **7** | 7 |
| clôtures (/8) | 3 | 7 | 7 | **6** | 6 |
| longueur moyenne (demandes ouvertes) | 250 | 204 | 148 | 270 | 230 |

v13 devient la version par défaut : clôtures bien meilleures sans perdre les
ouvertures, et rien ne recule au-delà du bruit entre graines (les faits tapés
vite : 21 et 23 pour v13, 24 et 23 pour v12). Restent les vraies ambiguïtés
(« Bon, j'y vais. Salut ! » pris pour un bonjour) et des réponses
générales encore faibles (« Que faut-il pour un gâteau au chocolat ? ») : à
125M, le fond vient du pré-entraînement.

    python data/faits_sft.py && python data/politesses.py
    python data/style.py data/sft/comparia_2000_outils.json data/sft/conversations_outils.json data/sft/claude_outils.json
    python sft.py --checkpoint checkpoints/carl_v8/best.pt \
        --data data/sft/faits.json data/sft/claude_outils_style.json data/sft/claude_outils_style.json \
               data/sft/calculs.json data/sft/comparia_2000_outils_style.json data/sft/conversations_outils_style.json \
               data/sft/politesses.json data/sft/politesses.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 200 --epochs 1 --lr 3e-5 --out checkpoints/carl_v13
    python examen_style.py checkpoints/carl_v11/best.pt checkpoints/carl_v13/best.pt

### L'aiguilleur : un « Système 1 » devant Carl

Idée venue de Jev (TypeSafe AI), un modèle qui ne génère pas de texte mais
répond à une question fermée par une décision et une probabilité. Presque
tous les ratés d'une vraie conversation avec v13 étaient des erreurs
d'aiguillage, pas d'écriture : « Bon, j'y vais. Salut ! » pris pour un
bonjour, « c'est une mauvaise histoire » -> « Très bien ! », « Cite-moi 3
capitales » -> un film de science-fiction.

`aiguilleur.py`, local et hors ligne : e5-small (déjà là pour Wikipédia) résume
le message en 384 nombres, une régression logistique apprise sur 1 817 exemples
(`data/aiguillage.py`, aucun des examens) le range dans une de 12 classes :
salut, au revoir, merci, critique, nom, créateur, capacités, heure, météo et
actualité, liste de capitales ou de pays, relance, autre. Sûr de lui (60 % au
moins, et quelques mots-clés en garde-fou : « quand est ne einstein » n'est pas
une question d'heure), il répond à la place de Carl là où une réponse fixe ou
un programme fait mieux : l'identité, l'heure de l'horloge du Mac, trois
capitales tirées de la base de faits, une excuse. Une relance : il montre
l'échange précédent. Sinon, Carl répond. 208 messages des examens sur 215 bien
aiguillés ; les 7 autres vont à Carl (confiance sous le seuil), sans dommage.

| Carl v13 | seul | avec l'aiguilleur |
|---|---|---|
| examen : nom / créateur / savoirs | 9 / 9 / 27 | **10 / 10** / 27 |
| conduite / calcul | 4 / 10 | 4 / 10 |
| conversations (tours justes) | 22/25 | 22/25 |
| ouvertures / clôtures | 7 / 6 | **8 / 7** |
| faits tapés vite, test | 21/37 | 21/37 |

Les savoirs, les faits et le style des réponses libres ne bougent pas : c'est
toujours Carl qui écrit (les recettes restent décousues). La note de conduite
compte désormais juste l'heure lue sur l'horloge. Actif par défaut dans
`chat.py` (`--sans-aiguilleur` pour l'enlever) ; les examens le prennent avec
`--aiguilleur`, pour mesurer Carl seul par défaut.

Deuxième version : l'aiguilleur **pose lui-même les questions de faits** à la
base (il reconnaît la relation par ses mots-clés, « capitale », « réalisé par »,
« né où », et prend le reste comme nom), et répond aux **« qui est… ? »** avec la
description Wikidata (54 000 descriptions ajoutées à la base : « Albert
Einstein (1879–1955) : physicien... », là où Carl disait « chimiste, inventeur
du premier ordinateur »). Carl ne fait plus que les phrases libres.

| Carl v13 | seul | avec l'aiguilleur |
|---|---|---|
| faits tapés vite, test (37 jamais vus) | 21 | **35** |
| faits tapés vite, déjà vus (13) | 10 | **13** |
| qui est… ? (`examen_qui.py`, 20) | 9 | **19** |
| conversations (tours justes /25) | 22 | 23 |
| examen : nom / créateur / savoirs | 9 / 9 / 27 | 10 / 10 / 27 |

Une analyse d'ensemble, domaine par domaine, est dans
`docs/analyse_capacites.md` : fiable pour tout ce qui a une réponse précise
(base, horloge, politesses), faible dès que Carl doit écrire ou raisonner seul.

Troisième version : **les fonctions de l'ordinateur** (`fonctions.py`), hors
ligne et sans risque, reconnues par des règles exactes avant le classifieur :
un minuteur ou un rappel (« préviens-moi dans 10 minutes pour sortir le
gâteau » : un message et une notification macOS, tant que la conversation
est ouverte), la batterie (`pmset`), les dates (« quel jour tombe Noël ? »,
« dans 10 jours on sera quel jour ? »), les conversions (« 5 miles en km »,
« 100 °F en °C ») et les calculs posés (« combien font 1 250 plus 3 780 ? »,
que Carl ratait : -3779). Rien ne modifie l'ordinateur ; Carl ne décide
jamais lui-même d'une action. Examens inchangés (conversations 23/25).

    python fonctions.py "quel jour tombe Noël ?" "5 miles en km"
    python aiguilleur.py --entrainer
    python aiguilleur.py --tester
    python examen_qui.py checkpoints/carl_v14/best.pt --aiguilleur
    python analyse_capacites.py checkpoints/carl_v14/best.pt
    python aiguilleur.py "Cite-moi 3 capitales d'Europe"
    python examen_style.py checkpoints/carl_v14/best.pt --aiguilleur

### Carl v14 : les évidences d'un enfant de trois ans

125M paramètres suffisent pour ce que sait un enfant de trois ans (environ
2 bits par paramètre) ; ce qui manque, ce sont les données. Personne n'écrit
« un chien a quatre pattes », parce que tout le monde le sait : Carl
répondait « une main a six doigts », « un siècle compte quatre ans », « le
mouton produit du bruit. Il s'agit d'un fromage à pâte molle ».

`data/evidences.py` : 213 évidences (pattes, cris et petits des animaux,
familles, milieux, couleurs, nombres du quotidien, parties du corps, objets,
contraires, origine des aliments, métiers, saisons), chacune écrite sous 20
formes, soit 4 260 phrases. La recette du groupe B de l'expérience des faits :
des phrases seulement, jamais une question, pour ne pas lui apprendre à
sauter sa base de faits. Les faits sont écrits par Claude et gardés en local
(`data/claude/lot_evidences.txt`). Aucun ne vient des examens : « le feu est
chaud » a été retiré, l'analyse des capacités le demande.

`examen_evidences.py` : 426 questions à l'aveugle, deux par fait, posées
autrement que dans les données (« Combien de pattes a un canard ? », « Que
fait un plombier ? ») ; `data/evidences.py` vérifie qu'aucune n'y figure.
Même recette que v13, les évidences ajoutées deux fois (13 min par graine) :

| | v13 | **v14** | v14 (graine 2) |
|---|---|---|---|
| évidences, Carl seul (/426) | 108 (25 %) | **343 (81 %)** | 339 (80 %) |
| évidences, avec l'aiguilleur | | **342** | 337 |
| examen : nom / créateur / savoirs | 9 / 9 / 27 | 9 / 8 / 25 | 10 / 8 / 29 |
| examen avec l'aiguilleur | 10 / 10 / 27 | 10 / 10 / 26 | 10 / 10 / 29 |
| conversations (tours justes /25), seul / aiguilleur | 22 / 23 | 22 / 22 | 22 / 22 |
| ouvertures / clôtures (/8) | 7 / 6 | **8 / 7** | 8 / 6 |
| listes ou gras / vouvoiement (/8) | 1 / 1 | **0 / 0** | 0 / 0 |
| longueur moyenne (demandes ouvertes) | 270 | 128 | 117 |
| faits tapés vite, test (/37) | 21 | 25 | 23 |
| qui est, seul (/20) | 9 | 9 | 9 |

v14 devient la version par défaut : les évidences passent de 25 % à 81 %, et
rien ne recule au-delà du bruit entre graines. Les réponses sont plus courtes.
Contraires, métiers, origines, milieux : presque tout juste. Mais :
- « Combien de pattes a un chien ? » -> six, pour les 22 animaux : cette
  tournure exacte s'est figée sur « six », alors que « Un chien, ça a combien
  de pattes ? » -> quatre. L'analyse, elle, a enfin « un insecte a 6 pattes »
  (v13 : 15).
- Les parties du corps et les objets se mélangent (« le corps utilise le nez
  pour voir », « pour couper du papier, on utilise le bois ») : il a appris la
  forme de ces phrases mieux que le lien entre chaque action et son objet.
- Les évidences débordent parfois hors sujet : « Tu connais une devinette ? »
  -> « L'inverse de plus jeune, c'est vieux » ; la banane mûre est « de
  couleur rouge » (la forme d'une réponse de couleur, sans le fait, absent des
  données).

    python data/evidences.py
    python sft.py --checkpoint checkpoints/carl_v8/best.pt \
        --data data/sft/faits.json data/sft/claude_outils_style.json data/sft/claude_outils_style.json \
               data/sft/calculs.json data/sft/comparia_2000_outils_style.json data/sft/conversations_outils_style.json \
               data/sft/politesses.json data/sft/politesses.json data/sft/evidences.json data/sft/evidences.json \
        --extra data/identite_carl.json --extra-repeat 2 --enchainer 200 --epochs 1 --lr 3e-5 --out checkpoints/carl_v14
    python examen_evidences.py checkpoints/carl_v13/best.pt checkpoints/carl_v14/best.pt

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
