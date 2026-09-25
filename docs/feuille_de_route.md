# Feuille de route pour Carl (septembre 2026)

> **État (25 septembre 2026)** : actions 1 (réparer la mesure) et 2 (réparer le chemin vers l'outil) faites ; voir la section « Réparer la mesure, puis le chemin vers l'outil » du README. Prochaine : l'action 3 (un SFT v12 bien mesuré).

*Produite par 69 agents (10 explorateurs, fusion, un vérificateur sceptique par proposition, synthèse, critique), en lecture seule sur le dépôt. 81 propositions brutes, 56 après fusion, 46 retenues. Correction depuis : la perte des dates (92 %) et la coupure à 5 Go sont vérifiées et reportées dans le README.*

**Vue d'ensemble.** Ton gros modèle va bien. Ce qui coince surtout, c'est la tuyauterie autour : la règle qui décide ce que Carl voit de la conversation, la recherche dans la base de faits et la notation des examens. On peut presque tout réparer sur le Mac, sans GPU et sans argent. Les chantiers payants n'ont de sens que si tu décides un jour de refaire un pré-entraînement.

**Deux mises à jour par rapport à ce que dit le README :**
- **La v11 est terminée** : 24/40 en savoirs, 12/15 sur les questions tapées vite. Ce 12/15 est probablement trop beau (voir le bug n°3).
- **La Wikipédia téléchargée est bien plus abîmée qu'annoncé.** Environ **90 %** des « né le » ont perdu leur date, et non 9 %. En plus, le téléchargement s'est arrêté à 5 Go : environ un tiers des articles n'a jamais été vu. Des articles comme Espagne ou Tour Eiffel manquent dans l'index de recherche.

**Un rappel pour lire les chiffres.** Sur 40 questions, un écart de ±2 à 3 points peut venir du hasard d'un entraînement à l'autre. La plupart des gains ci-dessous tiennent dans ce bruit. Je le signale à chaque fois.

---

## 1. Bugs réels à corriger tout de suite

Tous se corrigent sur le CPU, sans réentraîner Carl, sauf mention contraire.

| # | Où | Ce qui se passe | Correction | Coût |
|---|---|---|---|---|
| 1 | `chat.py:57`, `:74`, `:77-81` (est_relance, a_montrer) | Toute question de 3 mots ou moins compte comme une « relance » (une question qui prolonge la précédente). En plus, « ou » sans accent est pris pour un mot de relance. Résultat : 10 des 15 questions tapées vite, dont « Capital du Portigal ? », montrent à Carl l'échange précédent. Souvent ce n'est qu'un « Bonjour ». Le README a mesuré que voir l'échange précédent fait tomber Carl de 6/6 à 2/6. | Ne jamais montrer un échange qui n'est qu'un salut ou une politesse. Retirer « ou » de la liste des mots de relance. Ajouter des tests. | 45 min |
| 2 | `examen.py:106` | La notation cherche le mot attendu n'importe où, même au milieu d'un autre mot. « au » (l'or) est trouvé dans « aussi », « 5 » dans « 577 millions », « napoléon » dans « Napoléon III ». Tu as déjà dû corriger deux fois à la main (README:196-198 et 360-363). | Chercher le mot entier (la fonction `contient()` existe déjà dans `examen_rag.py:31-33`). Noter seulement les ~150 premiers caractères, sans les appels d'outils. | 1-2 h |
| 2b | `examen.py:67` | « Dans quel pays se trouve la tour Eiffel ? » n'accepte que « paris ». La bonne réponse « En France. » est comptée fausse. | Accepter `['paris','france']`. | 1 min |
| 2c | `examen_faits.py:73-77` | Pour l'oxygène, la clé est « o », une lettre qu'on trouve partout : 2 points gratuits sur 30. Pour Microsoft, seul « allen » est accepté. | Comparer à la valeur complète de la base, en mot entier. | 30 min |
| 3 | `examen_faits.py:40-56` et `data/faits_sft.py:248-275` | Au moins 8 des 15 questions tapées vite reprennent mot pour mot un gabarit (un modèle de phrase à trous) utilisé à l'entraînement. Le 12/15 de v11 mesure donc surtout « reconnaît-il une phrase apprise ? ». Autres fuites : « Combien fait 37 au carré ? » est dans `data/sft/calculs.json`, alors que le commentaire `examen.py:92-93` affirme le contraire. « Qui a peint la Joconde ? » est aussi dans les données. | Ajouter `tests/test_fuite_examen.py` (sans torch). Écrire 20 à 30 questions tapées vite vraiment nouvelles. Corriger README:355-358 et les commentaires d'examen.py. | 2-3 h |
| 4 | `faits.py:78-97` (_candidats) | La base répond à presque tout, et souvent faux. Exemples : Lune → 2011 (le film Transformers 3), Apollo 11 → 2011 (Apollo 18), « Napoléon Bonaparte » → 1856 (le prince impérial), Mona Lisa → William Gibson, Hamlet → Laurence Olivier. La question d'examen sur la Lune est exposée. | Refuser les morceaux courts ou faits de mots vides. N'accepter un nom partiel que pour un nom de famille, « Napoléon » + chiffre romain, ou un début de titre. Refuser difflib (la comparaison approximative de textes) quand les chiffres diffèrent. Compléter `tests/test_faits.py`, qui existe déjà. | 3-5 h |
| 5 | `faits.py:95-97` | Quand Carl fait une faute de frappe, difflib compare d'abord à tous les noms, puis filtre la relation. « slovaki » et « nietzche » renvoient donc rien : ce sont 2 des 3 échecs restants de v11. | Filtrer par relation avant difflib, puis corriger mot par mot en dernier recours. **Seuil 0,85 sur les grands ensembles** : à 0,75, « Kevin Pacini » devient Kevin Bacon. | 2-4 h |
| 6 | `chat.py:167` | La pénalité de répétition s'applique aussi à « [fait: … \| capitale = » de la réponse précédente. Pour une relance (« et de l'Italie ? »), le premier « [ » est pénalisé mais pas son concurrent « La » : ça pousse Carl à ne pas appeler son outil. | `tok.encode(afficher(m["content"]))` (une ligne). | 5 min |
| 7 | `chat.py:84-93`, `:183` | Le blocage des trigrammes (interdiction de répéter une suite de 3 tokens) déforme les noms coupés en 3 tokens ou plus : Portigal → Portigol → Portiro. Ça touche environ 40 % des réponses longues. | N'interdire que les tokens qui commencent un mot, et exempter les mots de la question. | 1-2 h |
| 7b | `chat.py:272` | Arrivée à 150 tokens, la réponse est coupée en pleine phrase. | Couper à la dernière phrase complète et ajouter « (réponse coupée) ». | 20 min |
| 8 | `chat.py:233-236` | Le journal est nommé à la minute près et réécrit avec `write_text`. Relancer chat.py dans la même minute (pour comparer deux modèles, par exemple) efface la session précédente. | Suffixe `_2` ou secondes dans le nom. | 15 min |
| 8b | `outils.py:57` | `calculer("1 200*3")` et `calculer("7 ÷ 2")` renvoient « erreur » (séparateur de milliers à la française, symbole ÷). | Retirer les espaces entre groupes de chiffres, remplacer ÷ par /. | 15 min |
| 9 | `sft.py:107` et `utils.py:101-102` | Le premier SFT (l'entraînement par conversations) charge `run_150m/best.pt` sur MPS avec l'état de l'optimiseur, et ne le libère jamais : environ 1 Go gaspillé. | `map_location="cpu"`, puis `del ck`. | 10 min |

**Bugs de données, qui ne comptent qu'au prochain SFT :**
- **`chat_format.py:36-40`** : Carl est corrigé sur le résultat que le programme insère (« Madrid] », « 8023] »). Ça représente 17 % des tokens corrigés dans `faits.json` et 20 % dans `calculs.json`. Il apprend par cœur des valeurs qu'il n'écrit jamais lui-même. C'est aussi la cause de la rustine `chat.py:187-194`, qui ne sert qu'à la calculatrice.
- **`faits.py:118-121`** : des populations en double (« 23 262 544 et 23 317 031 habitants »).
- **`data/faits_sft.py:180, 185, 189, 234`** : fautes d'élision (« de Adele », « à Le Pecq »). Environ 2,6 % des réponses sont touchées.
- **`sft.py:116-135`** : 12 à 24 % de la validation sont des copies exactes de l'entraînement, et le commentaire des lignes 132-133 est faux. Sans conséquence tant que tu entraînes sur une seule epoch (un seul passage sur les données).

---

## 2. Gains rapides, gratuits, sur le Mac (moins d'une journée)

Classés par priorité. « Mac libre » veut dire : à lancer une fois l'entraînement en cours terminé.

| Prio | Quoi (en une phrase) | Pourquoi (chiffre attendu) | Coût | Première étape |
|---|---|---|---|---|
| **1** | **Sauvegarder chaque réponse d'examen et comparer deux versions question par question** (p32) | Aujourd'hui seul le total est gardé. Avec les réponses brutes, on peut renoter sans relancer Carl et lister les questions gagnées ou perdues. Et 27/40 veut en fait dire « entre 52 et 80 % » (intervalle de Wilson, une marge d'erreur honnête). | 2-3 h, 0 $ | Après `examen.py:131`, écrire une ligne JSON par question dans `resultats/`. Écrire `comparer.py`, puis le valider en refaisant la comparaison v10/v11 que tu as faite à la main. |
| **1** | **Notation en mot entier, examen « v2 »** (p33, bug n°2) | Scores justes à ±1-3 points près. Tu n'auras plus à corriger à la main. | 2 h + ~5 min de MPS par version | Voir le bug n°2. Repasser v8 à v11 et publier le nouveau tableau à côté de l'ancien. |
| **1** | **Corriger la relance** (p22, bug n°1) | Questions tapées vite prises pour des relances : de 10/15 à environ 1-3/15. Effet sur les réponses de Carl : de 0 à +4/15, **à mesurer d'abord**. | 45 min + 15 min de MPS | Test gratuit avant de coder : « Hello Carl ! » suivi des 15 questions, avec la mémoire normale puis avec `--memoire 0` (option déjà présente, `chat.py:286`). |
| **1** | **Recherche floue filtrée par relation** (p27, bug n°5) | Questions tapées vite : 12/15 → probablement 14/15 sans réentraîner (slovaki, nietzche). | 2-4 h | Petit script CPU : `slovaki/capitale` doit donner Bratislava, et `('Kevin Pacini','naissance')` doit donner None. |
| 2 | **Base de faits plus stricte** (p26, bug n°4) | Réponses fausses sur des mots courants : d'environ 71 à environ 15-20 sur 124 tests. Savoirs +0 à +1 (la Lune). Surtout, moins d'erreurs présentées avec l'autorité de l'outil en vraie conversation. | 3-5 h | Compléter `tests/test_faits.py`. Liste « doit réussir » : les 30 entités d'examen_faits, Hugo, Napoléon, Everest. Liste « doit rendre None » : Lune, Apollo 11, Carl, Paris|naissance. |
| 2 | **Test de fuite et nouvelles questions tapées vite** (p34, bug n°3) | Un vrai chiffre sur des fautes jamais vues. Probablement nettement sous 12/15, mais on ne peut pas savoir combien avant de mesurer. | 2-3 h | `tests/test_fuite_examen.py` (lecture par ast et json, moins d'1 s). |
| 2 | **Tests CPU de non-régression** (p39, niveau CPU seulement) | Les erreurs de notation et de relance ne peuvent plus revenir en silence. | 1-2 h | Sortir la notation dans un `notation.py` sans torch, et ajouter `tests/test_notation.py` avec 10 réponses pièges. |
| 3 | **Examen « pièges »** : 20 entités inventées et 20 vraies entités absentes de la base (p35) | Mesure pour la première fois le taux d'invention. Prévision : 90-100 % pour v10 et v11. L'examen servira surtout à mesurer les corrections futures. | 3-4 h + 2 min de MPS | Vérifier sur CPU que `faits.chercher` renvoie None pour chaque entité, puis noter en trois cas : juste, abstention (« je ne sais pas »), faux. |
| 3 | **Décomposer chaque échec de fait** (p37) : pas d'appel, mauvais arguments, base muette, mauvaise recopie | Dit tout de suite quoi réparer. Donne aussi la probabilité d'appeler l'outil, plus sensible que 12/15 contre 15/15. | 2-4 h + 2 min de MPS | Faire noter par `repondre` les appels tentés, **même ceux qu'il efface** (`chat.py:208-216`). Sans ça, les cas 3 et 4 sont invisibles. |
| 3 | **Grand examen Wikidata** : 300 questions pour régler, 300 pour tester (p36) | Détecte des écarts de 5 à 8 points au lieu de 20. Avec outils, on attend un plafond près de 100 %. Il sert surtout à mesurer le taux d'appel et le score sans outils. | 4-5 h + 25-40 min de MPS | Tirage stratifié par relation. Retirer toute entité déjà présente dans `data/sft/*.json`. Ne **pas** toucher à `faits.json`. |
| 4 | Petites corrections d'inférence (bugs 6, 7, 7b, 8, 8b, 9) | Confort en conversation. Environ 0 à l'examen. | 2-3 h au total | Écrire d'abord un fichier des 8 questions qui bouclaient (il n'existe pas), pour vérifier qu'elles restent à 0/8. |
| 4 | **Calibration** : Carl est-il moins sûr de lui quand il a faux ? (p38) | Remplace la phrase du README « même assurance » par un chiffre, l'AUROC (0,5 = pur hasard, 1 = tri parfait). Au-dessus d'environ 0,75, on pourrait forcer l'appel à l'outil quand Carl hésite. | 2-3 h + 10-20 min de MPS | `calibration.py` : générer la réponse, puis refaire une passe (comme `dpo.py:52-61`) pour lire la probabilité du premier token de la réponse. |
| 5 | **Situer Carl face aux petits modèles publiés** (p10, p40) : bits par octet, HellaSwag-fr | Pédagogique. Les bits par octet ne dépendent pas du tokenizer (~0,82 attendu pour Carl). Ne suffit pas à décider seul d'un modèle 350M. | 3-4 h, ~10 Go à télécharger | Ne comparer qu'à l'intérieur d'une même famille (SmolLM2 135M/360M, Pleias 350m/1.2B), en notant pour chaque modèle combien de tokens il a vus. |

---

## 3. Chantiers moyens (quelques jours, un SFT sur le Mac, 0 $)

L'idée : **regrouper en un seul SFT v12** tout ce qui touche les données. Chaque SFT fait bouger l'examen de ±3 points, donc en enchaîner plusieurs brouille tout.

| Chantier | Ce que c'est | Gain honnête | Coût |
|---|---|---|---|
| **Masquer le résultat d'outil** (p20) | Carl n'est plus corrigé sur « Madrid] ». Le résultat est encodé comme à l'utilisation, ce qui permet de supprimer la rustine « ]. ». | Savoirs 0 à +2, dans le bruit. La cohérence entre entraînement et utilisation est réelle. **Ne répare pas** « il n'appelle pas l'outil ». | 1-2 h de code + 1 SFT. Adapter `dpo.py` à part. |
| **Données de faits propres** (p43, et la régénération après p26/p27) | Une seule population, des élisions correctes, plus de « langues en Guinée ». | Environ 0 à l'examen. Environ 2,6 % des exemples deviennent en français correct. | 1-1,5 h |
| **Alias Wikidata** (p28, en partie) | Ajouter les autres noms (Hollande, É.-U., Apollo 11 bien reconnu), rangés **après** les noms officiels et filtrés. | Corrige des erreurs réelles de la recherche. examen.py : 0 à +2 seulement. | 1-2 jours. Écrire un `examen_culture.py` **avant** le changement. |
| **Refaire l'expérience des faits sans casser l'outil** (p25, p52) | Deux essais. (a) La recette carl_exp sans les questions du groupe A. (b) La même avec des demandes toujours génériques. Rappel : 63 % des « phrases » nommaient déjà l'entité, ce qui apprend aussi à répondre de mémoire. | Au mieux on retrouve 26-27/40 avec un rappel sans outil de 40 à 90 %. Leçon pédagogique claire sur ce qui fait perdre le réflexe d'outil. | 2-3 essais de ~30 min sur le Mac |
| **Chemin « la base ne sait pas »** (p19) | Quand l'outil ne trouve rien, répondre « je ne trouve pas » au lieu d'effacer l'appel. **Sans** la suggestion « tu voulais dire » à 0,6 : elle proposerait Kevin Bacon pour Kevin Pacini. | Ne fait rien sur « Portigal » (la base trouve déjà Lisbonne). Utile seulement si l'examen pièges (p35) montre que Carl appelle souvent l'outil. | 1 h (version programme) à 5 h (version apprise) |
| **Rééquilibrer compar:IA** (p23, niveau 1) | compar:IA fait 59 % des caractères de réponse et apprend le style « Voici quelques points clés : 1. ** ». Filtrer pour garder les réponses de 450 caractères ou moins, sans markdown. | Moins de gras et de listes. Savoirs environ 0. Risque de -1 en conduite. **Ajouter d'abord la mesure** (longueur, taux de markdown). | 1-1,5 h + 1 SFT |
| **Capacités et mémoire** (p56) | Plus de 60 variantes de « que sais-tu faire » dans `data/identite_carl.py`. « Ma première question » est interceptée par chat.py, un 125M ne peut pas l'apprendre. | Ces deux défauts disparaissent. Échec vu sur v8 : le revérifier sur v11. | ~3 h + 1 SFT |
| **Données sans Claude** (p54) | D'abord : réentraîner v8 avec les 1 289 réponses courtes sous licence libre déjà présentes. Gemma seulement si ça ne suffit pas. | Au mieux la parité, mais avec un Carl commercialement propre. | 0,5 j (essai) à 1,5 j + génération nocturne |
| **Diagnostic de la recherche Wikipédia (RAG)** (p45 → p48 → p51/p46/p47) | D'abord `examen.py` avec et sans `--wikipedia`, en comparant les lignes ✗ (10 min). Ensuite, mesurer le plafond du top 20 e5 avant d'essayer un reclasseur (bge-reranker-v2-m3, Apache 2.0). | Au mieux, le RAG cesse de nuire : 25 → 26-29/40. Presque aucune chance de dépasser nettement 27. | Une demi-journée à 2 jours selon la suite |
| **Routeur date et heure** (p30) | Seulement pour « quel jour sommes-nous », « quelle heure est-il ». Pas pour « quel jour est né X ». | Supprime les fausses dates apprises de compar:IA. 0 à l'examen. | 1-2 h |
| Wiktionnaire (p31) | D'abord mesurer 20 définitions de mémoire. Poursuivre seulement si Carl fait 10/20 ou moins. Bonne URL : `kaikki.org/dictionary/downloads/fr/fr-extract.jsonl.gz`. | 0 à +8/20 sur un volet à créer. | 2-3 jours si on va au bout |

---

## 4. Gros chantiers (GPU loué, argent)

**À retenir :** rien de tout ça n'améliore la Carl d'aujourd'hui. Ce sont des briques pour un **futur pré-entraînement**, et le gain à l'examen est faible ou incertain (ses défauts viennent surtout du post-training et des outils).

| Chantier | Coût chiffré | Gain honnête |
|---|---|---|
| **Préparer train.py** (p09) : taux qui descend jusqu'à 0 (D2Z) ou WSD (palier puis descente), validation fixe par source, perte moyennée sur les micro-lots | Gratuit : ~0,5 j de code, testé sur `debug_mac.py`. DDP (plusieurs GPU) à reporter. | D2Z : environ -0,02 à -0,04 de perte. La validation fixe rend enfin les comparaisons fiables. |
| **FineWiki à la place de notre Wikipédia** (p01, p12) | Données : 0 $, 1-2 h de code, 5-8 Go. Utile seulement avec un nouveau run (~18 $) ou un recuit (~5 $). | Fini les « né le  à Ulm », environ 1,2 million d'articles en plus. Test à trous (compléter une phrase) : +1 à +3/40. Examen : environ 0. |
| **Banc de mesure du modèle de base** (p03) : 1 000 faits à trous et perte par source | Gratuit sur le Mac (les frontières des sources se retrouvent dans le `val.bin` actuel). Comparaisons de corpus : 2-3 $ chacune au 125M. | Rien sur Carl. Évite un run raté à 18 $ ou plus. |
| **Recuit** (p02) : court entraînement supplémentaire sur les 148 000 faits en phrases, à partir de `latest.pt` | 5-12 $ + 3-5 jours humains + toute la chaîne SFT à refaire | Examen 0 ±2 (surtout pédagogique). Rappel sans outil peut-être 30-70 %. Pour couvrir plus d'entités, il suffit d'étendre `faits.json`, sur CPU. |
| **Carl-350M** (p08) | ~45-55 $ (secure) ou ~30-35 $ (community), ~42-48 h sur une 5090 ; + SFT | Perte 2,51 → ~2,35-2,40. Savoirs +1 à +4, dans le bruit. C'est le vrai levier pour le fond, mais cher et lent à vérifier. |
| Balayage du taux d'apprentissage + QK-norm (p16) | ~3-4 $ + un run complet | 0 à -0,02 de perte. |
| Muon (p14), un autre optimiseur | ~10 $ de comparaisons + un run | 10-20 % de GPU en moins sur un futur run. |
| Chargeur de données sans remise (p17) | 2-3 h, gratuit | 0 à -0,01, sous le bruit. Hygiène, rien de plus. |
| HyperCloning (p13) : faire grandir le 125M | Test jouet gratuit. Vraie validation : 25-50 $ | Incertain. Seulement si le 350M est décidé. |
| Forme 30×576 façon SmolLM2 (p18) | 1-2 $ pour 2 runs courts, puis 20-25 $ | 0 à -0,02. |
| Corpus v2 au-delà de 20 G tokens (p12) | ~130 $ de GPU | Hors budget pour l'instant. |
| Carl 1B (p15) | 140-360 $ | **Déconseillé** (voir section 5). |

**Si tu refais un jour un pré-entraînement**, le paquet cohérent serait : FineWiki + D2Z/WSD + validation fixe + chargeur sans remise + banc de mesure (p03), avec ou sans passage à 350M. Soit environ 20 $ (125M) ou 50-60 $ (350M), plus la chaîne SFT à refaire.

---

## 5. Ce qu'il ne faut PAS faire

| Idée | Pourquoi non |
|---|---|
| Données synthétiques « manuel scolaire » (p04) | « 365 jours » apparaît déjà 2 141 fois dans le corpus, « planète rouge » 1 917 fois. Il ne manque pas de données. Et SYNTH est en format question-réponse, qui casse le réflexe d'outil. |
| Vikidia et Common Corpus pour « simplifier » (p05) | Mêmes comptages : les faits simples sont déjà vus des centaines de fois. Vikidia pèserait environ 0,3 % du corpus. |
| Préfixer chaque document par sa source (p06) | À 600M, MeCo mesure +0,2 point seulement, et une étiquette grossière *dégrade* les résultats. En plus, il faudrait tout réentraîner. |
| Coller l'espace aux nombres dans le tokenizer (p07) | ~1,4 % de tokens en moins, pour un effet invisible. Et un même nombre aurait deux écritures différentes. |
| Identité « Carl ≠ Kevin » avec un SFT dédié (p21) | Le défaut n'est pas prouvé. Le gain tient dans le bruit (1 question sur 10). À intégrer au v12 si besoin, pas en SFT séparé. |
| DPO sur les erreurs de Carl (p24) | Le premier DPO n'a presque pas bougé. Il n'y a presque plus de marge sur les volets mesurés, et deux des trois échecs restants viennent de la recherche. |
| Routeur qui impose la réponse de la base (p29) | Face à v11 : +1/15 seulement, avec de faux déclenchements (« Qui est né à Paris ? » → Paris Hilton). Mieux vaut corriger la recherche (p27). |
| Réapprendre la lecture de passages avec pièges (p49) | Déjà testé en v7 : savoirs 25 → 18/40. |
| Lire en citant la phrase source (p50) | Le filtre rejette 46 % des bonnes lectures et laisse passer les passages voisins trompeurs. |
| Forcer l'appel dès que « [ » est dans le top-10 (p53) | Les échecs restants viennent de la recherche. Et « Carl|fondateur » renvoie Carl Karcher. |
| Carl 1B (p15) | 140-360 $ pour un gain dans le bruit. Gemma 4 existe déjà en local. |
| Superlatifs calculés et relation « c'est quoi » (p28) | Ça revient à coder en dur les réponses de l'examen. En plus, `interdits_examen` les exclut du SFT. |
| Seuil difflib à 0,75 partout, clé phonétique (p27) | Faux positifs graves : Kevin Pacini → Kevin Bacon. |
| Avertissement « réponse longue = invention » (p55) | Pas validé : « Explique la photosynthèse » est une question courte qui appelle une réponse longue et juste. |
| Test de McNemar comme feu vert automatique (p39) | Trop faible sur 10 à 40 questions (6 perdues contre 1 gagnée ne déclenche rien). Afficher simplement la liste des questions qui changent. |
| Ablations sur des modèles jouets de 0,8M ou 20M (p03, p18) | Leurs résultats ne se transposent pas au 125M, et ils ne retiennent aucun fait. |
| DDP multi-GPU, normalisation de la perte du SFT par défaut (p09) | Pas avant qu'un gros run soit décidé. La normalisation risque d'affaiblir le réflexe d'outil. |
| Mettre 1 million d'entités dans les poids (p02) | 10 à 20 expositions par fait, donc un rappel faible. Agrandir `faits.json` fait mieux, gratuitement. |
| Relancer le RAG Wikipédia sans filtre | Déjà mesuré : 21 puis 25/40, contre 27 sans. |

---

## 6. Ma recommandation : les 3 prochaines actions

1. **Réparer la mesure** (environ une demi-journée, CPU, 0 $) : p32, p33 et p34. Sauvegarde des réponses en JSONL, notation en mot entier (tour Eiffel, oxygène compris), test de fuite, et une vingtaine de questions tapées vite vraiment nouvelles. Une fois le Mac libre, repasser v8 à v11.
   *Pourquoi en premier :* toutes tes décisions (v9 contre v10, v10 contre v11) reposent sur des écarts de 1 à 3 points, et la notation actuelle en fausse au moins deux par version.

2. **Réparer le chemin vers l'outil, sans réentraîner** (environ une journée, CPU, 0 $) : p22, p27 et p26, plus les petits bugs 6 et 8.
   - Relance : retirer « ou », ne pas montrer un simple salut.
   - Recherche : filtrer par relation avant difflib, avec le garde-fou Kevin Pacini.
   - Base : refuser les faux positifs (Lune, Apollo 11, Napoléon Bonaparte).
   - Attendu : questions tapées vite ~14/15, et moins d'erreurs « certifiées par l'outil ».
   - Avant de coder la relance, faire le test gratuit « Hello Carl ! » + questions, avec et sans `--memoire 0`.

3. **Préparer un seul SFT v12, bien mesuré** (2 à 3 jours) :
   - D'abord les petits instruments : examen pièges (p35) et décomposition des échecs (p37).
   - Puis un v12 qui regroupe le masquage du résultat d'outil (p20), des données de faits régénérées et propres (p43) et, si p35 le justifie, le chemin « inconnu » (p19).
   - On garde v12 seulement s'il ne recule sur aucun volet de l'examen v2.

**En parallèle, pendant que le GPU est pris** (CPU seulement) : lire en streaming environ 2 000 articles de FineWiki, en ne gardant que les colonnes `text`, `title` et `wikidata_id`. Refaire les comptages « né le <date> » contre « né le  ». Ça chiffre la réparation de Wikipédia avant de décider quoi que ce soit de payant.

**Mes incertitudes :**
- Les gains de p22 et p27 sont estimés, pas mesurés.
- Le vrai score « tapées vite » sur des fautes nouvelles est inconnu : il peut être bien plus bas que 12/15.
- Personne n'a encore vérifié sur v11 le défaut « comment peux-tu m'aider » (vu sur v8).

---

**Petit lexique**
- **SFT** : entraînement sur des conversations.
- **RAG** : chercher un passage de Wikipédia et le donner à lire à Carl.
- **Glouton** : Carl prend toujours le mot le plus probable.
- **Intervalle de Wilson** : la marge d'erreur d'un score.
- **McNemar** : test qui compare deux versions sur les mêmes questions.
- **AUROC** : à quel point une probabilité sépare les bonnes réponses des fausses (0,5 = hasard, 1 = parfait).
- **D2Z / WSD** : deux façons de faire descendre le taux d'apprentissage.
- **Bits par octet** : une perte qui ne dépend pas du tokenizer.

Fichiers concernés (lecture seule, rien n'a été modifié) : `/Users/vincepanik/Documents/mon-llm/chat.py`, `examen.py`, `examen_faits.py`, `faits.py`, `outils.py`, `chat_format.py`, `sft.py`, `utils.py`, `data/faits_sft.py`, `data/download.py`, `tests/test_faits.py`, `tests/test_chat.py`.

---

# Critique de complétude (ce qui manque à la feuille de route)

Voici ce qui manque à la feuille de route, en 10 points. J'ai d'abord vérifié plusieurs de ses affirmations, en lecture seule et sur le CPU :
- **Wikipédia abîmée** : confirmé. Sur les 800 premiers Mo de `data/big/raw/wikipedia_fr.txt`, 2 995 « né le » sont suivis d'une date et 30 943 n'ont plus rien. Donc environ 91 % sont perdus, et non 9 %.
- **Téléchargement arrêté à 5 Go** : confirmé, le fichier fait 5,0 Go.
- **Articles absents de l'index** : « espagne » et « tour eiffel » ne sont pas dans `data/big/rag/titres.json`.
- **Fuite de « 37 au carré »** : confirmée, la question est bien dans `data/sft/calculs.json`.

### Ce qui manque

**1. Le « bruit de ±2-3 points » n'a jamais été mesuré, alors que la feuille de route écarte une dizaine d'idées en son nom.**
- Tous les SFT utilisent la même graine, 1337 (`sft.py:102`). Deux versions ne diffèrent donc que par leurs données.
- Le README se contente de l'affirmer : « un écart de 1 ou 2 points est du bruit » (README:109).
- Chez d'autres, la variance d'un réglage fin selon la graine est souvent plus grande qu'on ne croit (Dodge et al. 2020, https://arxiv.org/abs/2002.06305).
- **À faire** : refaire la recette v11 avec 2 autres graines. Ça coûte environ 2 × 20 min de MPS, plus environ 5 min d'examen par modèle.
- C'est ce qui chiffre le vrai seuil de décision. Ça doit passer avant tout rejet « dans le bruit ».

**2. La lignée SFT est empilée 7 fois, et le v12 prévu serait la 8e couche.**
- Les commandes du README l'enchaînent : run_150m → v3 → v4 → v5 → v6 → v8 → v10 → carl_exp (README:125-343). Selon le README, v11 suit la même recette, donc elle part sans doute de v10.
- Des oublis peuvent s'accumuler (Everest → Elbrouz), et personne ne sait refaire Carl d'un seul coup.
- **À faire** : ajouter un « v12 à plat », entraîné une seule fois sur le mélange final à partir de `carl_v4`, ou de `run_150m` avec compar:IA (2 h 40 de Mac, README:103). On le compare ensuite au v12 empilé.
- Gain inconnu. Mais c'est la seule façon de rendre Carl reproductible.

**3. Aucune fusion de poids, alors que c'est gratuit.**
- v11 et carl_exp sont tous deux des réglages fins de v10.
- On peut mélanger les poids : α·v10 + (1−α)·v11 pour α = 0,3, 0,5 et 0,7, calculé sur le CPU avec un script d'environ 20 lignes, puis environ 5 min de MPS par examen. C'est la méthode WiSE-FT / « model soups » (https://arxiv.org/abs/2109.01903, https://arxiv.org/abs/2203.05482).
- Espoir : garder les 27/40 de v10 et le 12/15 de v11 à la fois. Même chose entre v10 et carl_exp, pour garder une partie des 96 % de faits appris sans perdre le réflexe d'outil.
- Gain incertain : le mélange peut aussi tomber entre les deux.
- Ça trancherait aussi une question que la feuille de route laisse ouverte : quelle version par défaut ? Le README dit « v10 reste la version par défaut », mais ses commandes lancent `carl_v11` (README:97, 176, 232, 277).

**4. Ordre des tâches : les corrections p27 et p22 sont calées sur les échecs de l'examen lui-même.**
- « slovaki » et « nietzche » sont justement 2 des 3 échecs de v11. Les réparer puis annoncer 14/15, c'est apprendre l'examen.
- **À faire** : écrire d'abord les nouvelles questions tapées vite (p34). Les 15 anciennes servent alors seulement à régler.
- L'estimation 14/15 est aussi incohérente avec son propre garde-fou. J'ai mesuré avec difflib :

| Comparaison | Ressemblance (0 à 1) |
|---|---|
| slovaki / slovaquie | 0,750 |
| kevin pacini / kevin bacon | 0,783 |
| nietzche / friedrich nietzsche | 0,593 |
| nietzche / nietzsche | 0,941 |

- Avec le seuil de 0,85, « slovaki » échoue même après le filtre par relation. Il ne passe que si l'on descend vers 0,7, et seulement parmi les 197 entités qui ont une capitale (là, on obtient bien `['slovaquie', 'slovenie']`).
- « nietzche » demande une comparaison mot à mot.

**5. Le champ `type` des entités existe mais n'est pas utilisé. C'est un correctif plus simple et plus sûr que le bug n°4.**
- `faits.json` classe chaque entité en 11 types (personne : 29 894, ville : 7 162, pays : 197…).
- `faits.chercher` ne s'en sert pas. Résultats mesurés :
  - « Portugal | naissance » → « 5 juillet 1717 » (un roi du Portugal, trouvé par nom partiel) ;
  - « Carl | naissance » → « 23 mai 1707 » (Linné).
- **À faire** : n'accepter que les bons types pour chaque relation. Par exemple naissance, décès, lieu de naissance → personne ; capitale, monnaie, langue, continent → pays. Environ 30 min.
- Au passage : « portugal | capital » (sans e) renvoie None, car « capital » manque dans `RELATIONS` (`faits.py:27`). Si Carl recopie la faute de l'utilisateur, l'appel est effacé et Carl invente.

**6. Rien ne limite ce que Carl écrit dans l'appel d'outil.**
- Les données d'entraînement n'utilisent que 21 relations exactes (`data/sft/faits.json` : date 366, lieu de naissance 325… fondateur 30).
- Après « | », on pourrait n'autoriser que les tokens qui mènent à l'une d'elles. Le mécanisme existe déjà : `interdits` dans `chat.py:183`. Environ 1 h.
- Même idée pour le nom : un arbre des 55 859 noms normalisés, comme GENRE (https://arxiv.org/abs/2010.00904). Carl ne pourrait plus écrire « slovaki » et serait poussé vers « slovaquie ».
- Risque : il serait forcé vers un nom existant quand l'entité n'est pas dans la base (Kevin Pacini). Il faut donc une porte de sortie vers « inconnu ». Environ 4 à 6 h.
- C'est la vraie réparation de la catégorie « mauvais arguments » de p37.

**7. Pas de cache KV (la mémoire des calculs déjà faits) : chaque token recalcule tout le contexte.**
- Vu dans `chat.py:180-181` (`model.generate(contexte, 1)`) et `model.py:399-402`.
- Accélération estimée : ×2 à ×4 sur le MPS. Ce n'est pas mesuré.
- Coût : 3 à 4 h, plus un test d'équivalence des logits dans `tests/test_model.py`.
- Aucun gain de qualité, mais la stratégie « mesurer d'abord » repose sur des dizaines de passages d'examen : graines (point 1), mélanges (point 3), grand examen de 25-40 min. Ce chantier les rend moins chers.

**8. Aucun examen en plusieurs tours, et l'examen ne teste pas les réglages du chat.**
- `examen.py:125` pose chaque question seule. Le bug de relance (n°1) et `a_montrer` sont donc invisibles.
- Le chiffre « 6/6 contre 2/6 » repose sur une seule conversation rejouée 3 fois.
- L'examen coupe à 80 tokens (`examen.py:117`), le chat à 150 (`chat.py:278`).
- Les échecs en savoirs ne sont pas affichés (`examen.py:128`).
- **À faire** : un `examen_conversation.py` avec une dizaine de conversations scriptées (salut, question, relance, identité), notées tour par tour. Environ 2 h.

**9. On mesure les fausses réponses de la base (bug n°4), jamais ses trous.**
- capitale, monnaie, langue et continent n'existent que pour les 197 pays : aucune région, aucun État américain. La base ne couvre que 11 types d'entités.
- Quand elle ne trouve rien, l'appel est effacé et Carl répond de mémoire, sans outil (`chat.py:209-216`). C'est exactement là qu'il invente.
- **À faire** : 100 questions factuelles réalistes, hors examen. Pour chacune, noter si un appel correct aurait trouvé la réponse. Environ 2 h de CPU.
- Ce chiffre dit si les alias ou l'extension de la base (p28) valent 1 à 2 jours.
- Il n'y a que 3 vraies conversations dans `conversations/`. Il en faudrait plus pour fonder ces choix.

**10. Les licences ne sont pas passées en revue avant l'usage commercial.**
- La feuille de route ne traite que les données écrites par Claude (p54).
- Il manque une vérification, source par source :
  - PIAF et SQuAD, utilisés par `data/pieges.py` et `lecture.py`. SQuAD est en CC BY-SA 4.0 : ses obligations de partage à l'identique sont à vérifier.
  - Wikipédia (CC BY-SA), dans le pré-entraînement et dans l'index de recherche.
  - Les conditions d'Anthropic sur l'entraînement d'un modèle avec des sorties de Claude.
- **À faire** : un tableau « source → licence → sert au commercial oui/non » dans le README, avant de parler de reeve.os. Environ 1 h.

Aucun fichier du dépôt n'a été modifié. 