# Ce que Carl sait faire (septembre 2026)

Carl v13 (125M paramètres), seul puis avec l'aiguilleur devant lui, sur 55
demandes variées et jamais vues (`analyse_capacites.py`), plus les examens.
Les réponses libres (explications, conseils, créativité) sont jugées à la
lecture ; le reste est noté automatiquement, corrigé à la main quand la
notation se trompe (signalé par *).

## En un tableau

| Domaine | Carl seul | Carl + aiguilleur | Qui répond | Verdict |
|---|---|---|---|---|
| Politesse (bonjour, merci, au revoir) | 4/4 | 4/4 | aiguilleur | fiable |
| Identité (nom, créateur, capacités) | 3/4 | 4/4 | aiguilleur | fiable |
| Faits de la base, bien écrits | 6/6 | 6/6 | Carl ou base | fiable |
| Faits tapés vite (examen, 37 jamais vus) | 21/37 | **35/37** | base | fiable |
| Qui est… ? (examen, 20) | 9/20 | **19/20** | base | fiable si la personne est dans la base |
| Heure, date, météo | 2/3 | 3/3 | aiguilleur (horloge du Mac) | fiable |
| Listes de capitales ou de pays | 2/3 | 3/3 | base | fiable |
| Calcul | 2/4 | 2/4 | Carl + calculatrice | fragile dès que la question sort du moule |
| Culture générale hors base | 1/5 | 1/5 | Carl | faible |
| Raisonnement simple | 0/4 * | 0/4 * | Carl | très faible |
| Langue française | 1/4 * | 1/4 * | Carl | faible |
| Explications | 2/3 | 2/3 | Carl | inégal |
| Conseils | 1,5/3 | 1,5/3 | Carl | inégal |
| Créativité | 0,5/3 | 0,5/3 | Carl | faible |
| Code | 0/1 | 0/1 | Carl | hors de portée |

## Ce qui marche

- **Tout ce qui a une réponse précise dans la base** : capitales, dates,
  auteurs, réalisateurs, populations, descriptions de personnes, même tapé
  vite (« einstien né quand » -> 14 mars 1879, « c koi la capitale du marok »
  -> Rabat). L'aiguilleur reconnaît la question et interroge la base lui-même ;
  la base se tait quand ce n'est pas net (homonymes, noms absents).
- **Les échanges courts** : saluer, remercier, dire au revoir, s'excuser d'une
  erreur, dire qui il est et qui l'a créé, donner l'heure réelle, refuser la
  météo sans l'inventer.
- **Quelques explications courtes et justes**, héritées des conversations de
  l'étape D : « Pourquoi la mer est salée ? », « C'est quoi la gravité ? »,
  « Comment préparer un entretien d'embauche ? ».

## Ce qui ne marche pas

- **La culture générale hors base** : « un insecte a 15 pattes », « la planète
  la plus grosse est Neptune », « 100 jours dans une année ». À 125M, le
  pré-entraînement a retenu peu de faits (l'expérience des faits l'a montré :
  il faut voir un fait sous vingt formes pour le retenir).
- **Le raisonnement**, même élémentaire : « 5 pommes, j'en mange 2 » -> « il
  te reste 5 pommes et une pomme est morte » ; « Paul est plus grand que
  Marie, qui est le plus petit ? » -> il répète la question (* la notation
  automatique la comptait juste, parce que « Marie » y figure).
- **La langue elle-même** : synonymes (« rapide » -> « lent »), conjugaison ;
  « chevaux » est juste, suivi de « et se dit joc » (*).
- **L'écriture longue** : recettes décousues (« la crêpe est un plat alsacien
  (...) des œufs, du lait, des œufs battus »), histoires sans fil, devinette
  absurde. Le poème d'automne tient en quatre vers, sans rimes.
- **Les calculs hors du moule** : « 1 250 plus 3 780 » -> « -3779 » (l'appel
  à la calculatrice est mal écrit), un petit problème (3 cahiers à 4 euros)
  mal posé.
- **Le code** : aucun.
- **Des personnes hors base** : la base garde les 30 000 personnes les plus
  connues ; Victor Schœlcher n'y est pas, et Carl invente (« peintre
  suisse-allemand »).

## Ce que ça dit pour la suite

Les gains de ces dernières étapes viennent presque tous de **ce qui entoure
Carl** : la base de faits, l'aiguilleur, l'horloge, la calculatrice. Là où Carl
doit écrire ou raisonner seul, il reste un modèle de 125M : il imite la forme
d'une bonne réponse sans le fond. Pistes, de la moins chère à la plus chère :

1. **D'autres réponses sûres par l'aiguilleur**, gratuites : calculs posés
   directement (« 1 250 plus 3 780 »), conversions d'unités, dates
   (« quel jour tombe Noël »), une base plus large (plus de personnes, les
   planètes, les animaux).
2. **Carl Gemma** (Gemma 4 E4B, déjà prêt dans `carl_gemma/`) derrière le même
   aiguilleur : il écrit, explique et raisonne bien mieux, en local.
3. **Un Carl plus grand** (350M, environ 50 $ de GPU) : un gain réel mais
   limité sur le fond ; surtout pédagogique.
