"""
Exemples étiquetés pour l'aiguilleur (aiguilleur.py) : quelle sorte de message
est-ce ? Écrits à la main, plus, pour « autre », de vraies questions des
données d'entraînement (faits, calculs, oasst).

    python aiguilleur.py --entrainer   # lit ces exemples, écrit checkpoints/aiguilleur.pt

Aucune question des examens (examen*.py) : elles servent à le mesurer
(tests/test_fuite_examen.py le vérifie).
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

EXEMPLES = {
    "salut": [
        "bonjour quelle belle journée", "salut, quelle journée !", "coucou, belle journée hein", "hello, quel beau matin", "le bonjour !", "bonjour à vous",
        "bonjour", "salut", "hello", "coucou", "bonsoir", "hey", "yo", "slt", "re", "bonjour carl", "salut carl",
        "hey carl", "coucou carl", "bonsoir carl", "hello carl", "bonjour à toi", "salut toi", "hello toi",
        "salut, ça va ?", "bonjour, comment vas-tu ?", "coucou, ça va ?", "hey, ça roule ?", "salut, la forme ?",
        "bonjour, j'ai une question", "salut, tu peux m'aider ?", "bonsoir, j'ai besoin d'aide",
        "hello, je peux te demander un truc ?", "coucou, tu as deux minutes ?", "salut, tu es dispo ?",
        "tu es là ?", "il y a quelqu'un ?", "allô ?", "carl ?", "tu m'entends ?", "bonjour bonjour",
        "wesh", "salut à toi carl", "bonjour, enchanté", "rebonjour", "me revoilà",
    ],
    "au_revoir": [
        "au revoir", "bye", "à plus", "a+", "à bientôt", "ciao", "à demain", "je dois y aller", "je te laisse",
        "allez, salut", "bon, je file, salut", "je m'en vais", "allez bonsoir", "bonne nuit", "bonne soirée",
        "bonne journée", "bon week-end", "à la prochaine", "on se reparle plus tard", "c'est tout pour aujourd'hui",
        "je vais y aller, bonne soirée", "bon j'y retourne, salut", "je dois partir, à plus", "à tout à l'heure",
        "allez, bonne nuit carl", "je me déconnecte", "fin de la conversation", "stop, on arrête là",
        "bye bye", "adieu", "je te dis à demain", "salut, à plus tard",
        "allez j'y vais, salut", "bon, je te laisse. salut !", "je file. salut !", "ok je m'en vais, salut",
    ],
    "merci": [
        "merci", "merci beaucoup", "merci bien", "super merci", "top merci", "génial merci", "parfait merci",
        "cool merci", "ok merci", "merci carl", "bonnes idées, merci", "c'est parfait, merci", "merci pour ton aide",
        "merci c'est gentil", "merci infiniment", "mille mercis", "thx", "merci, ça m'aide", "excellent, merci",
        "nickel merci", "génial", "super", "parfait", "top", "trop bien", "c'est exactement ça", "génial, c'est ce qu'il me fallait",
        "ça a l'air bon", "bien joué", "bravo", "c'est super, merci", "t'es génial", "merci pour les infos",
        "ok ça marche", "d'accord, merci",
    ],
    "critique": [
        "c'est faux", "tu te trompes", "ce n'est pas ça", "mauvaise réponse", "c'est nul", "n'importe quoi",
        "ta réponse n'a aucun sens", "tu racontes n'importe quoi", "c'est une mauvaise réponse", "pas du tout",
        "non, c'est faux", "tu as tort", "c'est pas terrible", "c'est une mauvaise histoire", "ton histoire est nulle",
        "ça ne veut rien dire", "tu n'as pas compris", "ce n'est pas ce que je t'ai demandé", "tu t'es trompé",
        "ta recette est bizarre", "c'est incompréhensible", "ce n'est pas du tout la bonne réponse",
        "tu dis des bêtises", "ça ne répond pas à ma question", "tu es nul", "c'est incohérent",
        "ton poème est nul", "c'est raté", "ta réponse est ratée", "ton texte est mauvais", "c'est complètement faux",
        "ta blague n'est pas drôle", "ça ne marche pas ta recette", "tu inventes", "c'est archi faux",
    ],
    # « ah oui c'est vrai » était pris pour un doute (« Pas forcément... »), « ah ok » pour un merci.
    "reaction": [
        "ah oui c'est vrai", "ah ok", "ah d'accord", "d'accord je vois", "je vois", "ah étonnant", "intéressant",
        "c'est intéressant", "ah bon d'accord", "c'est vrai", "effectivement", "ah oui", "incroyable", "waouh",
        "ok je vois", "c'est drôle", "ah c'est marrant", "ah je ne savais pas", "je ne savais pas", "ah tiens",
        "c'est fou", "ok d'accord", "ah oui en effet", "logique", "ça se tient", "ah mince", "oh", "wow",
    ],
    "doute": [
        "tu es sûr ?", "t'es sûr ?", "vraiment ?", "tu en es certain ?", "c'est vrai ?", "tu es certain ?", "sûr ?",
        "t'es sûr de toi ?", "ah bon ?", "tu es sûr de ta réponse ?", "sérieux ?", "c'est sûr ?", "tu confirmes ?",
        "tu me le garantis ?", "t'es certain ?", "vraiment vrai ?", "tu ne te trompes pas ?", "c'est bien ça ?",
    ],
    "humeur": [
        "très bien", "ça va bien", "bien et toi", "ça va, et toi ?", "très bien merci", "super bien", "ça va super",
        "pas mal", "tranquille", "nickel et toi ?", "bien bien", "la forme", "en pleine forme", "je vais bien",
        "bof", "pas terrible", "ça va pas trop", "je suis fatigué", "je suis un peu triste", "moyen", "je vais mal",
    ],
    "nom": [
        "comment tu t'appelles ?", "c'est quoi ton nom ?", "tu es qui ?", "qui es-tu ?", "quel est ton nom ?",
        "présente-toi", "tu peux te présenter ?", "t'es qui toi ?", "tu es un robot ?", "tu es une IA ?",
        "tu es humain ?", "à qui je parle ?", "tu es chatgpt ?", "comment dois-je t'appeler ?", "ton prénom ?",
        "tu es quoi exactement ?", "tu es un chatbot ?", "qui êtes-vous ?", "vous êtes qui ?", "tu es une personne ?",
        "tu es quel modèle ?", "tu t'appelles comment ?", "on t'appelle comment ?",
        "bonjour, qui es-tu ?", "salut, comment tu t'appelles ?", "hello, tu es qui ?", "coucou, c'est quoi ton nom ?",
        "tu es basé sur chatgpt ?", "tu es une copie de chatgpt ?", "tu es un modèle d'openai ?", "tu es gemini ?",
        "tu es mistral ?", "t'es claude ?", "tu viens de chatgpt ?", "tu es un gpt ?",
        "présente-toi rapidement", "présente-toi en deux mots", "fais une présentation de toi", "dis-moi qui tu es",
    ],
    "createur": [
        "qui t'a créé ?", "qui t'a fait ?", "qui est ton créateur ?", "qui t'a programmé ?", "qui t'a développé ?",
        "qui t'a entraîné ?", "tu viens d'où ?", "qui t'a conçu ?", "de quelle entreprise viens-tu ?",
        "qui est derrière toi ?", "qui t'a inventé ?", "qui a codé ce modèle ?", "qui est ton auteur ?",
        "qui t'a fabriqué ?", "tu as été créé par qui ?", "qui possède ce modèle ?", "c'est qui ton papa ?",
        "qui t'a appris à parler ?", "tu appartiens à quelle société ?", "qui t'a mis au point ?",
        "d'où tu viens ?", "tu sors d'où ?", "d'où viens-tu ?", "tu viens de quelle entreprise ?", "tu es d'où ?",
        "présente ton créateur", "c'est qui qui t'a fait ?",
    ],
    "capacites": [
        "que sais-tu faire ?", "tu sais faire quoi ?", "qu'est-ce que tu peux faire ?", "à quoi tu sers ?",
        "comment tu peux m'aider ?", "tu sers à quoi ?", "quelles sont tes capacités ?", "tu sais calculer ?",
        "tu sais lire ?", "tu parles anglais ?", "tu as accès à internet ?", "tu connais quoi ?",
        "tu peux faire quoi pour moi ?", "tu es fort en quoi ?", "quelles sont tes limites ?", "tu sais tout ?",
        "tu apprends de nos conversations ?", "tu te souviens de moi ?",
    ],
    "heure": [
        "quelle heure est-il ?", "il est quelle heure ?", "tu as l'heure ?", "quelle heure il est ?", "l'heure ?",
        "on est quel jour ?", "quel jour sommes-nous ?", "c'est quoi la date aujourd'hui ?", "on est le combien ?",
        "quelle est la date du jour ?", "on est en quelle année ?", "quel jour on est ?", "il est quelle heure là ?",
        "tu peux me donner l'heure ?", "nous sommes quel jour ?", "quelle date sommes-nous ?",
    ],
    "meteo": [
        "quel temps fait-il ?", "il fait beau ?", "il va pleuvoir aujourd'hui ?", "la météo de demain ?",
        "il fait chaud dehors ?", "il fait froid ?", "quelle température fait-il ?", "il neige ?",
        "c'est quoi la météo ?", "il fera beau ce week-end ?", "quelles sont les actualités ?",
        "quoi de neuf dans le monde ?", "c'est quoi les infos du jour ?", "qui a gagné le match hier ?",
        "quel est le cours du bitcoin ?", "les résultats des élections ?", "il pleut à Paris ?",
    ],
    "liste": [
        "cite-moi 3 capitales", "donne-moi 5 capitales", "cite moi trois pays", "donne-moi quatre pays d'Europe",
        "cite 3 capitales européennes", "donne moi 5 pays", "liste-moi des capitales", "quelques capitales ?",
        "cite-moi des pays d'Afrique", "donne-moi 3 pays d'Asie", "nomme 4 capitales", "cite 5 pays",
        "tu peux me citer 3 capitales ?", "donne-moi une liste de pays", "cite-moi deux capitales",
        "énumère 3 capitales d'Afrique", "cite six pays d'Amérique", "des exemples de capitales ?",
    ],
    "relance": [
        "et de la France ?", "et en Italie ?", "et lui ?", "pourquoi ?", "comment ça ?", "et après ?",
        "tu peux développer ?", "explique la 2", "décris-moi la recette 3", "la deuxième", "et la dernière ?",
        "raconte la suite", "continue", "dis-m'en plus", "plus de détails", "c'est-à-dire ?", "et sa population ?",
        "où est-il né ?", "et elle ?", "et en quelle année ?", "et la capitale ?", "développe le premier point",
        "tu peux préciser ?", "et ensuite ?", "détaille la 1", "la troisième idée ?",
    ],
}
# Des demandes ordinaires, écrites à la main, pour que « autre » ne soit pas
# seulement des questions de faits.
AUTRES_ECRITS = [
    "donne-moi une idée de repas", "raconte-moi une histoire", "écris un poème sur la mer", "comment faire une omelette ?",
    "explique-moi la photosynthèse", "pourquoi le ciel est bleu ?", "donne-moi un conseil pour dormir",
    "comment apprendre l'anglais ?", "écris une lettre de motivation", "résume la révolution française",
    "c'est quoi un trou noir ?", "comment fonctionne un moteur ?", "que faire à Paris ?", "une blague ?",
    "donne-moi 3 recettes", "propose-moi un prénom pour un chat", "comment rester motivé ?",
    "traduis bonjour en anglais", "c'est quoi l'intelligence artificielle ?", "aide-moi à écrire un message",
    # « donne moi une recette » était pris pour une relance (« décris-moi la recette 3 ») : Carl
    # voyait l'échange précédent, un simple « ok merci ».
    "donne moi une recette", "donne-moi une idée", "donne moi un conseil", "une recette ?", "un conseil ?",
    "une idée de sortie ?", "propose-moi une activité", "donne-moi un exemple de phrase", "trouve-moi un titre",
    "donne moi une astuce pour ranger", "une blague ?", "un exercice de maths ?",
    # Contrastes : des questions de faits qui ressemblent à d'autres classes.
    "qui a fondé Apple ?", "qui a créé Facebook ?", "qui a inventé l'imprimerie ?", "qui a construit la tour Eiffel ?",
    "qui a écrit Harry Potter ?", "qui a fondé Google ?", "qui est le créateur de Mickey ?", "qui a découvert l'Amérique ?",
    "à quelle température fond la glace ?", "quelle est la température du soleil ?", "à quelle température cuire un poulet ?",
    "quelle est la température normale du corps ?", "pourquoi fait-il froid en hiver ?", "comment se forme la pluie ?",
    "quelle est la capitale de la Suède ?", "c'est quoi la capitale du Pérou ?", "capitale du kenya ?",
    "quel est le plus grand pays du monde ?", "combien y a-t-il de pays en Europe ?", "d'où vient le chocolat ?",
    "d'où vient le mot café ?", "à qui appartient Instagram ?", "qui est le président de la France ?",
    "le soleil est-il chaud ?", "la glace est-elle froide ?", "pourquoi le feu brûle ?", "l'eau chaude monte-t-elle ?",
    "est-ce que la neige fond au soleil ?", "pourquoi fait-il nuit ?",
]


# Des mots de politesse qu'on ne peut pas reformuler : « salut », « merci ».
POLITESSE = set("bonjour salut hello coucou bonsoir merci carl".split())


def norm(texte: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", sans_accents(texte.lower())).strip()


def sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def variantes(message: str, r: random.Random) -> list[str]:
    """Le message tel qu'on le tape : majuscule, point d'exclamation, sans accents."""
    v = [message, message[:1].upper() + message[1:]]
    if r.random() < 0.5:
        v.append(message.rstrip(" ?!") + r.choice([" !", " ?", ""]))
    if r.random() < 0.4:
        v.append(sans_accents(message))
    return list(dict.fromkeys(v))


def exemples(graine: int = 3) -> list[tuple[str, str]]:
    """(message, classe) pour l'entraînement."""
    from examen import TESTS
    from examen_conversation import CONVERSATIONS
    from examen_style import CLOTURES, OUVERTES, OUVERTURES

    interdits = {norm(q) for q in list(TESTS) + OUVERTURES + CLOTURES + OUVERTES}
    interdits |= {norm(q) for conv in CONVERSATIONS for q, _ in conv}
    interdits -= {i for i in interdits if set(i.split()) <= POLITESSE}  # « salut », « merci » : universels
    r = random.Random(graine)
    paires = [(v, classe) for classe, messages in EXEMPLES.items() for m in messages for v in variantes(m, r)]
    autres = list(AUTRES_ECRITS)
    for fichier, n in [("faits", 250), ("calculs", 80), ("conversations_outils_style", 250)]:
        chemin = Path(f"data/sft/{fichier}.json")
        if chemin.exists():
            convs = json.loads(chemin.read_text(encoding="utf-8"))
            autres += [c[0]["content"] for c in r.sample(convs, min(n, len(convs)))]
    paires += [(v, "autre") for m in autres for v in variantes(m, r)]
    return [(m, c) for m, c in paires if norm(m) not in interdits]
